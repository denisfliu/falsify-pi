"""End-to-end smoke test runner.

Usage::

    PYTHONPATH=src python -m falsify.cli.smoke_test \\
        --config configs/falsification/smoke.yaml

The default config drives a `MockStraightLine` policy against the left_gate
scene. With ``--policy configs/policies/mock_noisy.yaml`` it swaps in the
noisy mock that's intended to trip the (Phase-4) failure detector once it
lands.

No GPU / no FiGS imports are exercised for mock policies because they
declare empty ``required_modalities`` — the sensor rig contains only a
`StateSensor`. Run with a VLA policy to engage the gsplat renderer.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from falsify.geometry import Point
from falsify.io import load_yaml, build_frame_graph
from falsify.orchestrator import EpisodeConfig, run_episode
from falsify.policy import (
    MockNoisy, MockNoisyConfig,
    MockStraightLine, MockStraightLineConfig,
)
from falsify.perturbations import (
    ImageBlur, ImageGaussianNoise, PerturbationSuite,
    PositionBias, PositionNoise, StateNoise, VelocityScale,
)
from falsify.recovery import RecoveryConfig, SplatNavPlanner
from falsify.safety import (
    BoundsCriterion, DroneBody, FailureDetector,
    MissGateCriterion, PointCloudCollisionCriterion,
    TiltCriterion, VelocityCriterion,
)
from falsify.visualization import dump_episode, html_replay, read_ply


_PERT_OBSERVATION = {
    "ImageGaussianNoise": ImageGaussianNoise,
    "ImageBlur": ImageBlur,
    "StateNoise": StateNoise,
}
_PERT_ACTION = {
    "PositionNoise": PositionNoise,
    "PositionBias": PositionBias,
    "VelocityScale": VelocityScale,
}


def _build_perturbations_factory(top_cfg: dict):
    pert_path = top_cfg.get("perturbations")
    if not pert_path:
        return None
    repo_root = Path(__file__).resolve().parents[3]
    cfg_path = Path(pert_path)
    if not cfg_path.is_absolute():
        cfg_path = repo_root / cfg_path
    spec = load_yaml(cfg_path)

    def factory(_frame_graph, _episode_cfg):
        obs_perts = [_PERT_OBSERVATION[e["type"]](**{k: v for k, v in e.items() if k != "type"})
                     for e in spec.get("observation", [])]
        act_perts = [_PERT_ACTION[e["type"]](**{k: v for k, v in e.items() if k != "type"})
                     for e in spec.get("action", [])]
        return PerturbationSuite(
            observation=obs_perts,
            action=act_perts,
            seed=spec.get("seed"),
        )
    return factory


def _policy_factory_from_yaml(policy_cfg: dict, *, scene_cfg=None, scene_dir=None):
    """Build a policy factory closure from a parsed policy YAML.

    Some policy kinds (``pi_gateway``) need a ``FrameGraph`` for NED↔MOCAP
    conversions; those are passed through ``scene_cfg`` / ``scene_dir``. Mock
    kinds ignore both.
    """
    kind = policy_cfg["type"]
    if kind == "mock_straight_line":
        def factory(goal: Point, _episode_cfg):
            return MockStraightLine(MockStraightLineConfig(
                goal=goal,
                speed=float(policy_cfg.get("speed", 1.0)),
                horizon_s=float(policy_cfg.get("horizon_s", 5.0)),
                n_waypoints=int(policy_cfg.get("n_waypoints", 50)),
            ))
        return factory
    if kind == "mock_noisy":
        def factory(goal: Point, _episode_cfg):
            return MockNoisy(MockNoisyConfig(
                goal=goal,
                speed=float(policy_cfg.get("speed", 1.0)),
                horizon_s=float(policy_cfg.get("horizon_s", 5.0)),
                n_waypoints=int(policy_cfg.get("n_waypoints", 50)),
                position_noise_std=float(policy_cfg.get("position_noise_std", 0.05)),
                seed=policy_cfg.get("seed"),
            ))
        return factory
    if kind == "pi_gateway":
        if scene_cfg is None or scene_dir is None:
            raise ValueError(
                "pi_gateway policy needs scene_cfg + scene_dir to build a FrameGraph"
            )
        # Lazy import — keeps mock-only runs from importing pi_inference_client.
        from falsify.policy import PiGatewayConfig, PiGatewayPolicy

        record_dir_arg = policy_cfg.get("record_dir")
        if record_dir_arg is not None:
            record_dir_arg = Path(record_dir_arg)
        cfg = PiGatewayConfig(
            gateway_url=policy_cfg["gateway_url"],
            api_key=policy_cfg.get("api_key", ""),
            execute_chunk_size=int(policy_cfg.get("execute_chunk_size", 25)),
            prompt=policy_cfg.get("prompt", ""),
            hz=int(policy_cfg.get("hz", 30)),
            state_dim=int(policy_cfg.get("state_dim", 7)),
            action_dim=int(policy_cfg.get("action_dim", 7)),
            action_pos_slice=tuple(policy_cfg.get("action_pos_slice", (0, 3))),
            action_yaw_index=policy_cfg.get("action_yaw_index", 3),
            camera_map=dict(policy_cfg.get("camera_map") or {}),
            state_key=policy_cfg.get("state_key", "observation/state"),
            server_frame=policy_cfg.get("server_frame", "mocap"),
            use_rtc=bool(policy_cfg.get("use_rtc", False)),
            traceability=dict(policy_cfg.get("traceability") or {}),
            record_dir=record_dir_arg,
        )

        def factory(_goal: Point, _episode_cfg):
            fg = build_frame_graph(scene_cfg, base_path=scene_dir)
            return PiGatewayPolicy(cfg, fg)
        return factory
    raise ValueError(f"unknown policy type {kind!r} in policy config")


class _StubLineBackend:
    """Stub recovery backend that returns a straight line.

    Used when ``--stub-recovery`` is set so the smoke test can exercise the
    recovery code path without needing splatnav/torch/CUDA.
    """
    def __init__(self, n: int = 30):
        self.n = n
    def generate_path(self, x0_ns, xf_ns):
        return np.linspace(np.asarray(x0_ns), np.asarray(xf_ns), self.n)


def _build_recovery_factory(scene_cfg: dict, scene_dir, *, stub: bool):
    def factory(frame_graph, recovery_cfg: dict):
        cfg = RecoveryConfig(
            bounds_lower_ned=recovery_cfg.get("bounds_lower_ned", [-2.5, -2.5, 0.2]),
            bounds_upper_ned=recovery_cfg.get("bounds_upper_ned", [2.5, 2.5, 3.0]),
            radius_m=float(recovery_cfg.get("radius_m", 0.05)),
            vmax=float(recovery_cfg.get("vmax", 2.0)),
            amax=float(recovery_cfg.get("amax", 3.0)),
            voxel_resolution=int(recovery_cfg.get("voxel_resolution", 100)),
        )
        if stub:
            return SplatNavPlanner(cfg, frame_graph, backend=_StubLineBackend())
        gsplat_path = scene_dir / scene_cfg["gsplat_config_yml"]
        return SplatNavPlanner(cfg, frame_graph, gsplat_config_path=gsplat_path)
    return factory


def _build_detector_factory(scene_cfg: dict, scene_dir: Path):
    """Build a `(frame_graph, safety_cfg) -> FailureDetector` factory.

    The closure captures the scene context so the new criteria — which
    need scene-side data (object PLY paths, gate aperture) — can resolve
    everything they need without changing the orchestrator's factory
    signature.

    Recognised ``safety_cfg`` keys (all optional)::

        bounds_frame, bounds_lower, bounds_upper      # BoundsCriterion
        max_speed                                     # VelocityCriterion
        max_tilt_rad                                  # TiltCriterion
        drone_body:                                   # required by collision
          half_extents: [hx, hy, hz]                  #   in body FRD
          center_offset_body: [ox, oy, oz]            #   optional
        collision:                                    # PointCloudCollisionCriterion
          enabled: true                               #   default false
          gate_objects: [gate]                        #   names from scene_objects
          other_objects: [table]
        miss_gate:                                    # MissGateCriterion
          corners_frame: mocap
          corners: [[x,y,z], [x,y,z], [x,y,z], [x,y,z]]
          margin_m: 0.0                               # optional inward shrink
    """
    def factory(frame_graph, safety_cfg: dict):
        bounds_frame_name = safety_cfg.get("bounds_frame", "ned")
        bounds_frame = frame_graph.frame(bounds_frame_name)
        bounds_lower = safety_cfg.get("bounds_lower", [-2.5, -2.5, 0.2])
        bounds_upper = safety_cfg.get("bounds_upper", [2.5, 2.5, 3.0])
        criteria: list = [
            BoundsCriterion(
                lower=Point.of(*bounds_lower, bounds_frame),
                upper=Point.of(*bounds_upper, bounds_frame),
            ),
            VelocityCriterion(max_speed=float(safety_cfg.get("max_speed", 5.0))),
            TiltCriterion(max_tilt_rad=float(safety_cfg.get("max_tilt_rad", 1.2))),
        ]

        collision_cfg = safety_cfg.get("collision")
        if collision_cfg and collision_cfg.get("enabled", False):
            criteria.append(_build_collision_criterion(
                scene_cfg, scene_dir, frame_graph,
                safety_cfg=safety_cfg, collision_cfg=collision_cfg,
            ))

        miss_gate_cfg = safety_cfg.get("miss_gate")
        if miss_gate_cfg:
            criteria.append(_build_miss_gate_criterion(miss_gate_cfg))

        return FailureDetector(criteria, frame_graph)
    return factory


def _drone_body_from_cfg(safety_cfg: dict) -> DroneBody:
    body_cfg = safety_cfg.get("drone_body")
    if not body_cfg:
        raise ValueError(
            "safety.collision is enabled but safety.drone_body is not set "
            "— declare drone_body.half_extents (in body FRD frame)."
        )
    half_extents = np.asarray(body_cfg["half_extents"], dtype=np.float64)
    offset = np.asarray(
        body_cfg.get("center_offset_body", [0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    return DroneBody(half_extents=half_extents, center_offset_body=offset)


def _build_collision_criterion(
    scene_cfg: dict,
    scene_dir: Path,
    frame_graph,
    *,
    safety_cfg: dict,
    collision_cfg: dict,
) -> PointCloudCollisionCriterion:
    """Load `scene_objects` PLYs into NED-frame labeled clouds.

    The collision criterion uses the *extracted* per-object PLYs the scene
    YAML already declares (under ``scene_objects:``) as the source of
    truth for what the drone can collide with. Each scene_object is
    classified ``"gate"`` (if its name is in ``gate_objects``) or
    ``"other"`` (catch-all for anything else listed under
    ``other_objects``). The PLYs are loaded once at factory build time,
    transformed into NED, and frozen — no per-step file IO.
    """
    body = _drone_body_from_cfg(safety_cfg)
    gate_objects = set(collision_cfg.get("gate_objects", ["gate"]))
    other_objects = set(collision_cfg.get("other_objects", []))
    requested = gate_objects | other_objects

    scene_objects = scene_cfg.get("scene_objects") or []
    by_name = {entry["name"]: entry for entry in scene_objects}
    missing = requested - set(by_name)
    if missing:
        raise ValueError(
            f"safety.collision references scene_objects {sorted(missing)} "
            f"that are not declared in the scene YAML "
            f"(found: {sorted(by_name)})"
        )

    # Stack gate-object points and other-object points separately,
    # converting from each PLY's authored frame to NED via the FrameGraph.
    gate_pts: list[np.ndarray] = []
    other_pts: list[np.ndarray] = []
    for name, entry in by_name.items():
        if name not in requested:
            continue
        ply_path = Path(entry["ply"])
        if not ply_path.is_absolute():
            ply_path = (scene_dir / ply_path).resolve()
        frame = frame_graph.frame(entry["frame"])
        pc = read_ply(ply_path, frame)
        # Convert the point cloud to NED in bulk.
        pc_ned = frame_graph.convert(pc, to="ned")
        xyz = np.asarray(pc_ned.points, dtype=np.float64)
        if name in gate_objects:
            gate_pts.append(xyz)
        else:
            other_pts.append(xyz)

    labeled: dict[str, np.ndarray] = {}
    if gate_pts:
        labeled["gate"] = np.vstack(gate_pts)
    if other_pts:
        labeled["other"] = np.vstack(other_pts)
    if not labeled:
        raise ValueError(
            "safety.collision is enabled but gate_objects and other_objects "
            "are both empty — nothing to collide with."
        )
    return PointCloudCollisionCriterion(body, labeled_clouds=labeled)


def _build_miss_gate_criterion(miss_gate_cfg: dict) -> MissGateCriterion:
    corners = miss_gate_cfg.get("corners")
    if corners is None or len(corners) != 4:
        raise ValueError(
            "safety.miss_gate.corners must be a list of 4 (x, y, z) points "
            "ordered to trace the aperture rectangle"
        )
    corners_arr = np.asarray(corners, dtype=np.float64)
    return MissGateCriterion(
        corners_arr,
        frame_name=miss_gate_cfg.get("corners_frame", "mocap"),
        margin_m=float(miss_gate_cfg.get("margin_m", 0.0)),
    )


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run one smoke-test falsification episode.")
    ap.add_argument("--config", required=True, help="Top-level falsification YAML")
    ap.add_argument("--policy", help="Override the policy YAML referenced in --config")
    ap.add_argument("--scene", help="Override the scene YAML referenced in --config")
    ap.add_argument("--frame", help="Override the drone-frame YAML")
    ap.add_argument("--out", help="Output directory (default: from config)")
    ap.add_argument("--no-detector", action="store_true", help="Skip failure detection")
    ap.add_argument("--no-visualize", action="store_true", help="Skip ply+html dumps")
    ap.add_argument("--no-recovery", action="store_true", help="Skip recovery planning even on failure")
    ap.add_argument(
        "--stub-recovery", action="store_true",
        help="Use the straight-line stub recovery backend instead of SplatNav (no GPU needed)",
    )
    args = ap.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[3]
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = repo_root / cfg_path
    top = load_yaml(cfg_path)

    scene_path = repo_root / (args.scene or top["scene"])
    frame_path = repo_root / (args.frame or top["frame"])
    policy_path = repo_root / (args.policy or top["policy"])

    ep_cfg = EpisodeConfig.from_yaml(scene_path, frame_path, policy_path)
    # Stitch sim-loop settings from the top-level config into episode_cfg.
    ep_cfg.episode_cfg.setdefault("hz", top.get("hz", 10))
    ep_cfg.episode_cfg.setdefault("policy_hz", top.get("policy_hz", 1))
    ep_cfg.episode_cfg.setdefault("horizon_s", top.get("horizon_s", 10.0))
    if "safety" in top:
        ep_cfg.episode_cfg["safety"] = top["safety"]

    policy_factory = _policy_factory_from_yaml(
        load_yaml(policy_path),
        scene_cfg=ep_cfg.scene_cfg,
        scene_dir=ep_cfg.scene_cfg_dir,
    )

    detector_factory = None
    if not args.no_detector:
        detector_factory = _build_detector_factory(
            ep_cfg.scene_cfg, ep_cfg.scene_cfg_dir,
        )
    recovery_factory = None
    if not args.no_recovery:
        recovery_factory = _build_recovery_factory(
            ep_cfg.scene_cfg, ep_cfg.scene_cfg_dir, stub=args.stub_recovery,
        )
    perturbations_factory = _build_perturbations_factory(top)
    t0 = time.time()
    ep = run_episode(
        ep_cfg,
        policy_factory=policy_factory,
        detector_factory=detector_factory,
        recovery_factory=recovery_factory,
        perturbations_factory=perturbations_factory,
    )
    wall = time.time() - t0

    print(ep.summary())
    print(f"  wall: {wall:.2f}s")

    # Persist a minimal episode summary.
    out_root = Path(args.out or top.get("output_dir", "runs"))
    if not out_root.is_absolute():
        out_root = repo_root / out_root
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    traj = ep.trace.trajectory()
    summary = {
        "scene": str(scene_path),
        "frame": str(frame_path),
        "policy": str(policy_path),
        "n_states": len(ep.trace.states),
        "n_policy_queries": len(ep.trace.policy_outputs),
        "start_ned": ep.trace.states[0].pos.xyz.tolist(),
        "end_ned": ep.trace.states[-1].pos.xyz.tolist(),
        "goal_ned": ep.goal.xyz.tolist() if ep.goal is not None else None,
        "goal_frame": ep.goal.frame.name if ep.goal is not None else None,
        "trajectory_frame": traj.frame.name,
        "wall_seconds": wall,
        "succeeded": ep.succeeded,
    }
    if ep.failure is not None:
        summary["failure"] = {
            "type": ep.failure.failure_type.name,
            "description": ep.failure.description,
            "criterion": ep.failure.criterion_name,
            "step": ep.failure.failure_step,
            "last_safe_step": ep.failure.last_safe_step,
        }
    if ep.recovery_trajectory is not None:
        summary["recovery"] = {
            "n_waypoints": len(ep.recovery_trajectory),
            "frame": ep.recovery_trajectory.frame.name,
            "start_ned": ep.recovery_trajectory.positions[0].tolist(),
            "end_ned": ep.recovery_trajectory.positions[-1].tolist(),
        }
    if ep.metadata.get("perturbations") is not None:
        summary["perturbations"] = ep.metadata["perturbations"]
    (out_dir / "episode_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  saved: {out_dir / 'episode_summary.json'}")

    if not args.no_visualize:
        frame_graph = build_frame_graph(ep_cfg.scene_cfg, base_path=ep_cfg.scene_cfg_dir)
        plys = dump_episode(ep, frame_graph, out_dir / "frames")
        for entity, paths in plys.items():
            for fname, p in paths.items():
                print(f"  ply:   {entity}/{fname} -> {p}")
        html_path = html_replay(ep, frame_graph, out_dir / "episode.html", view_frame="ned")
        if html_path:
            print(f"  html:  {html_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
