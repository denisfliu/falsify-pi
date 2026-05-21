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
    GateRigidPerturbation, ImageBlur, ImageGaussianNoise, PerturbationSuite,
    PositionBias, PositionNoise, StateNoise, VelocityScale,
)
from falsify.recovery import RecoveryConfig, SplatNavPlanner
from falsify.safety import (
    BoundsCriterion, DroneBody, FailureDetector,
    MissGateCriterion, OrderedMissGateCriterion, PointCloudCollisionCriterion,
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
_PERT_ENVIRONMENT = {
    "GateRigidPerturbation": GateRigidPerturbation,
}


def build_perturbations_factory(pert_path):
    """Build a `PerturbationSuite` factory from a YAML path.

    YAML schema::

        seed: <int|null>
        observation: [{type: ImageGaussianNoise, camera: forward, std: 5.0}, ...]
        action:      [{type: PositionNoise, std: 0.02}, ...]
        environment: []   # stubs only until Splat-MOVER lands

    `pert_path` may be a `Path`, a string path, or `None` (returns `None`).
    Repo-relative paths resolve from the repo root.
    """
    if pert_path is None:
        return None
    repo_root = Path(__file__).resolve().parents[3]
    cfg_path = Path(pert_path)
    if not cfg_path.is_absolute():
        cfg_path = repo_root / cfg_path
    spec = load_yaml(cfg_path)

    def factory(_frame_graph, episode_cfg):
        # Environment perturbations need the scene YAML for gate_region etc.
        # The orchestrator stashes the parsed scene_cfg under
        # episode_cfg["scene_cfg"] before calling the factory.
        scene_cfg = (episode_cfg or {}).get("scene_cfg")
        # Trial-card replay path: per-perturbation absolute overrides,
        # keyed by `name` (or the type name as fallback).
        overrides = (episode_cfg or {}).get("overrides") or {}
        obs_perts = [_PERT_OBSERVATION[e["type"]](**{k: v for k, v in e.items() if k != "type"})
                     for e in spec.get("observation", [])]
        act_perts = [_PERT_ACTION[e["type"]](**{k: v for k, v in e.items() if k != "type"})
                     for e in spec.get("action", [])]
        env_perts = []
        for e in spec.get("environment", []):
            kls = _PERT_ENVIRONMENT[e["type"]]
            kwargs = {k: v for k, v in e.items() if k != "type"}
            # Scene-aware perturbations (the only kind we currently ship)
            # get the parsed scene_cfg injected so their __init__ can read
            # gate_region etc. Future scene-agnostic env perturbations
            # would simply omit `scene_cfg` from their dataclass.
            from dataclasses import fields as _fields
            if any(f.name == "scene_cfg" for f in _fields(kls)) and scene_cfg is not None:
                kwargs.setdefault("scene_cfg", scene_cfg)
            pert = kls(**kwargs)
            # If the trial card carries absolute deltas for this perturbation,
            # switch the instance into replay mode (currently supported by
            # GateRigidPerturbation via `set_absolute_deltas`).
            key = kwargs.get("name") or e["type"]
            override = overrides.get(key)
            if override is not None and hasattr(pert, "set_absolute_deltas"):
                pert.set_absolute_deltas(
                    delta_xyz=override["delta_xyz"],
                    delta_yaw_rad=override["delta_yaw_rad"],
                )
            env_perts.append(pert)
        return PerturbationSuite(
            observation=obs_perts,
            action=act_perts,
            environment=env_perts,
            seed=spec.get("seed"),
        )
    return factory


def _build_perturbations_factory(top_cfg: dict):
    return build_perturbations_factory(top_cfg.get("perturbations"))


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
            image_size=policy_cfg.get("image_size"),
            channel_order=str(policy_cfg.get("channel_order", "RGB")),
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

        # Gate-perturbation deltas (injected by the orchestrator). When
        # present, the collision PLYs + aperture corners are rigidly
        # transformed by the same Δxyz / Δyaw about the gate anchor so
        # they stay aligned with the moved gate Gaussians.
        gate_deltas = safety_cfg.get("_gate_deltas")

        collision_cfg = safety_cfg.get("collision")
        if collision_cfg and collision_cfg.get("enabled", False):
            criteria.append(_build_collision_criterion(
                scene_cfg, scene_dir, frame_graph,
                safety_cfg=safety_cfg, collision_cfg=collision_cfg,
                gate_deltas=gate_deltas,
            ))

        miss_gate_cfg = safety_cfg.get("miss_gate")
        if miss_gate_cfg:
            criteria.append(_build_miss_gate_criterion(
                miss_gate_cfg, gate_deltas=gate_deltas, scene_cfg=scene_cfg,
            ))

        ordered_cfg = safety_cfg.get("ordered_miss_gate")
        if ordered_cfg:
            criteria.append(_build_ordered_miss_gate_criterion(
                ordered_cfg, scene_cfg=scene_cfg,
            ))

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
    gate_deltas: Optional[dict] = None,
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
    # Two transforms layer on each gate-labelled PLY before NED conversion,
    # in the order the renderer applies them to the Gaussians themselves:
    #   1. Static `scene_edits` (e.g. center_gate.yaml's `move_gate` edit
    #      that translates the gate from its authored location to the
    #      center). Loaded via `load_scene_edits` and applied via
    #      `apply_edits_to_scene_object` — the same visualizer-side path
    #      `inspect_scene_plotly` / `visualize_waypoints` use, so what we
    #      collide against agrees with what the inspector shows.
    #   2. Per-episode `gate_deltas` (the `GateRigidPerturbation` Δ for
    #      this trial, on top of the static edit).
    # Without (1), center_gate's collision cloud sits at the *un-edited*
    # left_gate footprint while the renderer shows the gate at its new
    # center pose — the drone fires phantom COLLISION_GATE in empty space.
    from falsify.geometry import PointCloud
    from falsify.sim.scene_edits import apply_edits_to_scene_object, load_scene_edits
    scene_edits = load_scene_edits(scene_cfg)
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
        pts = np.asarray(pc.points, dtype=np.float64)
        # (1) Static scene_edits — applies to every named scene_object in
        # `applies_to_scene_objects`. No-op when the scene declares no
        # edits (left_gate, right_gate).
        if scene_edits:
            pts = apply_edits_to_scene_object(name, pts, scene_edits, frame_graph)
        # (2) Per-episode gate perturbation — gate-labelled clouds only.
        if name in gate_objects and gate_deltas is not None:
            # The deltas are authored in MOCAP. PLYs live in their own
            # `entry["frame"]` (typically MOCAP). If it isn't MOCAP we'd
            # need a frame conversion before+after — we don't support
            # non-MOCAP gate-collision PLYs yet (none of the shipped
            # scenes use one) so error out loudly.
            if entry["frame"] != "mocap":
                raise ValueError(
                    f"gate perturbation requires gate scene_object {name!r} "
                    f"to live in 'mocap'; got {entry['frame']!r}"
                )
            pts = _apply_gate_rigid_transform(pts, gate_deltas)
        pc = PointCloud(points=pts, frame=frame)
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


def _apply_gate_rigid_transform(
    points: np.ndarray,
    gate_deltas: dict,
) -> np.ndarray:
    """Apply a (Δxyz, Δyaw about anchor) rigid transform in MOCAP to ``points``.

    Mirrors `GateRigidPerturbation._build_edit`: rotation is about MOCAP +z
    around the gate anchor, then translation by Δxyz. Used by the safety
    layer to move aperture corners + collision PLYs onto the perturbed
    gate without re-deriving them from the scene.
    """
    pts = np.asarray(points, dtype=np.float64)
    anchor = np.asarray(gate_deltas["anchor_mocap"], dtype=np.float64)
    dxyz = np.asarray(gate_deltas["delta_xyz_mocap"], dtype=np.float64)
    dyaw = float(gate_deltas["delta_yaw_rad"])
    c, s = np.cos(dyaw), np.sin(dyaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return (pts - anchor) @ R.T + anchor + dxyz


def _build_miss_gate_criterion(
    miss_gate_cfg: dict,
    *,
    gate_deltas: Optional[dict] = None,
    scene_cfg: Optional[dict] = None,
) -> MissGateCriterion:
    corners = miss_gate_cfg.get("corners")
    if corners is None or len(corners) != 4:
        raise ValueError(
            "safety.miss_gate.corners must be a list of 4 (x, y, z) points "
            "ordered to trace the aperture rectangle"
        )
    corners_arr = np.asarray(corners, dtype=np.float64)
    # Goal + progress fields are optional — if absent, the criterion
    # degrades to the original geometric-only miss check.
    goal = miss_gate_cfg.get("goal_position")
    goal_arr = np.asarray(goal, dtype=np.float64) if goal is not None else None

    # If a gate-perturbation rigid transform is in play, move corners +
    # goal to match. We require corners_frame == mocap for this — the
    # deltas are authored in mocap; any other frame would need explicit
    # conversion which we haven't needed yet.
    if gate_deltas is not None:
        corners_frame = miss_gate_cfg.get("corners_frame", "mocap")
        if corners_frame != "mocap":
            raise ValueError(
                f"gate perturbation requires miss_gate.corners_frame=='mocap', "
                f"got {corners_frame!r}"
            )
        corners_arr = _apply_gate_rigid_transform(corners_arr, gate_deltas)
        # The goal stays *in place* — we only move the aperture, not the
        # hover-over-stuffed-animal goal. (The goal is task-defined, not
        # gate-defined.) Skip transforming goal.

    # Runtime AABB-transit latch: when the scene declares a `gate_region`,
    # pass its perturbation-aware AABB to the criterion so GOAL_REACHED
    # only fires once the drone has been inside the gate at least once.
    # Without this, an early-graze of the 10cm goal sphere on a still
    # in-progress approach could cut the rollout short — which the user
    # explicitly wants to avoid.
    transit_aabb_min = None
    transit_aabb_max = None
    if scene_cfg is not None and scene_cfg.get("gate_region"):
        region = scene_cfg["gate_region"]
        aabb_min_raw = np.asarray(region["aabb_min"], dtype=np.float64)
        aabb_max_raw = np.asarray(region["aabb_max"], dtype=np.float64)
        if gate_deltas is not None:
            corners = np.array([
                [aabb_min_raw[0], aabb_min_raw[1], aabb_min_raw[2]],
                [aabb_max_raw[0], aabb_min_raw[1], aabb_min_raw[2]],
                [aabb_min_raw[0], aabb_max_raw[1], aabb_min_raw[2]],
                [aabb_max_raw[0], aabb_max_raw[1], aabb_min_raw[2]],
                [aabb_min_raw[0], aabb_min_raw[1], aabb_max_raw[2]],
                [aabb_max_raw[0], aabb_min_raw[1], aabb_max_raw[2]],
                [aabb_min_raw[0], aabb_max_raw[1], aabb_max_raw[2]],
                [aabb_max_raw[0], aabb_max_raw[1], aabb_max_raw[2]],
            ])
            moved = _apply_gate_rigid_transform(corners, gate_deltas)
            transit_aabb_min = moved.min(axis=0)
            transit_aabb_max = moved.max(axis=0)
        else:
            transit_aabb_min = aabb_min_raw
            transit_aabb_max = aabb_max_raw

    half_extents = miss_gate_cfg.get("goal_tolerance_half_extents")
    half_extents_arr = (
        np.asarray(half_extents, dtype=np.float64)
        if half_extents is not None else None
    )

    return MissGateCriterion(
        corners_arr,
        frame_name=miss_gate_cfg.get("corners_frame", "mocap"),
        margin_m=float(miss_gate_cfg.get("margin_m", 0.0)),
        goal_position=goal_arr,
        goal_tolerance_m=float(miss_gate_cfg.get("goal_tolerance_m", 0.30)),
        goal_tolerance_half_extents=half_extents_arr,
        min_progress_window_s=(
            float(miss_gate_cfg["min_progress_window_s"])
            if miss_gate_cfg.get("min_progress_window_s") is not None else None
        ),
        min_progress_m=float(miss_gate_cfg.get("min_progress_m", 0.05)),
        eval_stop_mode=bool(miss_gate_cfg.get("eval_stop_mode", False)),
        transit_aabb_min=transit_aabb_min,
        transit_aabb_max=transit_aabb_max,
    )


def _build_ordered_miss_gate_criterion(
    cfg: dict,
    *,
    scene_cfg: Optional[dict] = None,
) -> OrderedMissGateCriterion:
    """YAML schema::

        ordered_miss_gate:
          corners_frame: mocap
          gates:
            - name: gate_1                     # informational only
              corners: [[x,y,z], ...]          # 4 corners (rectangle order)
            - name: gate_2
              corners: [[x,y,z], ...]
          margin_m: 0.0
          goal_position: [x, y, z]
          goal_tolerance_m: 0.30
          min_progress_window_s: 4.0
          min_progress_m: 0.05
          eval_stop_mode: false   # optional; True disables in-flight
                                  # plane-cross check and uses per-gate
                                  # AABB latches from `scene_cfg.gate_regions`

    When ``eval_stop_mode=True``, the criterion needs each gate's MOCAP
    AABB to drive the per-gate transit latch + goal-proximity stop. We
    pull those from ``scene_cfg.gate_regions`` (plural — list of two
    entries, in gate-1 / gate-2 order matching ``gates`` above).
    """
    gates = cfg.get("gates") or []
    if len(gates) != 2:
        raise ValueError(
            f"safety.ordered_miss_gate.gates must have exactly 2 entries; got {len(gates)}"
        )
    c1 = np.asarray(gates[0]["corners"], dtype=np.float64)
    c2 = np.asarray(gates[1]["corners"], dtype=np.float64)
    goal = cfg.get("goal_position")
    goal_arr = np.asarray(goal, dtype=np.float64) if goal is not None else None
    eval_stop = bool(cfg.get("eval_stop_mode", False))

    transit_aabb_1_min = transit_aabb_1_max = None
    transit_aabb_2_min = transit_aabb_2_max = None
    if eval_stop and scene_cfg is not None:
        regions = scene_cfg.get("gate_regions") or []
        if len(regions) != 2:
            raise ValueError(
                "ordered_miss_gate.eval_stop_mode requires "
                "scene_cfg.gate_regions to declare exactly 2 entries"
            )
        for r in regions:
            if r.get("aabb_frame", "mocap") != "mocap":
                raise NotImplementedError(
                    "gate_regions.aabb_frame == 'mocap' only"
                )
        transit_aabb_1_min = np.asarray(regions[0]["aabb_min"], dtype=np.float64)
        transit_aabb_1_max = np.asarray(regions[0]["aabb_max"], dtype=np.float64)
        transit_aabb_2_min = np.asarray(regions[1]["aabb_min"], dtype=np.float64)
        transit_aabb_2_max = np.asarray(regions[1]["aabb_max"], dtype=np.float64)

    return OrderedMissGateCriterion(
        c1, c2,
        frame_name=cfg.get("corners_frame", "mocap"),
        margin_m=float(cfg.get("margin_m", 0.0)),
        goal_position=goal_arr,
        goal_tolerance_m=float(cfg.get("goal_tolerance_m", 0.30)),
        min_progress_window_s=(
            float(cfg["min_progress_window_s"])
            if cfg.get("min_progress_window_s") is not None else None
        ),
        min_progress_m=float(cfg.get("min_progress_m", 0.05)),
        eval_stop_mode=eval_stop,
        transit_aabb_1_min=transit_aabb_1_min,
        transit_aabb_1_max=transit_aabb_1_max,
        transit_aabb_2_min=transit_aabb_2_min,
        transit_aabb_2_max=transit_aabb_2_max,
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
