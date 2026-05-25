"""Run one falsification episode driven by a real OpenPI VLA server.

End-to-end:

1. **Smoke checks first**: import `openpi_client`, `figs`, `nerfstudio`; open
   a websocket to the VLA server and shut it down. Fail fast if anything is
   missing.
2. Build the scene `FrameGraph`, the `GSplatRenderer`, the `VLAPolicy`, and
   a `SensorRig` covering the policy's required camera modalities.
3. Roll out one episode. The simulator re-queries the VLA after every
   `actions_per_chunk` waypoints (the SousVide-style chunk loop), and the
   replay integrator follows the chunk's positions + integrated yaw.
4. Write three output bundles under ``--out``:
   - ``frames/combined_<frame>.ply`` — the flown trajectory plus the scene
     object PLYs (gate, table), all in the same frame, one file per frame.
   - ``flythrough.mp4`` — forward-camera renders along the flown path.
   - ``vla_io/query_<NNNN>_step_<KKKKK>/`` — per-query inputs/outputs
     (images, state vector, raw actions). Written directly by the policy.

Frame contract: as everywhere else in falsify, NED is the simulator and
policy boundary frame; the renderer consumes NED-frame camera poses; the
VLA boundary is inside `VLAPolicy.observe`.

Example::

    PYTHONPATH=src .venv/bin/python -m falsify.cli.run_vla_episode \\
        --scene configs/scenes/left_gate.yaml \\
        --frame configs/frames/carl_dual.yaml \\
        --prompt "go through the gate and hover over the stuffed animal" \\
        --out runs/vla_$(date +%Y%m%d_%H%M%S) \\
        --hz 10 --actions-per-chunk 50 --horizon-s 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Smoke checks
# ---------------------------------------------------------------------------


def _smoke_imports(policy_backend: str = "openpi") -> None:
    if policy_backend == "openpi":
        print("[smoke] importing openpi_client …", end=" ", flush=True)
        import openpi_client  # noqa: F401
        print("ok")
    elif policy_backend == "pi_gateway":
        print("[smoke] importing pi_inference_client …", end=" ", flush=True)
        import pi_inference_client  # noqa: F401
        print("ok")
    else:
        raise ValueError(f"unknown policy_backend {policy_backend!r}")
    print("[smoke] importing figs            …", end=" ", flush=True)
    import figs  # noqa: F401
    print("ok")
    print("[smoke] importing nerfstudio      …", end=" ", flush=True)
    import nerfstudio  # noqa: F401
    print("ok")
    print("[smoke] importing torch (cuda?)   …", end=" ", flush=True)
    import torch
    has_cuda = torch.cuda.is_available()
    print(f"ok (cuda={has_cuda}, devices={torch.cuda.device_count()})")
    if not has_cuda:
        raise RuntimeError(
            "torch reports no CUDA device — the gsplat renderer needs a working "
            "local GPU. Check `nvidia-smi`."
        )


def _smoke_websocket(host: str, port: int, timeout: float = 5.0) -> None:
    """Probe the OpenPI websocket: open, ping with a tiny payload, close."""
    from falsify.policy.vla import _resolve_host
    resolved = _resolve_host(host)
    print(f"[smoke] handshake → ws://{resolved}:{port} …", end=" ", flush=True)
    from openpi_client import websocket_client_policy  # type: ignore
    t0 = time.time()
    client = websocket_client_policy.WebsocketClientPolicy(host=resolved, port=port)
    print(f"connected ({time.time() - t0:.2f}s)")
    if hasattr(client, "_ws"):
        try:
            client._ws.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Flythrough rendering
# ---------------------------------------------------------------------------


def _render_flythrough(
    states,
    renderer,
    cam_spec,
    body_to_world_fn,
    out_path: Path,
    *,
    fps: int,
    every: int = 1,
) -> Optional[Path]:
    """Render the forward camera at each (sub-sampled) state and encode mp4.

    ``every`` lets you stride through states for a cheaper render at the cost
    of smoothness.
    """
    import imageio.v2 as imageio
    frames = []
    n = len(states)
    sub = list(range(0, n, max(1, every)))
    print(f"[flythrough] rendering {len(sub)} of {n} states "
          f"(every={every}, fps={fps})…")
    for i, idx in enumerate(sub):
        state = states[idx]
        pose = body_to_world_fn(state, cam_spec.body_from_camera)
        rgb, _ = renderer.render(pose, cam_spec.intrinsics)
        frames.append(np.asarray(rgb, dtype=np.uint8))
        if (i + 1) % 20 == 0:
            print(f"  rendered {i + 1}/{len(sub)}")
    if not frames:
        print("[flythrough] no states to render")
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out_path, frames, fps=fps, quality=8)
    print(f"[flythrough] wrote {out_path}  ({len(frames)} frames)")
    return out_path


# ---------------------------------------------------------------------------
# Trajectory + scene PLYs
# ---------------------------------------------------------------------------


def _dump_trajectory_with_scene(
    traj_ned,
    fg,
    scene_cfg: dict,
    scene_dir: Path,
    out_dir: Path,
    *,
    target_frames=("ned", "mocap", "ns"),
    max_object_points: int = 8000,
) -> dict[str, Path]:
    """Same idea as `visualize_frames` but uses the actually-flown trajectory."""
    from falsify.visualization import (
        read_ply, stack_pointclouds, subsample, trajectory_to_pointcloud, write_ply,
    )

    def _resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (scene_dir / path).resolve()

    objects = []
    for entry in scene_cfg.get("scene_objects", []):
        cloud = read_ply(_resolve(entry["ply"]), fg.frame(entry["frame"]))
        color = tuple(entry.get("color", (0.5, 0.5, 0.5)))
        objects.append((entry["name"], cloud, color))

    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    trajectory_color = (1.0, 0.95, 0.20)

    for f in target_frames:
        traj_in_frame = fg.convert(traj_ned, to=f)
        traj_pc = trajectory_to_pointcloud(traj_in_frame, color=trajectory_color)
        combined = [traj_pc]
        for name, cloud, color in objects:
            in_frame = fg.convert(cloud, to=f)
            if max_object_points > 0:
                in_frame = subsample(in_frame, max_object_points)
            c = np.asarray(color, dtype=np.float64)
            tinted = type(in_frame)(
                points=in_frame.points,
                frame=in_frame.frame,
                colors=np.broadcast_to(c, in_frame.points.shape).copy(),
            )
            write_ply(tinted, out_dir / f"{name}_{f}.ply")
            combined.append(tinted)
        path = write_ply(stack_pointclouds(combined), out_dir / f"combined_{f}.ply")
        written[f] = path
    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


_DEFAULT_PROMPT_REGISTRY = Path("configs/prompts/atomic_dataset_prompts.yaml")


def _resolve_prompt(args: argparse.Namespace) -> str:
    """Return the literal prompt string. Either `--prompt` (verbatim) or
    `--prompt-name` (looked up in the registry) must be set."""
    import yaml
    if args.prompt is not None:
        return args.prompt
    registry_path = args.prompt_registry
    if not registry_path.is_absolute():
        repo_root = Path(__file__).resolve().parents[3]
        registry_path = repo_root / registry_path
    if not registry_path.is_file():
        raise SystemExit(f"--prompt-name set but registry not found: {registry_path}")
    registry = yaml.safe_load(registry_path.read_text()).get("prompts") or {}
    if args.prompt_name not in registry:
        known = ", ".join(sorted(registry)) or "(none)"
        raise SystemExit(
            f"--prompt-name {args.prompt_name!r} not in registry. "
            f"Known: {known}. "
            f"Refresh with `python scripts/dataset/build_prompt_registry.py`."
        )
    entry = registry[args.prompt_name]
    return str(entry["task"]) if isinstance(entry, dict) else str(entry)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--frame", required=True, type=Path,
                        help="Drone-frame YAML (e.g. configs/frames/carl_dual.yaml).")
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", type=str, default=None,
                              help="Literal prompt string to send to the VLA.")
    prompt_group.add_argument("--prompt-name", type=str, default=None,
                              help="Short name resolved against the prompt "
                                   "registry (default: "
                                   "configs/prompts/atomic_dataset_prompts.yaml). "
                                   "Keys come from `data/atomic_datasets/*/meta/"
                                   "tasks.jsonl`. Run "
                                   "`python scripts/dataset/build_prompt_registry.py "
                                   "--datasets-dir data/atomic_datasets` to "
                                   "refresh. Mutually exclusive with --prompt.")
    parser.add_argument("--prompt-registry", default=_DEFAULT_PROMPT_REGISTRY, type=Path,
                        help="Path to the prompt registry YAML used by "
                             "--prompt-name. Repo-relative paths resolve from "
                             "the repo root.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--host", default="moraband")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--hz", type=int, default=10)
    parser.add_argument("--actions-per-chunk", type=int, default=50)
    parser.add_argument("--horizon-s", type=float, default=30.0)
    parser.add_argument("--image-size", type=int, default=256,
                        help="Side length of the square image sent to the VLA.")
    parser.add_argument("--flythrough-every", type=int, default=1,
                        help="Render every Nth state for the flythrough mp4.")
    parser.add_argument("--flythrough-fps", type=int, default=10)
    parser.add_argument("--skip-handshake", action="store_true",
                        help="Skip the policy-server websocket handshake (useful "
                             "when smoke-testing the renderer offline).")
    parser.add_argument("--policy-config", type=Path, default=None,
                        help="Path to a policy YAML (e.g. configs/policies/pi_gateway/*.yaml). "
                             "When set, dispatches to that policy's type "
                             "(e.g. `pi_gateway`) instead of constructing a "
                             "`VLAPolicy` from --host/--port/--image-size. Ignores "
                             "those flags. Skips the openpi smoke handshake.")
    parser.add_argument("--safety", type=Path, default=None,
                        help="Optional safety YAML (same shape as the `safety:` "
                             "block in configs/falsification/smoke_collision.yaml). "
                             "When set, wires a FailureDetector that stops the "
                             "rollout on bounds / velocity / tilt / collision / "
                             "miss-gate. Without it, the rollout runs to "
                             "--horizon-s regardless.")
    parser.add_argument("--recovery", type=Path, default=None,
                        help="Optional recovery YAML. When set, engages the "
                             "FALSIFICATION pipeline: on a triggering failure "
                             "(MISS_GATE / COLLISION_* / OUT_OF_BOUNDS by "
                             "default), the configured course-based MPC "
                             "planner produces a recovery trajectory from "
                             "the last-safe state. Without it, the run is "
                             "EVALUATION-only — detector fires, rollout "
                             "stops, no recovery planned.")
    parser.add_argument("--perturbations", type=Path, default=None,
                        help="Optional perturbations YAML "
                             "(configs/perturbations/*.yaml). Wires a "
                             "PerturbationSuite over the action, observation, "
                             "and environment surfaces. Environment-surface "
                             "concrete impls are stubbed pending Splat-MOVER; "
                             "action + observation are live.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for the rollout RNG. Drives the scene's "
                             "`start_randomization` (and any future "
                             "noise/perturbation samplers). Unset → fresh "
                             "entropy each run; set → reproducible.")
    args = parser.parse_args(argv)
    # Resolve `--prompt-name` to its literal task string. Downstream code
    # treats `args.prompt` as the single source of truth.
    args.prompt = _resolve_prompt(args)

    # Decide policy backend up-front so smoke checks load the right deps.
    use_pi_gateway = args.policy_config is not None
    backend = "pi_gateway" if use_pi_gateway else "openpi"
    _smoke_imports(policy_backend=backend)
    if not args.skip_handshake and not use_pi_gateway:
        _smoke_websocket(args.host, args.port)

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Lazy heavy imports (after smoke checks) ----------------------
    from falsify.geometry import Point
    from falsify.io import build_frame_graph, load_yaml
    from falsify.orchestrator import EpisodeConfig, run_episode
    from falsify.policy.vla import VLAPolicy, VLAPolicyConfig
    from falsify.sensors import build_sensor_rig  # noqa: F401  (validation)
    from falsify.sim.poses import camera_to_world_pose
    from falsify.sim.renderer import GSplatRenderer
    from falsify.sensors.camera import make_camera_sensor_from_yaml  # noqa

    scene_cfg = load_yaml(args.scene)
    frame_cfg = load_yaml(args.frame)
    scene_dir = args.scene.parent

    # ---- Renderer (one instance, reused by every camera) --------------
    def _resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (scene_dir / path).resolve()

    fg = build_frame_graph(scene_cfg, base_path=scene_dir)
    renderer = GSplatRenderer.from_scene_cfg(scene_cfg, scene_dir=scene_dir)

    # ---- VLA-driven episode -------------------------------------------
    record_dir = out_dir / "vla_io"
    built_policies: list = []

    if use_pi_gateway:
        import hashlib
        from falsify.policy import PiGatewayConfig, PiGatewayPolicy
        policy_cfg = load_yaml(args.policy_config)
        if policy_cfg.get("type") != "pi_gateway":
            raise ValueError(
                f"--policy-config expects type: pi_gateway, got {policy_cfg.get('type')!r}"
            )
        policy_cfg_bytes = args.policy_config.read_bytes()
        policy_cfg_sha = hashlib.sha256(policy_cfg_bytes).hexdigest()
        print(
            f"[policy] {args.policy_config.name} sha256={policy_cfg_sha[:12]} "
            f"bridge_policy_id={policy_cfg.get('bridge_policy_id') or '(none)'} "
            f"bridge_admin_url={policy_cfg.get('bridge_admin_url') or '(none)'}"
        )
        pgcfg = PiGatewayConfig(
            gateway_url=policy_cfg["gateway_url"],
            api_key=policy_cfg.get("api_key", ""),
            execute_chunk_size=int(policy_cfg.get("execute_chunk_size", 25)),
            # CLI --prompt wins: the user explicitly supplied it (required arg),
            # so the YAML's prompt is just a default for un-flagged invocations
            # (and a traceability anchor — what the YAML was built around).
            prompt=args.prompt or policy_cfg.get("prompt", ""),
            hz=int(policy_cfg.get("hz", args.hz)),
            state_dim=int(policy_cfg.get("state_dim", 7)),
            action_dim=int(policy_cfg.get("action_dim", 7)),
            action_pos_slice=tuple(policy_cfg.get("action_pos_slice", (0, 3))),
            action_yaw_index=policy_cfg.get("action_yaw_index", 3),
            camera_map=dict(policy_cfg.get("camera_map") or {}),
            state_key=policy_cfg.get("state_key", "observation/state"),
            server_frame=policy_cfg.get("server_frame", "mocap"),
            bridge_admin_url=policy_cfg.get("bridge_admin_url"),
            bridge_policy_id=policy_cfg.get("bridge_policy_id"),
            use_rtc=bool(policy_cfg.get("use_rtc", False)),
            image_size=policy_cfg.get("image_size"),
            channel_order=str(policy_cfg.get("channel_order", "RGB")),
            traceability=dict(policy_cfg.get("traceability") or {}),
            record_dir=record_dir,
        )

        def policy_factory(goal_ned, _policy_cfg):
            fg2 = build_frame_graph(scene_cfg, base_path=scene_dir)
            pol = PiGatewayPolicy(pgcfg, fg2)
            built_policies.append(pol)
            return pol
    else:
        def policy_factory(goal_ned, _policy_cfg):
            cfg = VLAPolicyConfig(
                host=args.host, port=args.port, prompt=args.prompt,
                hz=args.hz, actions_per_chunk=args.actions_per_chunk,
                image_size=args.image_size,
                forward_camera="forward", downward_camera="downward",
                server_frame="mocap",
                record_dir=record_dir,
            )
            # The orchestrator constructs the FrameGraph; we re-build it here
            # purely to pass into VLAPolicy. Cheap (no I/O on the hot path).
            fg2 = build_frame_graph(scene_cfg, base_path=scene_dir)
            pol = VLAPolicy(cfg, fg2)
            built_policies.append(pol)
            return pol

    # pi_gateway draws hz / chunk size from its YAML; openpi uses CLI flags.
    # When use_rtc is set, the runner emits one action per call regardless of
    # execute_chunk_size, so the orchestrator must advance physics one step
    # between policy queries.
    if use_pi_gateway:
        effective_hz = pgcfg.hz
        effective_chunk = 1 if pgcfg.use_rtc else pgcfg.execute_chunk_size
    else:
        effective_hz = args.hz
        effective_chunk = args.actions_per_chunk
    episode_cfg = {
        "hz": effective_hz,
        "horizon_s": args.horizon_s,
        "chunk_steps": effective_chunk,
    }
    ec = EpisodeConfig(
        scene_cfg=scene_cfg, frame_cfg=frame_cfg,
        episode_cfg=episode_cfg, scene_cfg_dir=scene_dir,
    )

    # Wire the safety detector if a YAML was supplied. We borrow the helper
    # from smoke_test rather than duplicating it — both CLIs need the same
    # bounds/velocity/tilt/collision/miss-gate criteria.
    detector_factory = None
    if args.safety is not None:
        from falsify.cli.smoke_test import _build_detector_factory
        safety_cfg = load_yaml(args.safety)
        ec.episode_cfg["safety"] = safety_cfg
        detector_factory = _build_detector_factory(scene_cfg, scene_dir)
        print(f"[run] safety: detector wired from {args.safety} "
              f"(criteria=bounds+velocity+tilt"
              f"{'+collision' if safety_cfg.get('collision', {}).get('enabled') else ''}"
              f"{'+miss_gate' if safety_cfg.get('miss_gate') else ''})")

    # Wire the recovery planner if --recovery was supplied (falsification
    # pipeline). The recovery YAML names a course (or we resolve via the
    # prompt registry) plus an optional planner kind and trigger-failure
    # type filter.
    recovery_factory = None
    recovery_triggers = None
    recovery_cfg: dict = {}
    if args.recovery is not None:
        from falsify.recovery import CoursedMpcPlanner
        from falsify.safety import FailureType

        recovery_cfg = load_yaml(args.recovery)
        course_path = recovery_cfg.get("course")
        if course_path is None:
            # Resolve via the prompt registry's course mapping (TODO if added).
            raise SystemExit(
                "--recovery YAML must set `course:` (no prompt-registry course "
                "fallback yet). Add e.g. "
                "`course: configs/courses/through_left_gate.yaml`."
            )
        if not Path(course_path).is_absolute():
            course_path = Path(__file__).resolve().parents[3] / course_path
        planner_kind = recovery_cfg.get("planner", "mpc")
        triggers_raw = recovery_cfg.get(
            "trigger_failure_types",
            ["MISS_GATE", "COLLISION_GATE", "COLLISION_OTHER", "OUT_OF_BOUNDS"],
        )
        recovery_triggers = {FailureType[name] for name in triggers_raw}

        def recovery_factory(fg2, _episode_recovery_cfg):
            return CoursedMpcPlanner(
                course_path=course_path,
                frame_graph=fg2,
                planner=planner_kind,
                prompt=args.prompt,
            )
        print(f"[run] recovery: course={course_path} planner={planner_kind} "
              f"triggers={sorted(t.name for t in recovery_triggers)}")

    # Optional perturbation suite. Same YAML schema and factory as smoke_test
    # so observation+action surfaces stay symmetric across CLIs.
    perturbations_factory = None
    if args.perturbations is not None:
        from falsify.cli.smoke_test import build_perturbations_factory
        perturbations_factory = build_perturbations_factory(args.perturbations)
        print(f"[run] perturbations: {args.perturbations}")

    print(f"[run] rolling out for up to {args.horizon_s}s @ {effective_hz}Hz, "
          f"chunks of {effective_chunk} steps "
          f"(backend={'pi_gateway' if use_pi_gateway else 'openpi'}, "
          f"pipeline={'falsification' if recovery_factory else 'evaluation'})")
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    episode = run_episode(
        ec,
        policy_factory=policy_factory,
        renderer=renderer,
        detector_factory=detector_factory,
        recovery_factory=recovery_factory,
        recovery_triggers=recovery_triggers,
        perturbations_factory=perturbations_factory,
        rng=rng,
    )
    print(f"[run] rollout finished in {time.time() - t0:.1f}s "
          f"({len(episode.trace.states)} states)")

    # ---- Output bundles -----------------------------------------------
    # 1. Per-frame combined PLYs.
    traj_ned = episode.trace.trajectory()
    plys = _dump_trajectory_with_scene(
        traj_ned, fg, scene_cfg, scene_dir, out_dir / "frames",
    )

    # 2. Forward-camera flythrough.
    fwd_cam_yaml = frame_cfg["cameras"]["forward"]
    fwd_sensor = make_camera_sensor_from_yaml(
        "forward", fwd_cam_yaml, fg,
        renderer=renderer.render,
        body_to_world=camera_to_world_pose,
    )
    flythrough_path = _render_flythrough(
        episode.trace.states, renderer, fwd_sensor.spec, camera_to_world_pose,
        out_dir / "flythrough.mp4",
        fps=args.flythrough_fps, every=args.flythrough_every,
    )

    # 3. Summary.
    summary = {
        "scene": str(args.scene),
        "frame_yaml": str(args.frame),
        "prompt": args.prompt,
        "host": args.host,
        "port": args.port,
        "hz": effective_hz,
        "actions_per_chunk": effective_chunk,
        "horizon_s": args.horizon_s,
        "image_size": args.image_size,
        "seed": args.seed,
        "start_ned_resolved": episode.trace.states[0].pos.xyz.tolist() if episode.trace.states else None,
        "policy_backend": "pi_gateway" if use_pi_gateway else "openpi",
        "policy_config": str(args.policy_config) if use_pi_gateway else None,
        "policy_traceability": pgcfg.traceability if use_pi_gateway else None,
        "n_states": len(episode.trace.states),
        "n_chunks": len(episode.trace.policy_outputs),
        "failure": (None if episode.failure is None
                    else {"step": episode.failure.failure_step,
                          "type": episode.failure.failure_type.name,
                          "criterion": episode.failure.criterion_name,
                          "description": episode.failure.description}),
        "goal_ned": episode.goal.xyz.tolist() if episode.goal is not None else None,
        "ply_paths": {f: str(p) for f, p in plys.items()},
        "flythrough_path": str(flythrough_path) if flythrough_path else None,
        "vla_io_dir": str(record_dir),
        "perturbations": episode.metadata.get("perturbations"),
    }
    # 4. Recovery trajectory (falsification pipeline). Persisted as both a
    # Trajectory NPZ (so plot_rollout_trajectories.py can overlay it) and
    # a summary block.
    if episode.recovery_trajectory is not None:
        from falsify.training import save_trajectory
        from falsify.training.trajectory import Trajectory as TrainingTrajectory
        rt = episode.recovery_trajectory
        # episode.recovery_trajectory is a geometry.Trajectory; re-wrap as a
        # training.Trajectory NPZ for downstream compat.
        quats = rt.quaternions if rt.quaternions is not None else np.tile(
            np.array([0., 0., 0., 1.]), (len(rt.positions), 1),
        )
        save_trajectory(
            out_dir / "recovery_trajectory.npz",
            TrainingTrajectory(
                times=rt.times,
                positions_ned=rt.positions,
                quaternions_xyzw=quats,
                prompt=args.prompt,
                source="recovery",
            ),
        )
        seed_info = (episode.metadata or {}).get("recovery_seed") or {}
        summary["recovery"] = {
            "course": str(course_path),
            "planner": planner_kind,
            "triggers": sorted(t.name for t in recovery_triggers),
            "fired": True,
            "trajectory_npz": str(out_dir / "recovery_trajectory.npz"),
            "n_states": int(len(rt.positions)),
            "duration_s": float(rt.times[-1] - rt.times[0]),
            "seed_step": seed_info.get("step"),
            "seed_bias": seed_info.get("bias"),
            "n_safe_states": seed_info.get("n_safe"),
        }
    elif args.recovery is not None:
        summary["recovery"] = {
            "course": str(course_path),
            "planner": planner_kind,
            "triggers": sorted(t.name for t in recovery_triggers),
            "fired": False,
            "reason": (
                "no failure" if episode.failure is None
                else f"failure type {episode.failure.failure_type.name} not in triggers"
            ),
        }
    (out_dir / "episode_summary.json").write_text(json.dumps(summary, indent=2))

    # Policy manifest — independent of episode_summary, so a reviewer can
    # answer "which checkpoint produced this run?" without parsing the
    # whole summary. Captures the YAML sha, the bridge handshake result
    # (if any), and the YAML's traceability block.
    if use_pi_gateway:
        manifest = {
            "policy_config_path": str(args.policy_config),
            "policy_config_sha256": policy_cfg_sha,
            "bridge_admin_url": pgcfg.bridge_admin_url,
            "bridge_policy_id": pgcfg.bridge_policy_id,
            "bridge_handshake": (
                built_policies[0].bridge_manifest
                if built_policies and built_policies[0].bridge_manifest is not None
                else None
            ),
            "traceability": pgcfg.traceability,
        }
        (out_dir / "policy_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[done] {out_dir / 'episode_summary.json'}")

    # Close the websocket — openpi_client uses a non-daemon thread that
    # otherwise keeps the process alive after main() returns.
    for pol in built_policies:
        pol.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
