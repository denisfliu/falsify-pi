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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--frame", required=True, type=Path,
                        help="Drone-frame YAML (e.g. configs/frames/carl_dual.yaml).")
    parser.add_argument("--prompt", required=True, type=str)
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
    args = parser.parse_args(argv)

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
        from falsify.policy import PiGatewayConfig, PiGatewayPolicy
        policy_cfg = load_yaml(args.policy_config)
        if policy_cfg.get("type") != "pi_gateway":
            raise ValueError(
                f"--policy-config expects type: pi_gateway, got {policy_cfg.get('type')!r}"
            )
        pgcfg = PiGatewayConfig(
            gateway_url=policy_cfg["gateway_url"],
            api_key=policy_cfg.get("api_key", ""),
            execute_chunk_size=int(policy_cfg.get("execute_chunk_size", 25)),
            prompt=policy_cfg.get("prompt", args.prompt),
            hz=int(policy_cfg.get("hz", args.hz)),
            state_dim=int(policy_cfg.get("state_dim", 7)),
            action_dim=int(policy_cfg.get("action_dim", 7)),
            action_pos_slice=tuple(policy_cfg.get("action_pos_slice", (0, 3))),
            action_yaw_index=policy_cfg.get("action_yaw_index", 3),
            camera_map=dict(policy_cfg.get("camera_map") or {}),
            state_key=policy_cfg.get("state_key", "observation/state"),
            server_frame=policy_cfg.get("server_frame", "mocap"),
            use_rtc=bool(policy_cfg.get("use_rtc", False)),
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

    print(f"[run] rolling out for up to {args.horizon_s}s @ {effective_hz}Hz, "
          f"chunks of {effective_chunk} steps "
          f"(backend={'pi_gateway' if use_pi_gateway else 'openpi'})")
    t0 = time.time()
    episode = run_episode(
        ec,
        policy_factory=policy_factory,
        renderer=renderer.render,
        detector_factory=detector_factory,
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
    }
    (out_dir / "episode_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[done] {out_dir / 'episode_summary.json'}")

    # Close the websocket — openpi_client uses a non-daemon thread that
    # otherwise keeps the process alive after main() returns.
    for pol in built_policies:
        pol.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
