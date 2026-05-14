"""Re-render a previous run's flown trajectory using the current renderer.

Reads ``--run-dir/vla_io/query_*/{data.txt,waypoints_ned.npy,actions.npy}``,
stitches the chunks into one continuous (pos_ned, yaw_ned) sequence, and
renders the forward + downward cameras at every step. Writes:

  --out/per_step/step_<NNNN>_{fwd,dwn}.png
  --out/flythrough_fwd.mp4
  --out/flythrough_grid.mp4         (forward | downward side-by-side)

The simulator's chunked rollout (see ``Simulator.rollout_with_policy``)
advances the drone by indexing into the policy's emitted chunk: at step k
into a chunk it appends the *current* state then transitions to
``chunk.positions[offset]``. We mirror that here from the recorded chunks
so the rendered viewpoints exactly match what the simulator visited.

Usage::

    CC=gcc-11 CXX=g++-11 \\
    PYTHONPATH=src:external/FiGS/src:external/splatnav \\
    .venv/bin/python scripts/replay_renders.py \\
        --run-dir runs/vla_20260512_160932 \\
        --scene configs/scenes/left_gate.yaml \\
        --frame configs/frames/carl_dual.yaml \\
        --out runs/vla_20260512_160932/replay
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ChunkData:
    start_pos_ned: np.ndarray  # (3,)
    start_yaw_ned: float
    actions: np.ndarray        # (N, 7)
    waypoints_ned: np.ndarray  # (N+1, 3) — prepended start, then N integrated


_DATA_PAT = re.compile(r"^([^:]+):\s*(.*)$")


def _parse_kv(text: str) -> dict:
    out: dict = {}
    for line in text.splitlines():
        m = _DATA_PAT.match(line.strip())
        if not m:
            continue
        out[m.group(1).strip()] = m.group(2).strip()
    return out


def _parse_vec3(s: str) -> np.ndarray:
    return np.array(eval(s, {"__builtins__": {}}, {}), dtype=np.float64)  # noqa: S307


def _load_chunks(run_dir: Path) -> list[ChunkData]:
    qdirs = sorted([p for p in (run_dir / "vla_io").iterdir() if p.is_dir()])
    chunks: list[ChunkData] = []
    for d in qdirs:
        kv = _parse_kv((d / "data.txt").read_text())
        actions = np.load(d / "actions.npy")
        waypoints = np.load(d / "waypoints_ned.npy")
        chunks.append(ChunkData(
            start_pos_ned=_parse_vec3(kv["state_ned_pos"]),
            start_yaw_ned=float(kv["state_ned_yaw_rad"]),
            actions=actions,
            waypoints_ned=waypoints,
        ))
    return chunks


def _yaw_sequence(chunk: ChunkData) -> np.ndarray:
    """Per-waypoint NED yaw, matching ``VLAPolicy.observe`` exactly:

        yaws[0] = start_yaw_ned
        yaws[i+1] = yaws[i] - action[i, 3]    (MOCAP→NED yaw sign flip)
    """
    n = chunk.actions.shape[0]
    yaws = np.zeros(n + 1)
    yaws[0] = chunk.start_yaw_ned
    has_yaw = chunk.actions.shape[1] >= 4
    for i in range(n):
        yaws[i + 1] = yaws[i] - (float(chunk.actions[i, 3]) if has_yaw else 0.0)
    return yaws


def _yaw_to_quat_xyzw(yaw: float) -> np.ndarray:
    return np.array([0.0, 0.0, np.sin(0.5 * yaw), np.cos(0.5 * yaw)])


def _reconstruct_states(chunks: list[ChunkData], chunk_budget: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Walk chunks the way the simulator does: ``chunk_budget`` steps each,
    appending the *current* state at each step before transitioning. The
    returned list is (pos_ned, quat_xyzw) per simulator step.
    """
    states: list[tuple[np.ndarray, np.ndarray]] = []
    cur_pos = chunks[0].start_pos_ned.copy()
    cur_yaw = chunks[0].start_yaw_ned
    for ck_idx, chunk in enumerate(chunks):
        # Re-seed the chunk's start to its recorded value — handles small
        # drift between the previous chunk's end and the next chunk's prefix.
        cur_pos = chunk.start_pos_ned.copy()
        cur_yaw = chunk.start_yaw_ned
        yaws = _yaw_sequence(chunk)
        wp = chunk.waypoints_ned
        n = min(chunk_budget, len(wp))
        for step in range(n):
            states.append((cur_pos.copy(), _yaw_to_quat_xyzw(cur_yaw)))
            cur_pos = wp[step]
            cur_yaw = yaws[step]
    # Final state (mirrors trace.states.append after the loop in the simulator).
    states.append((cur_pos.copy(), _yaw_to_quat_xyzw(cur_yaw)))
    return states


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--frame", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--chunk-steps", type=int, default=50)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--every", type=int, default=1,
                        help="Render every Nth step to speed up (1 = all).")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "per_step").mkdir(exist_ok=True)

    from PIL import Image
    import imageio.v2 as imageio

    from falsify.geometry import Point
    from falsify.io import build_frame_graph, load_yaml
    from falsify.sensors.camera import make_camera_sensor_from_yaml
    from falsify.sim.dynamics_state import DroneState
    from falsify.sim.poses import camera_to_world_pose
    from falsify.sim.renderer import GSplatRenderer

    scene_cfg = load_yaml(args.scene)
    frame_cfg = load_yaml(args.frame)
    scene_dir = args.scene.parent

    def _resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (scene_dir / path).resolve()

    fg = build_frame_graph(scene_cfg, base_path=scene_dir)
    renderer = GSplatRenderer.from_scene_cfg(scene_cfg, scene_dir=scene_dir)

    fwd = make_camera_sensor_from_yaml(
        "forward", frame_cfg["cameras"]["forward"], fg,
        renderer=renderer.render, body_to_world=camera_to_world_pose,
    )
    dwn = make_camera_sensor_from_yaml(
        "downward", frame_cfg["cameras"]["downward"], fg,
        renderer=renderer.render, body_to_world=camera_to_world_pose,
    )

    chunks = _load_chunks(args.run_dir)
    print(f"[chunks] loaded {len(chunks)} chunks")
    states = _reconstruct_states(chunks, args.chunk_steps)
    print(f"[states] reconstructed {len(states)} per-step states")

    fwd_frames: list[np.ndarray] = []
    grid_frames: list[np.ndarray] = []
    ned_frame = fg.frame("ned")
    step_indices = list(range(0, len(states), max(1, args.every)))
    for i, step in enumerate(step_indices):
        pos_ned, quat = states[step]
        ds = DroneState(
            pos=Point(pos_ned, frame=ned_frame),
            vel=np.zeros(3), quat_xyzw=quat, t=step / args.fps,
        )
        pose_fwd = camera_to_world_pose(ds, fwd.spec.body_from_camera)
        pose_dwn = camera_to_world_pose(ds, dwn.spec.body_from_camera)
        rgb_fwd, _ = renderer.render(pose_fwd, fwd.spec.intrinsics)
        rgb_dwn, _ = renderer.render(pose_dwn, dwn.spec.intrinsics)

        Image.fromarray(rgb_fwd).save(args.out / "per_step" / f"step_{step:04d}_fwd.png")
        Image.fromarray(rgb_dwn).save(args.out / "per_step" / f"step_{step:04d}_dwn.png")
        fwd_frames.append(np.asarray(rgb_fwd, dtype=np.uint8))

        # Side-by-side grid (pad both to same height by upscaling smaller).
        a, b = rgb_fwd, rgb_dwn
        if a.shape[0] != b.shape[0]:
            target_h = max(a.shape[0], b.shape[0])
            from PIL import Image as _I
            def _scale(img, h):
                if img.shape[0] == h:
                    return img
                ar = img.shape[1] / img.shape[0]
                w = max(1, int(round(h * ar)))
                return np.asarray(_I.fromarray(img).resize((w, h), _I.BILINEAR))
            a = _scale(a, target_h)
            b = _scale(b, target_h)
        grid_frames.append(np.concatenate([a, b], axis=1))

        if (i + 1) % 20 == 0 or i + 1 == len(step_indices):
            print(f"  rendered {i + 1}/{len(step_indices)}")

    imageio.mimwrite(args.out / "flythrough_fwd.mp4", fwd_frames, fps=args.fps, quality=8)
    imageio.mimwrite(args.out / "flythrough_grid.mp4", grid_frames, fps=args.fps, quality=8)
    print(f"\nWrote {args.out / 'flythrough_fwd.mp4'}")
    print(f"Wrote {args.out / 'flythrough_grid.mp4'}")
    print(f"Wrote {len(step_indices)} step PNGs to {args.out / 'per_step'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
