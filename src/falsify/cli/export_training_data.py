"""Export training-data parquet(s) from one or more trajectories.

End-to-end:

  Trajectory NPZ (or VLA run dir)
      │
      ▼
  TrainingDataExporter (loads scene + embodiment + GSplatRenderer once)
      │
      ▼
  <out>/episode_<id>/episode_<id>.parquet  (LeRobot-style HF-Image schema)

The CLI supports three input modes, in order of precedence:

  --run-dir          : parse a recorded VLA run directly (calls
                       ``trajectory.from_vla_run_dir``) — one episode.
  --trajectory       : single Trajectory NPZ — one episode.
  --trajectories-dir : directory of NPZs — one episode per file, reusing
                       the renderer; intended for batch orchestration.

For multi-episode batches you typically want consecutive
``--episode-index`` and a global ``--index-offset`` so the resulting
parquets concatenate cleanly. The simplest pattern is to let
``--trajectories-dir`` auto-assign indices in sorted-filename order.

Example::

    CC=gcc-11 CXX=g++-11 \\
    PYTHONPATH=src:external/FiGS/src:external/splatnav \\
    .venv/bin/python -m falsify.cli.export_training_data \\
        --run-dir runs/vla_20260512_160932 \\
        --scene configs/scenes/left_gate.yaml \\
        --frame configs/frames/carl_dual.yaml \\
        --embodiment configs/embodiments/carl_dual_mocap.yaml \\
        --out runs/datasets/left_gate \\
        --episode-index 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _resolve_rel(p: str, base: Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (base / pp).resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--trajectory", type=Path,
                     help="Path to a Trajectory NPZ (one episode).")
    src.add_argument("--run-dir", type=Path,
                     help="Path to a VLA run directory (uses from_vla_run_dir).")
    src.add_argument("--trajectories-dir", type=Path,
                     help="Directory of Trajectory NPZs (one episode each, "
                          "sorted by filename).")
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--frame", required=True, type=Path)
    parser.add_argument("--embodiment", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path,
                        help="Output root. Each episode goes under "
                             "<out>/episode_<index>/.")
    parser.add_argument("--episode-index", type=int, default=0,
                        help="Episode index for a single-episode run. "
                             "Ignored when --trajectories-dir is set (the "
                             "CLI auto-numbers from this base).")
    parser.add_argument("--index-offset", type=int, default=0,
                        help="Starting value for the per-frame global "
                             "`index` column. Increment across calls so "
                             "concatenated datasets remain consistent.")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--hz", type=float, default=None,
                        help="Override the embodiment's fps for resampling.")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Override the trajectory's prompt for this run.")
    parser.add_argument("--chunk-steps", type=int, default=50,
                        help="VLA chunk size for --run-dir reconstruction.")
    args = parser.parse_args(argv)

    # Lazy imports — heavy.
    from falsify.io import build_frame_graph, load_yaml
    from falsify.sim.renderer import GSplatRenderer
    from falsify.training import (
        TrainingDataExporter,
        load_embodiment,
        load_trajectory,
        from_vla_run_dir,
    )

    scene_cfg = load_yaml(args.scene)
    frame_cfg = load_yaml(args.frame)
    embodiment = load_embodiment(args.embodiment)
    scene_dir = args.scene.parent
    fg = build_frame_graph(scene_cfg, base_path=scene_dir)

    gsplat_config = _resolve_rel(scene_cfg["gsplat_config_yml"], scene_dir)
    data_cwd = (_resolve_rel(scene_cfg["gsplat_data_cwd"], scene_dir)
                if "gsplat_data_cwd" in scene_cfg else None)
    print(f"[scene] loading gsplat at {gsplat_config} (cwd={data_cwd})")
    renderer = GSplatRenderer(
        gsplat_config, world_frame="ned", data_cwd=data_cwd, frame_graph=fg,
    )

    exporter = TrainingDataExporter(
        scene_cfg=scene_cfg, frame_cfg=frame_cfg, frame_graph=fg,
        renderer=renderer.render, embodiment=embodiment,
    )

    # Resolve trajectories to process.
    if args.run_dir is not None:
        traj = from_vla_run_dir(args.run_dir, chunk_steps=args.chunk_steps,
                                hz=int(embodiment.fps))
        episodes = [(args.episode_index, traj)]
    elif args.trajectory is not None:
        traj = load_trajectory(args.trajectory)
        episodes = [(args.episode_index, traj)]
    else:
        npz_paths = sorted(args.trajectories_dir.glob("*.npz"))
        if not npz_paths:
            raise SystemExit(f"no .npz files under {args.trajectories_dir}")
        episodes = [
            (args.episode_index + i, load_trajectory(p))
            for i, p in enumerate(npz_paths)
        ]

    out_root = args.out
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[export] {len(episodes)} episode(s) → {out_root}")

    running_index = args.index_offset
    for ep_idx, traj in episodes:
        ep_dir = out_root / f"episode_{ep_idx:06d}"
        result = exporter.export_episode(
            traj, ep_dir,
            episode_index=ep_idx,
            index_offset=running_index,
            task_index=args.task_index,
            prompt_override=args.prompt,
            hz_override=args.hz,
        )
        running_index += result.n_frames
        print(f"[ep {ep_idx:06d}] {result.n_frames} frames "
              f"({result.duration_s:.1f}s) → {result.parquet_path}  "
              f"({result.elapsed_s:.1f}s wall)")

    print(f"[done] global index reached {running_index}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
