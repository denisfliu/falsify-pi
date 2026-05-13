"""Visualize a sample trajectory + the scene's object point clouds in every
configured frame.

Use this to sanity-check a scene's `FrameGraph` end-to-end. The output per
target frame is a single ``combined_<frame>.ply`` that contains the sample
trajectory **plus** every PLY listed in the scene YAML's ``scene_objects``
block, all converted through the FrameGraph. Open the per-frame combined
files side by side in MeshLab / open3d / Blender — frame errors show up as
the trajectory floating away from the gate / table in one of the frames.

The trajectory is a deterministic helix in MOCAP (so two runs are
byte-identical), and a numerical round-trip ``MOCAP → <each frame> → MOCAP``
check is run before any files are written.

Usage::

    PYTHONPATH=src python -m falsify.cli.visualize_frames \\
        --scene configs/scenes/left_gate.yaml \\
        --out runs/viz/left_gate

    PYTHONPATH=src python -m falsify.cli.visualize_frames \\
        --scene configs/scenes/right_gate.yaml \\
        --out runs/viz/right_gate \\
        --frames mocap ned ns \\
        --max-object-points 5000
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from falsify.geometry import FrameGraph, PointCloud, Trajectory
from falsify.io import build_frame_graph, load_yaml
from falsify.visualization import (
    dump_trajectory_in_frames, read_ply, stack_pointclouds, subsample,
    trajectory_to_pointcloud, write_ply,
)


TRAJECTORY_COLOR = (1.0, 0.95, 0.20)  # bright yellow — pops over any scene


@dataclass
class SceneObject:
    name: str
    cloud: PointCloud  # in its native frame
    color: tuple[float, float, float]


def _resolve_path(p: str, base: Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (base / path).resolve()


def _load_scene_objects(scene_cfg: dict, scene_dir: Path, fg: FrameGraph) -> list[SceneObject]:
    objects = []
    for entry in scene_cfg.get("scene_objects", []):
        ply_path = _resolve_path(entry["ply"], scene_dir)
        frame_name = entry["frame"]
        frame = fg.frame(frame_name)
        cloud = read_ply(ply_path, frame)
        color = tuple(entry.get("color", (0.5, 0.5, 0.5)))
        if len(color) != 3:
            raise ValueError(f"scene_objects[{entry['name']}].color must be RGB triplet")
        objects.append(SceneObject(name=entry["name"], cloud=cloud, color=color))
    return objects


def _tinted(pc: PointCloud, color: Sequence[float]) -> PointCloud:
    c = np.asarray(color, dtype=np.float64)
    return PointCloud(
        points=pc.points,
        frame=pc.frame,
        colors=np.broadcast_to(c, pc.points.shape).copy(),
    )


def _helix_in_mocap(
    frame,
    *,
    center: tuple[float, float, float] = (0.5, 0.0, 1.2),
    radius: float = 0.6,
    height: float = 0.8,
    turns: float = 1.5,
    n: int = 400,
) -> Trajectory:
    """Deterministic helix trajectory in the given (mocap) frame."""
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)
    theta = 2.0 * np.pi * turns * t
    cx, cy, cz = center
    x = cx + radius * np.cos(theta)
    y = cy + radius * np.sin(theta)
    z = cz + height * (t - 0.5)
    pos = np.stack([x, y, z], axis=1)
    return Trajectory(times=t, positions=pos, frame=frame)


def _round_trip_check(traj: Trajectory, fg: FrameGraph, frames: Iterable[str], *, atol: float) -> None:
    origin = traj.frame.name
    for f in frames:
        if f == origin:
            continue
        out = fg.convert(fg.convert(traj, to=f), to=origin)
        max_err = float(np.max(np.abs(out.positions - traj.positions)))
        status = "OK " if max_err < atol else "FAIL"
        print(f"  round-trip  {origin} → {f} → {origin}   max |Δ| = {max_err:.3e}   [{status}]")
        if max_err >= atol:
            raise SystemExit(
                f"round-trip via {f!r} exceeded tolerance {atol:.1e} (got {max_err:.3e})"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", required=True, type=Path,
                        help="Path to a scene YAML (e.g. configs/scenes/left_gate.yaml).")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output directory for the per-frame .ply files.")
    parser.add_argument("--frames", nargs="*", default=None,
                        help="Frames to dump in. Defaults to mocap/ned/ns if "
                             "they are present, else all frames in the graph.")
    parser.add_argument("--origin-frame", default="mocap",
                        help="Frame the sample trajectory lives in (default: mocap).")
    parser.add_argument("--atol", type=float, default=1e-6,
                        help="Round-trip tolerance (default 1e-6).")
    parser.add_argument("--max-object-points", type=int, default=8000,
                        help="Subsample each scene object to at most this many "
                             "points (per frame). 0 disables subsampling.")
    args = parser.parse_args(argv)

    scene_cfg = load_yaml(args.scene)
    fg = build_frame_graph(scene_cfg, base_path=args.scene.parent)
    print(fg.describe())
    print()

    if args.origin_frame not in {f.name for f in fg.frames}:
        raise SystemExit(f"--origin-frame {args.origin_frame!r} not in graph")
    origin_frame = fg.frame(args.origin_frame)

    if args.frames:
        target_frames = tuple(args.frames)
    else:
        graph_frames = {f.name for f in fg.frames}
        target_frames = tuple(n for n in ("mocap", "ned", "ns") if n in graph_frames)
        if not target_frames:
            target_frames = tuple(sorted(graph_frames))

    # Center the helix near the scene's gate so it visually interacts with the
    # object PLYs (start in front of the gate, spiral through, exit behind).
    center = (0.5, 0.0, 1.2)
    if "gate_position_mocap" in scene_cfg:
        gp = scene_cfg["gate_position_mocap"]
        center = (float(gp[0]) - 0.3, float(gp[1]), float(gp[2]) - 0.3)
    traj = _helix_in_mocap(origin_frame, center=center)

    print(f"Round-trip check (origin = {origin_frame.name}):")
    _round_trip_check(traj, fg, target_frames, atol=args.atol)
    print()

    objects = _load_scene_objects(scene_cfg, args.scene.parent, fg)
    if objects:
        print(f"Loaded {len(objects)} scene object(s):")
        for obj in objects:
            print(f"  {obj.name:>10}  frame={obj.cloud.frame.name}  n={len(obj.cloud)}")
        print()
    else:
        print("(no scene_objects declared in YAML — only trajectory will be dumped)\n")

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Per-frame trajectory PLYs (existing behavior).
    written = dump_trajectory_in_frames(
        traj, fg, out_dir,
        name=scene_cfg.get("scene_key", "sample"),
        target_frames=target_frames,
        color=TRAJECTORY_COLOR,
    )

    # 2. Per-frame per-object PLYs + a combined-everything PLY per frame.
    for f in target_frames:
        traj_in_frame = fg.convert(traj, to=f)
        traj_pc = trajectory_to_pointcloud(traj_in_frame, color=TRAJECTORY_COLOR)
        combined = [traj_pc]
        for obj in objects:
            in_frame = fg.convert(obj.cloud, to=f)
            if args.max_object_points > 0:
                in_frame = subsample(in_frame, args.max_object_points)
            tinted = _tinted(in_frame, obj.color)
            write_ply(tinted, out_dir / f"{obj.name}_{f}.ply")
            combined.append(tinted)
        write_ply(stack_pointclouds(combined), out_dir / f"combined_{f}.ply")

    print("Wrote PLYs:")
    for f, p in written.items():
        print(f"  trajectory {f:>8}  →  {p}")
    for f in target_frames:
        print(f"  combined   {f:>8}  →  {out_dir / f'combined_{f}.ply'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
