"""PointCloud / Trajectory → PLY writers + small helpers.

Frame-aware: the writer encodes the frame name into the PLY header as a
comment so a reader (and a human) can tell which frame a `.ply` is in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from falsify.geometry import Frame, PointCloud, Trajectory


def trajectory_to_pointcloud(
    traj: Trajectory,
    color: Optional[Sequence[float]] = None,
) -> PointCloud:
    """View a `Trajectory` as a `PointCloud` in the same frame."""
    colors = None
    if color is not None:
        c = np.asarray(color, dtype=np.float64)
        if c.shape != (3,):
            raise ValueError("color must be a 3-tuple")
        colors = np.tile(c[None, :], (traj.positions.shape[0], 1))
    return PointCloud(points=traj.positions.copy(), frame=traj.frame, colors=colors)


def write_ply(pc: PointCloud, path: str | Path) -> Path:
    """Write a `PointCloud` to an ASCII PLY file, recording its frame in a comment."""
    path = Path(path)
    n = pc.points.shape[0]
    has_color = pc.colors is not None
    header = [
        "ply",
        "format ascii 1.0",
        f"comment falsify frame: {pc.frame.name}",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_color:
        header += [
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
    header.append("end_header")
    lines = ["\n".join(header)]

    if has_color:
        colors = pc.colors
        if colors.dtype.kind == "f" and colors.max() <= 1.0 + 1e-6:
            colors = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
        else:
            colors = np.clip(colors, 0, 255).astype(np.uint8)
        for p, c in zip(pc.points, colors):
            lines.append(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}")
    else:
        for p in pc.points:
            lines.append(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def stack_pointclouds(clouds: Iterable[PointCloud]) -> PointCloud:
    """Concatenate point clouds **in the same frame** into one."""
    clouds = list(clouds)
    if not clouds:
        raise ValueError("no clouds to stack")
    frame = clouds[0].frame
    for c in clouds[1:]:
        if c.frame.name != frame.name:
            raise ValueError(
                f"stack_pointclouds: frame mismatch ({c.frame.name!r} vs {frame.name!r})"
            )
    pts = np.concatenate([c.points for c in clouds], axis=0)
    if any(c.colors is None for c in clouds):
        return PointCloud(points=pts, frame=frame)
    cols = np.concatenate([c.colors for c in clouds], axis=0)
    return PointCloud(points=pts, frame=frame, colors=cols)
