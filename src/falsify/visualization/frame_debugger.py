"""Frame-debugger — dump an episode's geometry in every requested frame.

The workflow this enables: when something looks wrong, dump the trajectory
in ``ned`` and ``mocap`` and ``ns``, load both into a viewer (open3d /
meshlab / blender), and *visually* compare which frame the misalignment
lives in. Frame mistakes show up as obvious mis-projections.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from falsify.geometry import FrameGraph, Point, PointCloud, Trajectory
from .pointcloud import trajectory_to_pointcloud, write_ply, stack_pointclouds


# Distinct RGBs for each named entity. Colors chosen so they're easy to tell
# apart in a viewer.
DEFAULT_COLORS = {
    "nominal":  (0.20, 0.65, 0.95),
    "recovery": (0.85, 0.30, 0.30),
    "start":    (0.10, 0.85, 0.10),
    "goal":     (0.85, 0.85, 0.20),
    "failure":  (1.00, 0.50, 0.00),
    "last_safe":(0.60, 0.20, 0.85),
}


def _markers(points: dict[str, Point], frame_graph: FrameGraph, dst_frame: str) -> PointCloud:
    pts = []
    cols = []
    for name, p in points.items():
        q = frame_graph.convert(p, to=dst_frame)
        pts.append(q.xyz)
        cols.append(DEFAULT_COLORS.get(name, (0.5, 0.5, 0.5)))
    return PointCloud(
        points=np.asarray(pts, dtype=np.float64),
        frame=frame_graph.frame(dst_frame),
        colors=np.asarray(cols, dtype=np.float64),
    )


def dump_trajectory_in_frames(
    traj: Trajectory,
    frame_graph: FrameGraph,
    out_dir: str | Path,
    *,
    name: str = "nominal",
    target_frames: Iterable[str] = ("ned", "mocap", "ns"),
    color: Optional[Sequence[float]] = None,
) -> dict[str, Path]:
    """Dump one trajectory as a .ply per target frame.

    Returns a mapping `frame_name → ply_path`.
    """
    out_dir = Path(out_dir)
    color = color or DEFAULT_COLORS.get(name, (0.5, 0.5, 0.5))
    written: dict[str, Path] = {}
    for f in target_frames:
        converted = frame_graph.convert(traj, to=f)
        pc = trajectory_to_pointcloud(converted, color=color)
        path = write_ply(pc, out_dir / f"{name}_traj_{f}.ply")
        written[f] = path
    return written


def dump_episode(
    episode,
    frame_graph: FrameGraph,
    out_dir: str | Path,
    *,
    target_frames: Iterable[str] = ("ned", "mocap", "ns"),
) -> dict[str, dict[str, Path]]:
    """Dump every interesting bit of an episode in every requested frame.

    Returns a nested dict `{entity_name: {frame_name: ply_path}}`. Entities
    written: ``nominal`` trajectory; ``recovery`` trajectory (if present);
    ``markers`` (start, goal, failure point, last-safe point).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_frames = tuple(target_frames)

    written: dict[str, dict[str, Path]] = {}

    # 1. Nominal trajectory.
    if episode.trace.states:
        traj = episode.trace.trajectory()
        written["nominal"] = dump_trajectory_in_frames(
            traj, frame_graph, out_dir,
            name="nominal", target_frames=target_frames,
        )

    # 2. Recovery trajectory.
    if episode.recovery_trajectory is not None:
        written["recovery"] = dump_trajectory_in_frames(
            episode.recovery_trajectory, frame_graph, out_dir,
            name="recovery", target_frames=target_frames,
        )

    # 3. Markers: start, goal, failure point, last safe.
    markers: dict[str, Point] = {}
    if episode.trace.states:
        markers["start"] = episode.trace.states[0].pos
    if episode.goal is not None:
        markers["goal"] = episode.goal
    if episode.failure is not None:
        markers["failure"] = episode.failure.failure_state.pos
        if episode.failure.last_safe_state is not None:
            markers["last_safe"] = episode.failure.last_safe_state.pos

    if markers:
        marker_paths: dict[str, Path] = {}
        for f in target_frames:
            pc = _markers(markers, frame_graph, f)
            marker_paths[f] = write_ply(pc, out_dir / f"markers_{f}.ply")
        written["markers"] = marker_paths

    return written
