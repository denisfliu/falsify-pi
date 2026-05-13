"""Visualize a Course (waypoint list) inside a scene.

For each requested frame (``mocap`` / ``ned`` / ``ns`` by default),
writes:

  <out>/waypoints_<frame>.ply       — colored markers at each waypoint
  <out>/spline_<frame>.ply          — planned spline trajectory (if --plan)
  <out>/combined_<frame>.ply        — waypoints + spline + scene_objects
                                      stacked in one file for one-click viewing

Open the per-frame ``combined_*.ply`` in MeshLab / open3d / blender; the
waypoints appear as small colored markers, the spline as a yellow line,
and the gate/table as their tinted clouds. If the waypoints look wrong
(inside a gate post, below the floor, etc.), edit the course YAML and
re-run — no other code touches needed.

Sample invocation::

    PYTHONPATH=src .venv/bin/python -m falsify.cli.visualize_waypoints \\
        --course configs/courses/through_left_gate.yaml \\
        --scene configs/scenes/left_gate.yaml \\
        --out runs/wpviz/through_left_gate \\
        --plan
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from falsify.geometry import PointCloud
from falsify.io import build_frame_graph, load_yaml
from falsify.planning import load_course, plan_spline
from falsify.visualization import (
    read_ply, stack_pointclouds, subsample,
    trajectory_to_pointcloud, write_ply,
)


# Distinct colors for endpoint vs interior waypoints.
COLOR_START = (0.10, 0.85, 0.10)   # green
COLOR_GOAL = (0.85, 0.85, 0.20)    # yellow
COLOR_INTERIOR = (0.20, 0.65, 0.95)  # blue
COLOR_SPLINE = (1.0, 0.95, 0.20)     # bright yellow line


def _waypoint_markers_pc(course, fg, *, dst_frame: str) -> PointCloud:
    """Pack waypoints as a colored point cloud in dst_frame."""
    points_src = course.positions
    src_pcd = PointCloud(points=points_src, frame=fg.frame(course.frame))
    dst_pcd = fg.convert(src_pcd, to=dst_frame)

    n = len(course.waypoints)
    colors = []
    for i in range(n):
        if i == 0:
            colors.append(COLOR_START)
        elif i == n - 1:
            colors.append(COLOR_GOAL)
        else:
            colors.append(COLOR_INTERIOR)
    return PointCloud(
        points=dst_pcd.points,
        frame=dst_pcd.frame,
        colors=np.asarray(colors, dtype=np.float64),
    )


def _scene_object_clouds(scene_cfg, scene_dir, fg, max_points: int = 8000):
    objects = []
    for entry in scene_cfg.get("scene_objects", []):
        ply_path = Path(entry["ply"])
        if not ply_path.is_absolute():
            ply_path = (scene_dir / ply_path).resolve()
        cloud = read_ply(ply_path, fg.frame(entry["frame"]))
        color = tuple(entry.get("color", (0.5, 0.5, 0.5)))
        objects.append((entry["name"], cloud, color))
    return objects


def _tinted(pc: PointCloud, color) -> PointCloud:
    c = np.asarray(color, dtype=np.float64)
    return PointCloud(
        points=pc.points,
        frame=pc.frame,
        colors=np.broadcast_to(c, pc.points.shape).copy(),
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--course", required=True, type=Path)
    p.add_argument("--scene", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--frames", nargs="*", default=None,
                   help="Frames to dump (default: mocap, ned, ns when present).")
    p.add_argument("--plan", action="store_true",
                   help="Also plan + visualize the spline through the waypoints.")
    p.add_argument("--max-object-points", type=int, default=8000,
                   help="Subsample scene object PLYs to this many points "
                        "(0 disables).")
    args = p.parse_args(argv)

    scene_cfg = load_yaml(args.scene)
    course = load_course(args.course)
    scene_dir = args.scene.parent
    fg = build_frame_graph(scene_cfg, base_path=scene_dir)

    if args.frames:
        target_frames = tuple(args.frames)
    else:
        graph = {f.name for f in fg.frames}
        target_frames = tuple(n for n in ("mocap", "ned", "ns") if n in graph)
        if not target_frames:
            target_frames = tuple(sorted(graph))

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[course] {course.name}  ({len(course.waypoints)} waypoints, "
          f"frame={course.frame}, yaw_mode={course.yaw_mode}, "
          f"total_time={course.total_time_s}s)")
    times = course.resolved_times()
    yaws = course.resolved_yaws()
    for i, wp in enumerate(course.waypoints):
        print(f"  [{i}] {wp.name:12s}  p={wp.p.tolist()}  "
              f"t={times[i]:5.2f}  yaw_{course.frame}={yaws[i]:+.3f}rad "
              f"({np.degrees(yaws[i]):+.1f}°)")

    # Plan once (cheap) — needed for the spline overlay and also for sanity
    # output even when --plan isn't set so we surface gross issues early.
    traj = plan_spline(course, fg)

    objects = _scene_object_clouds(scene_cfg, scene_dir, fg)
    print(f"[scene] loaded {len(objects)} scene_objects")
    print(f"[frames] target = {target_frames}")

    for f in target_frames:
        # Waypoint markers
        wp_pc = _waypoint_markers_pc(course, fg, dst_frame=f)
        wp_path = write_ply(wp_pc, out_dir / f"waypoints_{f}.ply")

        # Optional spline
        spline_pc = None
        if args.plan:
            from falsify.geometry import Trajectory as GeoTrajectory
            from falsify.training.trajectory import Trajectory as TrTrajectory
            # Build a geometry.Trajectory in NED, then convert.
            geo_traj = GeoTrajectory(
                times=traj.times,
                positions=traj.positions_ned,
                frame=fg.frame("ned"),
                quaternions=traj.quaternions_xyzw,
            )
            geo_in_frame = fg.convert(geo_traj, to=f)
            spline_pc = trajectory_to_pointcloud(geo_in_frame, color=COLOR_SPLINE)
            write_ply(spline_pc, out_dir / f"spline_{f}.ply")

        # Combined: waypoints + spline + scene objects
        combined = [wp_pc]
        if spline_pc is not None:
            combined.append(spline_pc)
        for name, cloud, color in objects:
            in_frame = fg.convert(cloud, to=f)
            if args.max_object_points > 0:
                in_frame = subsample(in_frame, args.max_object_points)
            combined.append(_tinted(in_frame, color))
        write_ply(stack_pointclouds(combined), out_dir / f"combined_{f}.ply")

    print(f"[done] wrote PLYs to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
