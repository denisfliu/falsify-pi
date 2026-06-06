"""Cubic-spline trajectory producer.

Given a :class:`Course`, fit a cubic spline through the waypoints'
positions over their resolved ``t`` values, sample at the course's
``fps``, and emit a canonical NED :class:`Trajectory` ready for
``TrainingDataExporter`` or any other downstream consumer.

This is the *default* path from waypoints → trajectory. When the
FiGS-MPC integrator lands, ``plan_mpc`` will be a drop-in replacement
that produces dynamically-feasible trajectories instead of geometric
ones; both producers honour the same Course schema and emit the same
Trajectory NPZ, so consumers downstream don't notice the swap.

Yaw handling
------------
The course's ``yaw_mode`` decides per-waypoint yaws (see
``waypoints.py``). Spline samples in between use linear interpolation
of those resolved yaws with shortest-arc wrapping — yaw rates between
waypoints stay continuous even if the cumulative angle drifts past 2π.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.interpolate import CubicSpline

from falsify.geometry import FrameGraph, PointCloud, Trajectory
from falsify.training.trajectory import Trajectory as TrainingTrajectory

from .waypoints import Course


from falsify.geometry import yaw_to_quat_xyzw as _yaw_to_quat_xyzw  # noqa: E402


def _shortest_arc(target: float, reference: float) -> float:
    diff = target - reference
    return float(np.arctan2(np.sin(diff), np.cos(diff)))


def _unwrap_yaws_along_path(yaw_waypoints: np.ndarray) -> np.ndarray:
    """Make consecutive waypoint yaws differ by at most π so linear interp
    between them never takes the long way around the unit circle."""
    out = np.array(yaw_waypoints, dtype=np.float64).copy()
    for i in range(1, len(out)):
        out[i] = out[i - 1] + _shortest_arc(out[i], out[i - 1])
    return out


def plan_spline(
    course: Course,
    frame_graph: FrameGraph,
    *,
    prompt: str = "",
) -> TrainingTrajectory:
    """Cubic-spline plan in NED at ``course.fps``.

    Steps:
    1. Resolve per-waypoint ``t`` and ``yaw`` per the course.
    2. Convert waypoint positions from ``course.frame`` to NED via the
       active FrameGraph.
    3. Fit a cubic spline through positions (not-a-knot bc) in NED.
    4. Linearly interpolate (with shortest-arc unwrap) the yaws.
    5. Sample at ``fps``; build quaternions in NED directly.
    """
    waypoint_ts = course.resolved_times()
    waypoint_yaws = course.resolved_yaws()
    # Unwrap to keep linear interp from "shortcutting" around 2π.
    waypoint_yaws_unwrapped = _unwrap_yaws_along_path(waypoint_yaws)

    # Convert waypoint positions to NED in bulk.
    src_pcd = PointCloud(
        points=course.positions,
        frame=frame_graph.frame(course.frame),
    )
    ned_pcd = frame_graph.convert(src_pcd, to="ned")
    positions_ned_wps = ned_pcd.points

    # Sample times at fps.
    n_samples = max(2, int(round(course.total_time_s * course.fps)) + 1)
    sample_t = np.linspace(0.0, course.total_time_s, n_samples)

    # Spline positions.
    cs = CubicSpline(waypoint_ts, positions_ned_wps, axis=0, bc_type="not-a-knot")
    sampled_positions_ned = cs(sample_t)
    sampled_velocities_ned = cs(sample_t, 1)  # 1st derivative

    # Yaws in `course.frame` — for tangent mode in mocap, the yaw refers to
    # mocap's notion of "facing direction". Convert to NED yaw via the
    # known yaw-flip (perm5: yaw_ned = -yaw_<mocap>).
    sampled_yaws_src = np.interp(sample_t, waypoint_ts, waypoint_yaws_unwrapped)
    sampled_yaws_ned = _to_ned_yaw(sampled_yaws_src, course.frame, frame_graph)
    quats_xyzw = np.stack([_yaw_to_quat_xyzw(y) for y in sampled_yaws_ned], axis=0)

    return TrainingTrajectory(
        times=sample_t,
        positions_ned=sampled_positions_ned,
        quaternions_xyzw=quats_xyzw,
        velocities_ned=sampled_velocities_ned,
        prompt=prompt,
        source=f"spline:{course.name}",
    )


def _to_ned_yaw(yaws_src: np.ndarray, src_frame: str, frame_graph: FrameGraph) -> np.ndarray:
    """Map yaws expressed in ``src_frame`` to NED yaws.

    For frames that share NED's z-axis (NED itself, "ned"-like custom
    frames): yaw_ned = yaw_src.

    For z-up frames related to NED by the standard perm5
    (``R_mocap_from_ned = diag(1, -1, -1)``): yaw_ned = -yaw_src.

    The check is empirical via the FrameGraph rotation matrix's z-axis
    sign, so a future scene that uses a different frame convention can
    drop into this helper without code changes.
    """
    if src_frame == "ned":
        return np.asarray(yaws_src, dtype=np.float64)
    T = frame_graph.transform(src_frame, "ned")
    # If the rotation flips z (R[2,2] == -1), yaw flips sign; if it preserves
    # z (R[2,2] == +1), yaw is preserved. We treat anything between as a
    # warning case — call sites can override by specifying yaws in NED
    # directly via a future ``yaw_frame`` field on Course.
    z_sign = float(np.sign(T.R[2, 2]))
    return z_sign * np.asarray(yaws_src, dtype=np.float64)
