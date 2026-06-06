"""Canonical yaw ↔ quaternion (xyzw) helpers.

Before this module, the same two-line conversion was redefined six times
across `policy/`, `planning/`, `training/`, `recovery/`, and one script.
Centralizing them removes the drift surface — sign / convention changes
should only ever touch one file.

Convention: quaternions are body-to-parent rotations in xyzw layout
(scalar last). Yaw is the body's heading angle about the parent +z axis,
extracted via the standard ZYX-Euler yaw formula. NED and MOCAP yaws
differ by a sign because their z-axes are opposite — that sign flip is
*not* applied here; callers handle it at the frame boundary (see
`policy/vla.py` and `policy/pi_gateway.py` for the canonical pattern).
"""

from __future__ import annotations

import numpy as np


def quat_to_yaw_xyzw(q: np.ndarray) -> float:
    """Extract yaw (rotation about parent +z) from a body-to-parent xyzw quat."""
    qx, qy, qz, qw = q
    return float(np.arctan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    ))


def yaw_to_quat_xyzw(yaw: float) -> np.ndarray:
    """Build an xyzw quaternion representing a pure yaw about parent +z."""
    return np.array([0.0, 0.0, np.sin(0.5 * yaw), np.cos(0.5 * yaw)])
