"""Frame-tagged geometry types.

Every position / pose / trajectory / point cloud that crosses a module
boundary in falsify carries its `Frame`. The `assert_frame()` helper is the
one-line guard call-sites use to reject inputs in the wrong frame.

Internal helpers within a single module may operate on bare `np.ndarray`s for
speed, but the moment a value is returned to a caller it is wrapped in one of
these types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .frames import Frame


def _as_array(x, shape: tuple[int, ...] | tuple[int, ...] | None, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if shape is not None and arr.shape != shape:
        raise ValueError(f"{name} expected shape {shape}, got {arr.shape}")
    return arr


@dataclass(frozen=True)
class Point:
    """A single 3D position in a named frame."""
    xyz: np.ndarray  # (3,)
    frame: Frame

    def __post_init__(self):
        object.__setattr__(self, "xyz", _as_array(self.xyz, (3,), "Point.xyz"))

    @classmethod
    def of(cls, x: float, y: float, z: float, frame: Frame) -> "Point":
        return cls(np.array([x, y, z], dtype=np.float64), frame)


@dataclass(frozen=True)
class Pose:
    """An SE(3) pose: rotation R and translation t, both in the named frame.

    Interpreted as a camera-to-world style pose: a vector ``v_local`` in the
    local frame attached to this pose maps to the parent frame via
    ``R @ v_local + t``. The `frame` field is the *parent* frame.
    """
    R: np.ndarray  # (3, 3)
    t: np.ndarray  # (3,)
    frame: Frame

    def __post_init__(self):
        object.__setattr__(self, "R", _as_array(self.R, (3, 3), "Pose.R"))
        object.__setattr__(self, "t", _as_array(self.t, (3,), "Pose.t"))

    def as_matrix(self) -> np.ndarray:
        """Return the 4x4 homogeneous matrix."""
        M = np.eye(4)
        M[:3, :3] = self.R
        M[:3, 3] = self.t
        return M

    @classmethod
    def from_matrix(cls, M: np.ndarray, frame: Frame) -> "Pose":
        M = _as_array(M, (4, 4), "Pose.from_matrix")
        return cls(M[:3, :3].copy(), M[:3, 3].copy(), frame)


@dataclass(frozen=True)
class Trajectory:
    """A time-stamped sequence of states in one named frame.

    ``positions`` is (N, 3), required. ``velocities`` and ``quaternions``
    (xyzw) are optional and either both populated or None. ``times`` is (N,).
    """
    times: np.ndarray
    positions: np.ndarray
    frame: Frame
    velocities: Optional[np.ndarray] = None
    quaternions: Optional[np.ndarray] = None

    def __post_init__(self):
        times = _as_array(self.times, None, "Trajectory.times")
        positions = _as_array(self.positions, None, "Trajectory.positions")
        if times.ndim != 1:
            raise ValueError(f"times must be 1D, got shape {times.shape}")
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"positions must be (N,3), got {positions.shape}")
        if times.shape[0] != positions.shape[0]:
            raise ValueError(
                f"times/positions length mismatch: {times.shape[0]} vs {positions.shape[0]}"
            )
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "positions", positions)
        if self.velocities is not None:
            v = _as_array(self.velocities, None, "Trajectory.velocities")
            if v.shape != positions.shape:
                raise ValueError(f"velocities shape {v.shape} != positions {positions.shape}")
            object.__setattr__(self, "velocities", v)
        if self.quaternions is not None:
            q = _as_array(self.quaternions, None, "Trajectory.quaternions")
            if q.ndim != 2 or q.shape[1] != 4 or q.shape[0] != positions.shape[0]:
                raise ValueError(
                    f"quaternions must be (N,4) xyzw matching positions; got {q.shape}"
                )
            object.__setattr__(self, "quaternions", q)

    def __len__(self) -> int:
        return self.positions.shape[0]


@dataclass(frozen=True)
class PointCloud:
    """A (possibly colored) point cloud in one named frame."""
    points: np.ndarray         # (N, 3)
    frame: Frame
    colors: Optional[np.ndarray] = None  # (N, 3) in [0, 1] or [0, 255]

    def __post_init__(self):
        pts = _as_array(self.points, None, "PointCloud.points")
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"points must be (N,3), got {pts.shape}")
        object.__setattr__(self, "points", pts)
        if self.colors is not None:
            c = _as_array(self.colors, None, "PointCloud.colors")
            if c.shape != pts.shape:
                raise ValueError(f"colors shape {c.shape} != points {pts.shape}")
            object.__setattr__(self, "colors", c)

    def __len__(self) -> int:
        return self.points.shape[0]


# ---------------------------------------------------------------------------
# Frame-tag enforcement
# ---------------------------------------------------------------------------


def assert_frame(value, expected: Frame | str) -> None:
    """Raise if ``value.frame.name`` does not equal ``expected``.

    Accepts any of `Point`, `Pose`, `Trajectory`, `PointCloud`, or any object
    with a ``.frame`` attribute. The second argument may be a `Frame` or a
    bare string for convenience at call sites.
    """
    if not hasattr(value, "frame"):
        raise TypeError(f"value of type {type(value).__name__} has no .frame attribute")
    actual_name = value.frame.name
    expected_name = expected.name if isinstance(expected, Frame) else str(expected)
    if actual_name != expected_name:
        raise ValueError(
            f"frame mismatch: got {actual_name!r}, expected {expected_name!r} "
            f"(value type: {type(value).__name__})"
        )
