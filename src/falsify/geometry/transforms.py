"""Rigid (`SE3`) and similarity (`Sim3`) transforms.

Both are *frame-aware*: each carries the source frame it expects to consume
and the destination frame it produces. Applying a transform to the wrong
input frame raises immediately. Use the `@` operator: ``T @ point``.

Transforms compose left-to-right via `@`: ``T_b_from_a @ T_a_from_world``.

The implementation intentionally avoids inheritance — `SE3` and `Sim3` are
two independent dataclasses, and `@` dispatches by type. This keeps the
algebra explicit and the dataclass fields visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

from .frames import Frame
from .types import Point, Pose, Trajectory, PointCloud


# Anything a transform can be applied to.
Transformable = Union[Point, Pose, Trajectory, PointCloud]


@dataclass(frozen=True)
class SE3:
    """Rigid transform: ``p_dst = R @ p_src + t``.

    `src` is the frame of the input, `dst` is the frame of the output.
    """
    R: np.ndarray  # (3, 3)
    t: np.ndarray  # (3,)
    src: Frame
    dst: Frame

    def __post_init__(self):
        R = np.asarray(self.R, dtype=np.float64)
        t = np.asarray(self.t, dtype=np.float64)
        if R.shape != (3, 3):
            raise ValueError(f"SE3.R must be (3,3), got {R.shape}")
        if t.shape != (3,):
            raise ValueError(f"SE3.t must be (3,), got {t.shape}")
        object.__setattr__(self, "R", R)
        object.__setattr__(self, "t", t)

    def inv(self) -> "SE3":
        R_inv = self.R.T
        return SE3(R_inv, -R_inv @ self.t, src=self.dst, dst=self.src)

    def as_matrix(self) -> np.ndarray:
        M = np.eye(4)
        M[:3, :3] = self.R
        M[:3, 3] = self.t
        return M

    @classmethod
    def identity(cls, src: Frame, dst: Frame) -> "SE3":
        return cls(np.eye(3), np.zeros(3), src=src, dst=dst)

    # ---- application ----------------------------------------------------

    def __matmul__(self, other):
        if isinstance(other, SE3):
            return _se3_compose(self, other)
        if isinstance(other, Sim3):
            return _se3_after_sim3(self, other)
        return _apply(self, other)


@dataclass(frozen=True)
class Sim3:
    """Similarity transform: ``p_dst = s * (R @ p_src) + t``.

    Rotations and translations transform poses without scale; positions get
    the full similarity.
    """
    s: float
    R: np.ndarray  # (3, 3)
    t: np.ndarray  # (3,)
    src: Frame
    dst: Frame

    def __post_init__(self):
        R = np.asarray(self.R, dtype=np.float64)
        t = np.asarray(self.t, dtype=np.float64)
        if R.shape != (3, 3):
            raise ValueError(f"Sim3.R must be (3,3), got {R.shape}")
        if t.shape != (3,):
            raise ValueError(f"Sim3.t must be (3,), got {t.shape}")
        object.__setattr__(self, "s", float(self.s))
        object.__setattr__(self, "R", R)
        object.__setattr__(self, "t", t)

    def inv(self) -> "Sim3":
        s_inv = 1.0 / self.s
        R_inv = self.R.T
        return Sim3(s_inv, R_inv, -s_inv * (R_inv @ self.t), src=self.dst, dst=self.src)

    @classmethod
    def identity(cls, src: Frame, dst: Frame) -> "Sim3":
        return cls(1.0, np.eye(3), np.zeros(3), src=src, dst=dst)

    def __matmul__(self, other):
        if isinstance(other, Sim3):
            return _sim3_compose(self, other)
        if isinstance(other, SE3):
            return _sim3_after_se3(self, other)
        return _apply(self, other)


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def _check_compose(outer, inner) -> None:
    if inner.dst.name != outer.src.name:
        raise ValueError(
            f"cannot compose: inner produces {inner.dst.name!r} but outer expects {outer.src.name!r}"
        )


def _se3_compose(outer: SE3, inner: SE3) -> SE3:
    _check_compose(outer, inner)
    return SE3(
        R=outer.R @ inner.R,
        t=outer.R @ inner.t + outer.t,
        src=inner.src,
        dst=outer.dst,
    )


def _sim3_compose(outer: Sim3, inner: Sim3) -> Sim3:
    _check_compose(outer, inner)
    return Sim3(
        s=outer.s * inner.s,
        R=outer.R @ inner.R,
        t=outer.s * (outer.R @ inner.t) + outer.t,
        src=inner.src,
        dst=outer.dst,
    )


def _se3_after_sim3(outer: SE3, inner: Sim3) -> Sim3:
    """SE3 ∘ Sim3 = Sim3 with scale = inner.s."""
    _check_compose(outer, inner)
    return Sim3(
        s=inner.s,
        R=outer.R @ inner.R,
        t=outer.R @ inner.t + outer.t,
        src=inner.src,
        dst=outer.dst,
    )


def _sim3_after_se3(outer: Sim3, inner: SE3) -> Sim3:
    """Sim3 ∘ SE3 = Sim3 with scale = outer.s."""
    _check_compose(outer, inner)
    return Sim3(
        s=outer.s,
        R=outer.R @ inner.R,
        t=outer.s * (outer.R @ inner.t) + outer.t,
        src=inner.src,
        dst=outer.dst,
    )


# ---------------------------------------------------------------------------
# Application to frame-tagged values
# ---------------------------------------------------------------------------


def _apply(T, x: Transformable) -> Transformable:
    if not hasattr(x, "frame"):
        raise TypeError(
            f"Cannot apply transform to {type(x).__name__}: no .frame attribute. "
            f"Wrap raw arrays in Point/Pose/Trajectory/PointCloud first."
        )
    if x.frame.name != T.src.name:
        raise ValueError(
            f"frame mismatch applying transform: value is in {x.frame.name!r}, "
            f"transform consumes {T.src.name!r}"
        )

    is_sim3 = isinstance(T, Sim3)

    def _xform_pos(arr: np.ndarray) -> np.ndarray:
        rot = (T.R @ arr.T).T  # (..., 3)
        if is_sim3:
            rot = T.s * rot
        return rot + T.t  # broadcasts over leading dims

    def _xform_rot(R_local: np.ndarray) -> np.ndarray:
        return T.R @ R_local

    if isinstance(x, Point):
        return Point(_xform_pos(x.xyz[None, :])[0], frame=T.dst)

    if isinstance(x, Pose):
        return Pose(
            R=_xform_rot(x.R),
            t=_xform_pos(x.t[None, :])[0],
            frame=T.dst,
        )

    if isinstance(x, Trajectory):
        new_pos = _xform_pos(x.positions)
        new_vel = None
        if x.velocities is not None:
            # Velocity transforms with rotation+scale but no translation.
            v = (T.R @ x.velocities.T).T
            new_vel = T.s * v if is_sim3 else v
        new_q = None
        if x.quaternions is not None:
            # Quaternion transforms reflect the rotation only.
            # Apply R as a left-multiplication on the orientation.
            new_q = _rotate_quaternions(T.R, x.quaternions)
        return Trajectory(
            times=x.times.copy(),
            positions=new_pos,
            frame=T.dst,
            velocities=new_vel,
            quaternions=new_q,
        )

    if isinstance(x, PointCloud):
        return PointCloud(
            points=_xform_pos(x.points),
            frame=T.dst,
            colors=None if x.colors is None else x.colors.copy(),
        )

    raise TypeError(f"don't know how to apply transform to {type(x).__name__}")


def _rotate_quaternions(R: np.ndarray, q_xyzw: np.ndarray) -> np.ndarray:
    """Rotate a (N,4) xyzw quaternion array by a (3,3) rotation matrix.

    Implements q' = q_R ⊗ q, where q_R is the rotation matrix expressed as a
    quaternion. Local imports of scipy keep import-time costs low; this
    helper is only touched when trajectories include orientations.
    """
    from scipy.spatial.transform import Rotation as _R

    rot_R = _R.from_matrix(R)
    rot_q = _R.from_quat(q_xyzw)
    return (rot_R * rot_q).as_quat()
