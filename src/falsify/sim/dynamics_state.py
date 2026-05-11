"""`DroneState` — the canonical, frame-tagged drone-state value type.

The underlying FiGS state vector `[px, py, pz, vx, vy, vz, qx, qy, qz, qw]`
is encapsulated; nothing outside `sim/` should see the raw 10-vector.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from falsify.geometry import Frame, Point, NED


@dataclass(frozen=True)
class DroneState:
    """Frame-tagged drone state.

    Components share a single frame, discoverable via either ``state.pos.frame``
    or the ``state.frame`` property — they always travel together as one
    dataclass, never as bare arrays across module boundaries.

    - `pos`: position as a `Point`. Determines the frame of the whole state.
    - `vel`: linear velocity (3,) expressed in ``pos.frame``. Bare ndarray
      because it never leaves this dataclass; consumers always pull
      `state.vel` *together with* `state.frame`.
    - `quat_xyzw`: orientation as a unit quaternion (x, y, z, w) mapping
      ``cam_body`` axes into ``pos.frame``. Bare ndarray for the same reason.
    - `t`: simulation time in seconds (scalar, frame-independent).
    """

    pos: Point
    vel: np.ndarray
    quat_xyzw: np.ndarray
    t: float

    def __post_init__(self):
        v = np.asarray(self.vel, dtype=np.float64)
        q = np.asarray(self.quat_xyzw, dtype=np.float64)
        if v.shape != (3,):
            raise ValueError(f"vel must be (3,), got {v.shape}")
        if q.shape != (4,):
            raise ValueError(f"quat_xyzw must be (4,), got {q.shape}")
        object.__setattr__(self, "vel", v)
        object.__setattr__(self, "quat_xyzw", q)

    @property
    def frame(self) -> Frame:
        return self.pos.frame

    @classmethod
    def from_vector(cls, x: np.ndarray, t: float, frame: Frame = NED) -> "DroneState":
        """Build from the FiGS state convention ``[p(3), v(3), q_xyzw(4)]``."""
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (10,):
            raise ValueError(f"FiGS state must be (10,), got {x.shape}")
        return cls(
            pos=Point(x[0:3], frame=frame),
            vel=x[3:6],
            quat_xyzw=x[6:10],
            t=float(t),
        )

    def to_vector(self) -> np.ndarray:
        """Pack back into the FiGS 10-vector layout."""
        return np.concatenate([self.pos.xyz, self.vel, self.quat_xyzw])
