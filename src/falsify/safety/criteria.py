"""Pluggable safety criteria.

Each criterion votes on whether a `DroneState` is safe and emits a small
"violation" description if not. The detector fuses votes (any failure
flips the verdict to unsafe).

Frame-aware design
------------------
Bounds are declared in a named frame (typically ``"ned"`` for the simulator's
native frame, but anything in the active `FrameGraph` works). The criterion
converts the incoming state into its frame via the graph and operates there.
Adding a new criterion = subclassing `SafetyCriterion` and declaring its
``operates_in_frame``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as _R

from falsify.geometry import FrameGraph, Point, assert_frame
from falsify.sim.dynamics_state import DroneState


@dataclass
class Violation:
    description: str
    value: float
    threshold: float


class SafetyCriterion(ABC):
    """Base class for safety checks.

    Subclasses declare the frame they operate in and implement `check`.
    """

    operates_in_frame: str = "ned"
    name: str = ""

    @abstractmethod
    def check(self, state_in_frame: DroneState) -> Optional[Violation]:
        """Return a `Violation` if the state is unsafe, else None."""
        ...

    def check_with_graph(
        self, state: DroneState, frame_graph: FrameGraph
    ) -> Optional[Violation]:
        """Convert `state` into this criterion's frame, then `check`.

        Velocity/quaternion don't currently get re-rotated when the state
        crosses frames — for v0 the rule is that criteria pick frames where
        these are interpretable directly (e.g. velocity criterion in NED).
        Specialized criteria that need transformed velocity should override
        this method.
        """
        if state.pos.frame.name == self.operates_in_frame:
            return self.check(state)
        # Convert position only; velocity stays in its original frame and is
        # only used by criteria that don't care about cross-frame velocities.
        new_pos = frame_graph.convert(state.pos, to=self.operates_in_frame)
        moved = DroneState(
            pos=new_pos, vel=state.vel, quat_xyzw=state.quat_xyzw, t=state.t,
        )
        return self.check(moved)


class BoundsCriterion(SafetyCriterion):
    """Axis-aligned-box safety bounds in a named frame.

    Lower and upper corners are passed as `Point`s — the **frame is read
    from the points themselves**, eliminating the possibility of accidentally
    specifying bounds in the wrong frame. Both points must share a frame.
    """
    name: str = "bounds"

    def __init__(self, lower: Point, upper: Point):
        if lower.frame.name != upper.frame.name:
            raise ValueError(
                f"BoundsCriterion: lower/upper frames disagree "
                f"({lower.frame.name!r} vs {upper.frame.name!r})"
            )
        self.lower_pt = lower
        self.upper_pt = upper
        self.operates_in_frame = lower.frame.name

    def check(self, state_in_frame: DroneState) -> Optional[Violation]:
        p = state_in_frame.pos.xyz
        lo = self.lower_pt.xyz
        hi = self.upper_pt.xyz
        if np.any(p < lo) or np.any(p > hi):
            worst = float(max(np.max(lo - p), np.max(p - hi)))
            return Violation(
                description=f"position {p.tolist()} outside [{lo.tolist()}, {hi.tolist()}] in frame {self.operates_in_frame!r}",
                value=worst,
                threshold=0.0,
            )
        return None


@dataclass
class VelocityCriterion(SafetyCriterion):
    """Speed (||v||) exceeds threshold."""
    max_speed: float = 5.0
    operates_in_frame: str = "ned"
    name: str = "velocity"

    def check(self, state_in_frame: DroneState) -> Optional[Violation]:
        speed = float(np.linalg.norm(state_in_frame.vel))
        if speed > self.max_speed:
            return Violation(
                description=f"speed {speed:.3f} m/s exceeds {self.max_speed:.3f}",
                value=speed,
                threshold=self.max_speed,
            )
        return None


@dataclass
class TiltCriterion(SafetyCriterion):
    """Body roll/pitch magnitude exceeds threshold (radians).

    The check is frame-independent (it's an internal body orientation), but
    we still declare ``operates_in_frame == "ned"`` for consistency.
    """
    max_tilt_rad: float = 1.2   # ~70 degrees
    operates_in_frame: str = "ned"
    name: str = "tilt"

    def check(self, state_in_frame: DroneState) -> Optional[Violation]:
        rot = _R.from_quat(state_in_frame.quat_xyzw)
        # World-z axis expressed in body frame after rotation; tilt is its
        # angle from world-z.
        world_z = np.array([0.0, 0.0, 1.0])
        body_z = rot.apply(world_z, inverse=True)
        cos_tilt = float(np.clip(np.dot(body_z, world_z), -1.0, 1.0))
        tilt = float(np.arccos(cos_tilt))
        if tilt > self.max_tilt_rad:
            return Violation(
                description=f"tilt {tilt:.3f} rad exceeds {self.max_tilt_rad:.3f}",
                value=tilt,
                threshold=self.max_tilt_rad,
            )
        return None
