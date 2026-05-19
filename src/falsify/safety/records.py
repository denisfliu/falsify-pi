"""Failure-record data types shared between criteria, detector, and orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from falsify.sim.dynamics_state import DroneState


class FailureType(Enum):
    NONE = auto()
    OUT_OF_BOUNDS = auto()
    EXCESSIVE_VELOCITY = auto()
    EXCESSIVE_TILT = auto()
    PROXIMITY_COLLISION = auto()
    COLLISION_GATE = auto()
    COLLISION_OTHER = auto()
    MISS_GATE = auto()
    # Drone successfully transited the gate aperture but didn't reach the
    # task-completion ("hover-over-stuffed-animal") goal within the run
    # — i.e. the post-gate hover failed. Distinct from MISS_GATE because
    # the gate crossing did succeed.
    GOAL_NOT_REACHED = auto()
    CUSTOM = auto()


@dataclass
class FailureRecord:
    """Detailed record of a detected failure.

    `last_safe_state` is the most-recent safe state, the natural default
    seed for recovery. `safe_history` carries every safe (step, DroneState)
    pair the detector saw before the failure — the recovery layer samples
    from this list with a failure-type-aware bias (e.g. earlier for
    miss-gate / non-gate collision, later for gate clips).
    """
    failure_type: FailureType
    description: str
    failure_step: int
    failure_state: DroneState
    last_safe_step: int
    last_safe_state: DroneState
    criterion_name: str = ""
    extra: dict = field(default_factory=dict)
    safe_history: list[tuple[int, DroneState]] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"{self.failure_type.name} via {self.criterion_name!r}: "
            f"{self.description} "
            f"(step {self.failure_step}; last safe step {self.last_safe_step})"
        )
