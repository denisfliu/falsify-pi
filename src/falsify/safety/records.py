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
    CUSTOM = auto()


@dataclass
class FailureRecord:
    """Detailed record of a detected failure.

    `last_safe_state` is what SplatNav recovery plans from. Frame information
    is carried by the `DroneState`s themselves.
    """
    failure_type: FailureType
    description: str
    failure_step: int
    failure_state: DroneState
    last_safe_step: int
    last_safe_state: DroneState
    criterion_name: str = ""
    extra: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"{self.failure_type.name} via {self.criterion_name!r}: "
            f"{self.description} "
            f"(step {self.failure_step}; last safe step {self.last_safe_step})"
        )
