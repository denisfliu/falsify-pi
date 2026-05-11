"""`Policy` ABC.

A policy consumes an `Observation` and emits a `Trajectory` in the NED frame
(``Trajectory.frame.name == "ned"``). It declares its modality requirements
up front so the orchestrator can wire the matching sensors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from falsify.geometry import Trajectory
from .observation import Observation


class Policy(ABC):
    """Base class for all policies.

    Subclasses must set ``required_modalities`` (a class-level frozenset of
    dotted-key sensor names) and implement ``observe``.
    """

    required_modalities: frozenset[str] = frozenset()

    @abstractmethod
    def observe(self, obs: Observation) -> Trajectory:
        """Emit the next reference trajectory in the NED frame."""
        ...

    def reset(self) -> None:
        """Called once at episode start before any ``observe`` call."""
        return None
