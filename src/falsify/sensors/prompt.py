"""`PromptSensor` — emits a static text prompt every step (VLA policies)."""

from __future__ import annotations

from .base import Sensor
from falsify.policy.observation import ObservationBuilder
from falsify.sim.dynamics_state import DroneState


class PromptSensor(Sensor):
    """Sets ``Observation.prompt`` (not a data key) from a fixed string."""

    KEYS: frozenset[str] = frozenset()

    def __init__(self, prompt: str) -> None:
        self._prompt = str(prompt)

    @property
    def keys_provided(self) -> frozenset[str]:
        return self.KEYS

    def sense(self, state: DroneState, builder: ObservationBuilder) -> None:
        builder.set_prompt(self._prompt)
