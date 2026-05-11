"""`Observation` and `ObservationBuilder`.

An `Observation` is what a `Policy` sees. It carries:
- The drone state (always present).
- A free-form dictionary of named modalities populated by the sensor rig.
- An optional text prompt.

Modalities are addressed by dotted keys (``"images.forward"``, ``"depth.downward"``,
``"state.battery_voltage"``, …) so the sensor system can produce arbitrary
data without churning the type. Policies declare exactly which keys they
need via ``Policy.required_modalities`` and either read with ``.require(key)``
(raises on missing) or ``.get(key)`` (None on missing).

`ObservationBuilder` is the write-side surface a `Sensor.sense()` call
populates. The orchestrator instantiates one builder per timestep, lets each
sensor fill its keys, then freezes it into an `Observation`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from falsify.sim.dynamics_state import DroneState


@dataclass(frozen=True)
class Observation:
    state: DroneState
    data: dict[str, Any] = field(default_factory=dict)
    prompt: str = ""

    # ---- read API used by policies -------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def require(self, key: str) -> Any:
        if key not in self.data:
            raise KeyError(
                f"required observation key {key!r} not present; "
                f"available keys: {sorted(self.data)}"
            )
        return self.data[key]

    def keys(self) -> Iterable[str]:
        return self.data.keys()


class ObservationBuilder:
    """Write-side handle for sensors.

    Sensors call ``set("images.forward", arr)``. The builder enforces that no
    key is written twice (a configuration mistake) and tracks which keys were
    actually produced for coverage assertions.
    """

    def __init__(self, state: DroneState) -> None:
        self._state = state
        self._data: dict[str, Any] = {}
        self._prompt: str = ""

    @property
    def state(self) -> DroneState:
        return self._state

    def set(self, key: str, value: Any) -> None:
        if key in self._data:
            raise ValueError(
                f"observation key {key!r} already set; sensor pipeline must be "
                f"single-writer per key"
            )
        self._data[key] = value

    def set_prompt(self, prompt: str) -> None:
        self._prompt = prompt

    def keys_written(self) -> frozenset[str]:
        return frozenset(self._data)

    def freeze(self) -> Observation:
        return Observation(state=self._state, data=dict(self._data), prompt=self._prompt)
