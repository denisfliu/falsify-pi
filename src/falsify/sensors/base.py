"""Sensor abstraction + `SensorRig`.

A sensor produces one or more named modalities into an `ObservationBuilder`
each timestep. Sensors are independent and composable; the `SensorRig`
aggregates them and asserts that the policy's required keys are covered.

Design notes:
- Policies declare ``required_modalities: frozenset[str]``. The rig fails
  fast at construction if any key is unsupplied — the orchestrator never
  starts a rollout with a missing modality.
- Multiple sensors can coexist; each writes its own keys. Writing the same
  key twice is an error (single-writer-per-key invariant).
- Sensors that depend on the simulator (cameras) are typically *configured*
  with simulator-side handles (a renderer, a `FrameGraph`); they consume only
  the public `DroneState` at runtime.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Sequence

from falsify.policy.observation import Observation, ObservationBuilder
from falsify.sim.dynamics_state import DroneState


class Sensor(ABC):
    """One observation modality producer.

    Subclasses must declare:
    - ``keys_provided``: a constant frozenset of dotted-key names this sensor writes.
    - ``sense(state, builder)``: populates exactly those keys on the builder.
    """

    @property
    @abstractmethod
    def keys_provided(self) -> frozenset[str]:
        ...

    @abstractmethod
    def sense(self, state: DroneState, builder: ObservationBuilder) -> None:
        ...

    def reset(self) -> None:
        """Called once per episode before any ``sense()`` call."""
        return None


class SensorRig:
    """An ordered collection of sensors that builds one `Observation` per step.

    The rig is constructed once per episode (from scene + policy config). Use
    ``assert_covers(required_keys)`` immediately after construction so that
    missing-modality bugs surface before the rollout starts.
    """

    def __init__(self, sensors: Sequence[Sensor]) -> None:
        self._sensors: tuple[Sensor, ...] = tuple(sensors)
        self._verify_no_overlap()

    @property
    def sensors(self) -> tuple[Sensor, ...]:
        return self._sensors

    def keys_provided(self) -> frozenset[str]:
        out: set[str] = set()
        for s in self._sensors:
            out.update(s.keys_provided)
        return frozenset(out)

    def assert_covers(self, required: Iterable[str]) -> None:
        required_set = frozenset(required)
        missing = required_set - self.keys_provided()
        if missing:
            raise ValueError(
                f"SensorRig missing required modalities: {sorted(missing)}; "
                f"provided: {sorted(self.keys_provided())}"
            )

    def reset(self) -> None:
        for s in self._sensors:
            s.reset()

    def build(self, state: DroneState) -> Observation:
        builder = ObservationBuilder(state)
        for s in self._sensors:
            before = builder.keys_written()
            s.sense(state, builder)
            after = builder.keys_written()
            written = after - before
            expected = s.keys_provided
            unexpected = written - expected
            if unexpected:
                raise RuntimeError(
                    f"sensor {type(s).__name__} wrote unexpected keys "
                    f"{sorted(unexpected)}; declared {sorted(expected)}"
                )
        return builder.freeze()

    # ---- internals -----------------------------------------------------

    def _verify_no_overlap(self) -> None:
        seen: dict[str, type] = {}
        for s in self._sensors:
            for k in s.keys_provided:
                if k in seen:
                    raise ValueError(
                        f"sensor key conflict: {k!r} provided by both "
                        f"{seen[k].__name__} and {type(s).__name__}"
                    )
                seen[k] = type(s)
