"""`FailureDetector` — fuses safety-criterion votes and tracks last-safe state.

Typical usage from the simulator's rollout loop::

    detector = FailureDetector(criteria, frame_graph)
    for step in range(N):
        state = sim.step(...)
        rec = detector.update(state, step)
        if rec is not None:
            break

`rec.last_safe_state` is the input to the recovery planner.
"""

from __future__ import annotations

from typing import Optional, Sequence

from falsify.geometry import FrameGraph
from falsify.sim.dynamics_state import DroneState
from .criteria import SafetyCriterion
from .records import FailureRecord, FailureType


# Mapping from criterion-name → FailureType. Custom criteria default to CUSTOM.
_NAME_TO_TYPE = {
    "bounds": FailureType.OUT_OF_BOUNDS,
    "velocity": FailureType.EXCESSIVE_VELOCITY,
    "tilt": FailureType.EXCESSIVE_TILT,
    "proximity": FailureType.PROXIMITY_COLLISION,
    "collision_gate": FailureType.COLLISION_GATE,
    "collision_other": FailureType.COLLISION_OTHER,
    "miss_gate": FailureType.MISS_GATE,
}


class FailureDetector:
    def __init__(
        self,
        criteria: Sequence[SafetyCriterion],
        frame_graph: FrameGraph,
    ) -> None:
        self._criteria = tuple(criteria)
        self._graph = frame_graph
        self._last_safe_state: Optional[DroneState] = None
        self._last_safe_step: int = -1
        self._fired: Optional[FailureRecord] = None
        # Ordered list of every safe (step, state) the detector has seen this
        # episode. The recovery layer samples from it with a failure-type bias.
        self._safe_history: list[tuple[int, DroneState]] = []

    @property
    def criteria(self) -> tuple[SafetyCriterion, ...]:
        return self._criteria

    @property
    def fired(self) -> Optional[FailureRecord]:
        return self._fired

    @property
    def safe_history(self) -> list[tuple[int, DroneState]]:
        return list(self._safe_history)

    def reset(self) -> None:
        self._last_safe_state = None
        self._last_safe_step = -1
        self._fired = None
        self._safe_history = []
        for crit in self._criteria:
            crit.reset()

    def update(self, state: DroneState, step: int) -> Optional[FailureRecord]:
        """Vote on `state`. On first failure, build the `FailureRecord` and
        return it; subsequent calls return the cached record.

        State considered safe ⇒ becomes the new last-safe state.
        """
        if self._fired is not None:
            return self._fired

        for crit in self._criteria:
            violation = crit.check_with_graph(state, self._graph)
            if violation is not None:
                ftype = (
                    violation.failure_type
                    if violation.failure_type is not None
                    else _NAME_TO_TYPE.get(crit.name, FailureType.CUSTOM)
                )
                extra = {
                    "value": violation.value,
                    "threshold": violation.threshold,
                }
                extra.update(violation.extra)
                rec = FailureRecord(
                    failure_type=ftype,
                    description=violation.description,
                    failure_step=step,
                    failure_state=state,
                    last_safe_step=self._last_safe_step,
                    last_safe_state=self._last_safe_state if self._last_safe_state is not None else state,
                    criterion_name=crit.name,
                    extra=extra,
                    safe_history=list(self._safe_history),
                )
                self._fired = rec
                return rec

        self._last_safe_state = state
        self._last_safe_step = step
        self._safe_history.append((step, state))
        return None
