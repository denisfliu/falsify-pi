"""`StateSensor` — exposes the drone position as a frame-tagged observation key.

`DroneState` itself is always on `Observation.state` regardless of sensors —
no sensor is needed to access `obs.state.pos`. This sensor exists so policies
that prefer dotted-key access can require ``"state.pos"`` (a `Point`) and have
the sensor rig assert its presence at construction time.

We intentionally do **not** mirror `vel`, `quat`, or `t` as dotted keys: bare
ndarrays in the observation dict would lose their frame context. Consumers
access those fields via ``obs.state.vel`` / ``obs.state.quat_xyzw`` / ``obs.state.t``,
all of which share ``obs.state.frame`` as the single source of truth.
"""

from __future__ import annotations

from .base import Sensor
from falsify.policy.observation import ObservationBuilder
from falsify.sim.dynamics_state import DroneState


class StateSensor(Sensor):
    KEYS = frozenset({"state.pos"})

    @property
    def keys_provided(self) -> frozenset[str]:
        return self.KEYS

    def sense(self, state: DroneState, builder: ObservationBuilder) -> None:
        builder.set("state.pos", state.pos)   # frame-tagged Point
