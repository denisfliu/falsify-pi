"""Simulator wrapper.

For v0 we keep this thin and avoid a hard dependency on ACADOS — the rollout
loop drives the drone forward by **following** the policy's emitted
`Trajectory` directly (a "trajectory replay" integrator). The full FiGS MPC
+ ACADOS dynamics path will replace this once the env is fully set up;
the public API here is stable enough that swapping integrators won't churn
callers.

Frame contract
--------------
- All `DroneState`s in/out are in ``"ned"``.
- The simulator takes a `FrameGraph` so future sub-components (forces,
  collisions) can resolve world-aligned frames without hardcoding names.

Open-loop note
--------------
The mock policies in `falsify.policy.mock` emit `Trajectory` whose frame is
the same as the input state's frame. We sample the trajectory at the
controller rate (`hz`) and use the sample as the next state. This is enough
to exercise the sensor → policy → safety → recovery pipeline end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from falsify.geometry import FrameGraph, Point, Trajectory, assert_frame
from .dynamics_state import DroneState

# Avoid a circular import: `policy.observation` imports `DroneState` from
# this package. Policy + SensorRig are duck-typed here at call sites, so
# we only need them for type hints.
if TYPE_CHECKING:
    from falsify.policy import Policy
    from falsify.sensors import SensorRig
    from falsify.safety import FailureDetector, FailureRecord
    from falsify.perturbations import PerturbationSuite


@dataclass
class SimulatorConfig:
    hz: int = 10                # control / rollout rate
    horizon_s: float = 10.0     # episode time budget
    policy_hz: int = 1          # how often to re-query the policy (chunked)


@dataclass
class EpisodeTrace:
    """In-order log of `DroneState` snapshots and the policy outputs that
    drove each chunk. Frame-tagged throughout."""
    states: list[DroneState] = field(default_factory=list)
    policy_outputs: list[Trajectory] = field(default_factory=list)
    chunk_starts: list[int] = field(default_factory=list)
    failure: Optional["FailureRecord"] = None

    def trajectory(self) -> Trajectory:
        """Public access path — returns a frame-tagged `Trajectory`.

        Prefer this over reading state-list internals directly so the frame
        tag travels with the data.
        """
        if not self.states:
            raise ValueError("empty trace")
        return Trajectory(
            times=np.array([s.t for s in self.states]),
            positions=np.stack([s.pos.xyz for s in self.states]),
            frame=self.states[0].pos.frame,
            velocities=np.stack([s.vel for s in self.states]),
            quaternions=np.stack([s.quat_xyzw for s in self.states]),
        )


class Simulator:
    """Trajectory-replay simulator wrapper.

    A drop-in MPC-backed integrator will replace `_step_replay` once acados
    is available; everything else is integrator-agnostic.
    """

    def __init__(
        self,
        cfg: SimulatorConfig,
        frame_graph: FrameGraph,
    ) -> None:
        self.cfg = cfg
        self.frame_graph = frame_graph
        self._state: Optional[DroneState] = None

    # ---- lifecycle -----------------------------------------------------

    def reset(self, initial_state: DroneState) -> DroneState:
        assert_frame(initial_state.pos, "ned")
        self._state = initial_state
        return initial_state

    @property
    def state(self) -> DroneState:
        if self._state is None:
            raise RuntimeError("Simulator.reset() must be called before use")
        return self._state

    # ---- rollout -------------------------------------------------------

    def rollout_with_policy(
        self,
        policy: "Policy",
        sensor_rig: "SensorRig",
        max_steps: Optional[int] = None,
        detector: Optional["FailureDetector"] = None,
        perturbations: Optional["PerturbationSuite"] = None,
    ) -> EpisodeTrace:
        """Run one episode. Returns the time-stamped trace.

        - `policy` declares its `required_modalities`; the caller must have
          wired matching sensors into `sensor_rig`.
        - `sensor_rig.assert_covers(policy.required_modalities)` is checked
          once at the start of the rollout.
        - If `detector` is given, the loop calls `detector.update(state, step)`
          after each integration step and stops on the first failure. The
          `FailureRecord` is attached to the returned trace.
        """
        if self._state is None:
            raise RuntimeError("call reset() before rollout_with_policy")
        sensor_rig.assert_covers(policy.required_modalities)
        sensor_rig.reset()
        policy.reset()
        if detector is not None:
            detector.reset()
        if perturbations is not None:
            perturbations.reset()

        dt = 1.0 / self.cfg.hz
        if max_steps is None:
            max_steps = int(self.cfg.horizon_s * self.cfg.hz)
        re_query_every = max(1, int(self.cfg.hz / self.cfg.policy_hz))

        trace = EpisodeTrace()
        active_chunk: Optional[Trajectory] = None
        chunk_offset = 0

        for step in range(max_steps):
            # Re-query policy at the policy rate.
            if step % re_query_every == 0:
                obs = sensor_rig.build(self._state)
                if perturbations is not None:
                    obs = perturbations.apply_observation(obs)
                active_chunk = policy.observe(obs)
                assert_frame(active_chunk, "ned")
                if perturbations is not None:
                    active_chunk = perturbations.apply_action(active_chunk, self.frame_graph)
                    assert_frame(active_chunk, "ned")
                trace.policy_outputs.append(active_chunk)
                trace.chunk_starts.append(step)
                chunk_offset = 0

            trace.states.append(self._state)
            if detector is not None:
                rec = detector.update(self._state, step)
                if rec is not None:
                    trace.failure = rec
                    return trace
            self._state = self._step_replay(self._state, active_chunk, chunk_offset, dt)
            chunk_offset += 1

        # Final state.
        trace.states.append(self._state)
        if detector is not None:
            detector.update(self._state, max_steps)
            trace.failure = detector.fired
        return trace

    # ---- integrator (v0: trajectory replay) ---------------------------

    @staticmethod
    def _step_replay(
        state: DroneState,
        chunk: Trajectory,
        offset: int,
        dt: float,
    ) -> DroneState:
        """Advance one timestep by sampling the policy's trajectory chunk.

        Linear interpolation between adjacent waypoints by index — adequate
        for smoke testing. Once the FiGS MPC integrator is wired, this is
        replaced with a closed-loop dynamics step.
        """
        n = len(chunk)
        if n == 0:
            return _advance_dt(state, dt)
        idx = min(offset, n - 1)
        new_pos = Point(chunk.positions[idx], frame=chunk.frame)
        new_vel = chunk.velocities[idx] if chunk.velocities is not None else np.zeros(3)
        # Orientation: identity for replay; the MPC-backed integrator will
        # properly integrate body rates and update the quaternion.
        new_quat = state.quat_xyzw
        return DroneState(
            pos=new_pos, vel=new_vel, quat_xyzw=new_quat, t=state.t + dt,
        )


def _advance_dt(state: DroneState, dt: float) -> DroneState:
    """Fallback: hold position, advance time."""
    return DroneState(
        pos=state.pos, vel=state.vel, quat_xyzw=state.quat_xyzw, t=state.t + dt,
    )
