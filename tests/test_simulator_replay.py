"""Trajectory-replay simulator smoke test (no FiGS / no GPU required).

Verifies the sim + sensor + policy + state pipeline runs end-to-end while
preserving frame tags on every value that crosses a module boundary.
"""

from __future__ import annotations

import numpy as np

from falsify.geometry import NED, FrameGraph, Point, Trajectory, SE3, assert_frame
from falsify.policy import MockStraightLine, MockStraightLineConfig
from falsify.sensors import SensorRig, StateSensor
from falsify.sim import DroneState, Simulator, SimulatorConfig


def _minimal_graph() -> FrameGraph:
    g = FrameGraph()
    g.register_frame(NED)
    return g


def test_simulator_rollout_with_mock_policy_preserves_frames():
    sim = Simulator(SimulatorConfig(hz=10, horizon_s=2.0, policy_hz=1), _minimal_graph())
    start = DroneState(
        pos=Point.of(0.0, 0.0, 0.0, NED),
        vel=np.zeros(3),
        quat_xyzw=np.array([0, 0, 0, 1.0]),
        t=0.0,
    )
    sim.reset(start)

    policy = MockStraightLine(MockStraightLineConfig(
        goal=Point.of(5.0, 0.0, 0.0, NED), speed=1.0, horizon_s=2.0, n_waypoints=20,
    ))
    sensor_rig = SensorRig([StateSensor()])

    trace = sim.rollout_with_policy(policy, sensor_rig)
    assert len(trace.states) > 0
    # Every state in the trace carries the NED frame tag.
    for s in trace.states:
        assert s.pos.frame is NED
    # Every policy output carries a frame tag (NED).
    for t in trace.policy_outputs:
        assert_frame(t, "ned")
    # End position has progressed toward the goal.
    assert trace.states[-1].pos.xyz[0] > 0.0


def test_simulator_asserts_required_modalities():
    import pytest
    sim = Simulator(SimulatorConfig(hz=10, horizon_s=0.5), _minimal_graph())
    sim.reset(DroneState(
        pos=Point.of(0, 0, 0, NED),
        vel=np.zeros(3),
        quat_xyzw=np.array([0, 0, 0, 1.0]),
        t=0.0,
    ))

    class HungryPolicy(MockStraightLine):
        required_modalities = frozenset({"images.forward"})

    pol = HungryPolicy(MockStraightLineConfig(goal=Point.of(1, 0, 0, NED)))
    rig = SensorRig([StateSensor()])   # no camera
    with pytest.raises(ValueError, match="missing required modalities"):
        sim.rollout_with_policy(pol, rig)


def test_trace_trajectory_is_frame_tagged():
    sim = Simulator(SimulatorConfig(hz=20, horizon_s=1.0, policy_hz=2), _minimal_graph())
    sim.reset(DroneState(
        pos=Point.of(0, 0, 0, NED), vel=np.zeros(3),
        quat_xyzw=np.array([0, 0, 0, 1.0]), t=0.0,
    ))
    pol = MockStraightLine(MockStraightLineConfig(goal=Point.of(2, 0, 0, NED)))
    trace = sim.rollout_with_policy(pol, SensorRig([StateSensor()]))
    full = trace.trajectory()
    assert isinstance(full, Trajectory)
    assert full.frame is NED
