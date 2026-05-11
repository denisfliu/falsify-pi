"""Mock policy tests — straight line + noisy variant."""

from __future__ import annotations

import numpy as np
import pytest

from falsify.geometry import NED, Point
from falsify.policy import (
    MockStraightLine, MockStraightLineConfig,
    MockNoisy, MockNoisyConfig,
    Observation,
)
from falsify.sensors import SensorRig, StateSensor
from falsify.sim.dynamics_state import DroneState


def _obs(pos, t=0.0):
    state = DroneState(
        pos=pos, vel=np.zeros(3), quat_xyzw=np.array([0, 0, 0, 1.0]), t=t,
    )
    return SensorRig([StateSensor()]).build(state)


def test_straight_line_emits_trajectory_toward_goal():
    cfg = MockStraightLineConfig(
        goal=Point.of(10.0, 0.0, 0.0, NED), speed=1.0, horizon_s=2.0, n_waypoints=5,
    )
    pol = MockStraightLine(cfg)
    obs = _obs(Point.of(0.0, 0.0, 0.0, NED))
    traj = pol.observe(obs)
    assert traj.frame is NED
    assert traj.positions.shape == (5, 3)
    # Last waypoint should be along +x at speed * horizon = 2m
    np.testing.assert_allclose(traj.positions[-1], [2.0, 0.0, 0.0], atol=1e-12)
    # Velocity points along +x at unit speed
    np.testing.assert_allclose(traj.velocities[0], [1.0, 0.0, 0.0], atol=1e-12)


def test_straight_line_clamps_to_goal():
    cfg = MockStraightLineConfig(
        goal=Point.of(0.5, 0.0, 0.0, NED), speed=1.0, horizon_s=10.0, n_waypoints=4,
    )
    pol = MockStraightLine(cfg)
    obs = _obs(Point.of(0.0, 0.0, 0.0, NED))
    traj = pol.observe(obs)
    # The reach is 10m but distance to goal is 0.5m — so endpoint is the goal.
    np.testing.assert_allclose(traj.positions[-1], [0.5, 0.0, 0.0], atol=1e-12)


def test_straight_line_rejects_frame_mismatch():
    from falsify.geometry import MOCAP
    cfg = MockStraightLineConfig(goal=Point.of(1.0, 1.0, 1.0, MOCAP))
    pol = MockStraightLine(cfg)
    obs = _obs(Point.of(0.0, 0.0, 0.0, NED))
    with pytest.raises(ValueError, match="goal frame"):
        pol.observe(obs)


def test_noisy_policy_is_reproducible():
    cfg = MockNoisyConfig(
        goal=Point.of(5.0, 0.0, 0.0, NED), speed=1.0,
        horizon_s=1.0, n_waypoints=10, position_noise_std=0.1, seed=42,
    )
    obs = _obs(Point.of(0.0, 0.0, 0.0, NED))
    a = MockNoisy(cfg).observe(obs)
    b = MockNoisy(cfg).observe(obs)
    np.testing.assert_allclose(a.positions, b.positions)


def test_mock_declares_no_modalities():
    assert MockStraightLine.required_modalities == frozenset()
    assert MockNoisy.required_modalities == frozenset()
