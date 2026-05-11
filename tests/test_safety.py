"""Failure detector + criteria tests."""

from __future__ import annotations

import numpy as np
import pytest

import numpy as np
from falsify.geometry import NED, FrameGraph, Point
from falsify.safety import (
    BoundsCriterion, VelocityCriterion, TiltCriterion,
    FailureDetector, FailureType,
)
from falsify.sim.dynamics_state import DroneState


def _state(pos=(0, 0, 0), vel=(0, 0, 0), q=(0, 0, 0, 1.0), t=0.0):
    return DroneState(
        pos=Point.of(*pos, NED),
        vel=np.asarray(vel, dtype=np.float64),
        quat_xyzw=np.asarray(q, dtype=np.float64),
        t=t,
    )


def _graph():
    g = FrameGraph()
    g.register_frame(NED)
    return g


def test_bounds_criterion_inside_returns_none():
    c = BoundsCriterion(lower=Point.of(-1, -1, 0.1, NED), upper=Point.of(1, 1, 2, NED))
    assert c.check(_state(pos=(0.5, 0.5, 1.0))) is None
    assert c.operates_in_frame == "ned"


def test_bounds_criterion_outside_returns_violation():
    c = BoundsCriterion(lower=Point.of(-1, -1, 0.1, NED), upper=Point.of(1, 1, 2, NED))
    v = c.check(_state(pos=(2.0, 0.0, 1.0)))
    assert v is not None and v.value > 0


def test_bounds_criterion_rejects_cross_frame_corners():
    from falsify.geometry import MOCAP
    with pytest.raises(ValueError, match="frames disagree"):
        BoundsCriterion(lower=Point.of(-1, -1, 0, NED), upper=Point.of(1, 1, 2, MOCAP))


def test_velocity_criterion():
    c = VelocityCriterion(max_speed=2.0)
    assert c.check(_state(vel=(1, 1, 0))) is None
    v = c.check(_state(vel=(3, 0, 0)))
    assert v is not None
    assert "exceeds" in v.description


def test_tilt_criterion_upright_is_safe():
    c = TiltCriterion(max_tilt_rad=0.5)
    assert c.check(_state(q=(0, 0, 0, 1.0))) is None


def test_tilt_criterion_90deg_pitch_violates():
    from scipy.spatial.transform import Rotation as _R
    q = _R.from_euler("y", np.pi / 2).as_quat()
    c = TiltCriterion(max_tilt_rad=0.5)
    v = c.check(_state(q=q))
    assert v is not None


def test_detector_tracks_last_safe_state_then_fires():
    g = _graph()
    det = FailureDetector(
        [BoundsCriterion(lower=Point.of(-1, -1, 0, NED), upper=Point.of(1, 1, 2, NED))],
        g,
    )
    s0 = _state(pos=(0, 0, 1), t=0.0)
    s1 = _state(pos=(0.5, 0, 1), t=0.1)
    s_bad = _state(pos=(2.0, 0, 1), t=0.2)
    assert det.update(s0, 0) is None
    assert det.update(s1, 1) is None
    rec = det.update(s_bad, 2)
    assert rec is not None
    assert rec.failure_type is FailureType.OUT_OF_BOUNDS
    assert rec.last_safe_step == 1
    np.testing.assert_array_equal(rec.last_safe_state.pos.xyz, [0.5, 0, 1])
    assert rec.failure_state.pos.xyz[0] == 2.0


def test_detector_returns_cached_record_on_subsequent_updates():
    g = _graph()
    det = FailureDetector(
        [VelocityCriterion(max_speed=1.0)], g,
    )
    s_bad = _state(vel=(5, 0, 0))
    rec1 = det.update(s_bad, 0)
    rec2 = det.update(s_bad, 1)
    assert rec1 is rec2


def test_detector_fires_inside_rollout_and_attaches_to_trace():
    from falsify.policy import MockNoisy, MockNoisyConfig
    from falsify.sensors import SensorRig, StateSensor
    from falsify.sim import DroneState, Simulator, SimulatorConfig

    g = _graph()
    sim = Simulator(SimulatorConfig(hz=10, horizon_s=2.0, policy_hz=1), g)
    sim.reset(_state(pos=(0, 0, 1)))
    policy = MockNoisy(MockNoisyConfig(
        goal=Point.of(0.5, 0, 1, NED), speed=1.0,
        horizon_s=2.0, n_waypoints=20, position_noise_std=0.5, seed=0,
    ))
    rig = SensorRig([StateSensor()])
    # Tight bounds — noisy mock will breach them.
    det = FailureDetector(
        [BoundsCriterion(
            lower=Point.of(-0.4, -0.4, 0.0, NED), upper=Point.of(0.4, 0.4, 2.0, NED),
        )],
        g,
    )
    trace = sim.rollout_with_policy(policy, rig, detector=det)
    assert trace.failure is not None
    assert trace.failure.last_safe_state.pos.frame.name == "ned"
