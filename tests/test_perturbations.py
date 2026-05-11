"""Perturbation framework tests — frame preservation across all three surfaces."""

from __future__ import annotations

import numpy as np
import pytest

from falsify.geometry import NED, MOCAP, FrameGraph, Point, Trajectory, SE3
from falsify.perturbations import (
    PerturbationSuite,
    PositionNoise, PositionBias, VelocityScale,
    ImageGaussianNoise, ImageBlur, StateNoise,
    StubEnvironmentPerturbation,
)
from falsify.policy.observation import Observation
from falsify.sim.dynamics_state import DroneState


def _state():
    return DroneState(
        pos=Point.of(0, 0, 1, NED),
        vel=np.zeros(3),
        quat_xyzw=np.array([0, 0, 0, 1.0]),
        t=0.0,
    )


def _traj(frame=NED, n=10):
    return Trajectory(
        times=np.linspace(0, 1, n),
        positions=np.linspace([0, 0, 0], [1, 0, 0], n),
        frame=frame,
        velocities=np.tile([1.0, 0.0, 0.0], (n, 1)),
    )


# ---------------------------------------------------------------------------
# Action perturbations
# ---------------------------------------------------------------------------


def test_position_noise_preserves_frame_and_shape():
    suite = PerturbationSuite(action=[PositionNoise(std=0.1)], seed=0)
    suite.reset()
    out = suite.apply_action(_traj())
    assert out.frame is NED
    assert out.positions.shape == _traj().positions.shape


def test_position_bias_shifts_all_waypoints():
    suite = PerturbationSuite(action=[PositionBias(bias_xyz=(1.0, 0.0, 0.0))], seed=0)
    suite.reset()
    base = _traj()
    out = suite.apply_action(base)
    diff = out.positions - base.positions
    assert np.allclose(diff, np.array([1.0, 0.0, 0.0])[None, :])
    assert out.frame is NED


def test_velocity_scale_passes_positions_through():
    suite = PerturbationSuite(action=[VelocityScale(scale=0.5)], seed=0)
    suite.reset()
    base = _traj()
    out = suite.apply_action(base)
    np.testing.assert_allclose(out.positions, base.positions)
    np.testing.assert_allclose(out.velocities, base.velocities * 0.5)


# ---------------------------------------------------------------------------
# Observation perturbations
# ---------------------------------------------------------------------------


def _obs_with_image(state, key="images.forward"):
    img = np.full((8, 8, 3), 100, dtype=np.uint8)
    return Observation(state=state, data={"state.pos": state.pos, key: img})


def test_image_noise_changes_image_in_place_only_for_named_camera():
    suite = PerturbationSuite(observation=[ImageGaussianNoise(camera="forward", std=10.0)], seed=42)
    suite.reset()
    obs = _obs_with_image(_state(), key="images.forward")
    out = suite.apply_observation(obs)
    # Frame on state.pos preserved.
    assert out.require("state.pos").frame is NED
    # Image changed; state image still present and the right shape.
    new_img = out.require("images.forward")
    assert new_img.shape == (8, 8, 3)
    assert not np.array_equal(new_img, obs.require("images.forward"))


def test_image_noise_is_noop_when_camera_missing():
    suite = PerturbationSuite(observation=[ImageGaussianNoise(camera="rear")], seed=0)
    suite.reset()
    obs = _obs_with_image(_state())
    out = suite.apply_observation(obs)
    assert out is obs or out.data == obs.data


def test_image_blur_preserves_frame_and_dtype():
    suite = PerturbationSuite(observation=[ImageBlur(camera="forward", kernel=3)], seed=0)
    suite.reset()
    obs = _obs_with_image(_state())
    out = suite.apply_observation(obs)
    new_img = out.require("images.forward")
    assert new_img.shape == (8, 8, 3)
    assert new_img.dtype == np.uint8


def test_state_noise_preserves_frame():
    suite = PerturbationSuite(observation=[StateNoise(std=0.1)], seed=1)
    suite.reset()
    obs = Observation(state=_state(), data={"state.pos": _state().pos})
    out = suite.apply_observation(obs)
    new_pos = out.require("state.pos")
    assert new_pos.frame is NED
    # Position was actually perturbed.
    assert not np.array_equal(new_pos.xyz, _state().pos.xyz)


# ---------------------------------------------------------------------------
# Manifest & reproducibility
# ---------------------------------------------------------------------------


def test_suite_is_reproducible_with_same_seed():
    cfg = lambda: PerturbationSuite(action=[PositionNoise(std=0.1)], seed=7)
    s1, s2 = cfg(), cfg()
    s1.reset(); s2.reset()
    out1 = s1.apply_action(_traj())
    out2 = s2.apply_action(_traj())
    np.testing.assert_allclose(out1.positions, out2.positions)


def test_manifest_contains_every_perturbation():
    suite = PerturbationSuite(
        observation=[StateNoise(std=0.02)],
        action=[PositionBias(bias_xyz=(0.1, 0, 0))],
        environment=[StubEnvironmentPerturbation(description="future opacity edit")],
        seed=123,
    )
    m = suite.manifest()
    assert m["seed"] == 123
    assert len(m["observation"]) == 1
    assert len(m["action"]) == 1
    assert len(m["environment"]) == 1
    assert m["observation"][0]["type"] == "StateNoise"
    assert m["action"][0]["bias_xyz"] == [0.1, 0, 0]


def test_environment_stub_raises_loudly():
    p = StubEnvironmentPerturbation(description="placeholder")
    with pytest.raises(NotImplementedError, match="placeholder"):
        p.apply(gsplat=None)


# ---------------------------------------------------------------------------
# Integration with the simulator
# ---------------------------------------------------------------------------


def test_perturbations_apply_inside_rollout_and_preserve_frames():
    from falsify.policy import MockStraightLine, MockStraightLineConfig
    from falsify.sensors import SensorRig, StateSensor
    from falsify.sim import Simulator, SimulatorConfig

    g = FrameGraph()
    g.register_frame(NED)
    sim = Simulator(SimulatorConfig(hz=10, horizon_s=1.0, policy_hz=1), g)
    sim.reset(_state())

    pol = MockStraightLine(MockStraightLineConfig(
        goal=Point.of(2.0, 0.0, 1.0, NED), n_waypoints=20,
    ))
    rig = SensorRig([StateSensor()])

    suite = PerturbationSuite(
        observation=[StateNoise(std=0.01)],
        action=[PositionBias(bias_xyz=(0.0, 0.1, 0.0))],
        seed=0,
    )
    trace = sim.rollout_with_policy(pol, rig, perturbations=suite)
    assert all(s.pos.frame is NED for s in trace.states)
    assert all(t.frame is NED for t in trace.policy_outputs)
    # PositionBias added +0.1 y to every waypoint; trace y should reflect that
    # (waypoints sit at y=0.1, sim follows them, so trace max y ≈ 0.1).
    assert max(s.pos.xyz[1] for s in trace.states) >= 0.1 - 1e-9
