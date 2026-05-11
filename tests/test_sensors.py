"""Sensor pipeline tests — single-writer invariant, coverage, no-image policies."""

from __future__ import annotations

import numpy as np
import pytest

from falsify.geometry import NED, Point, Pose, SE3, Frame
from falsify.policy.observation import Observation, ObservationBuilder
from falsify.sensors import (
    Sensor, SensorRig, StateSensor, PromptSensor, CameraSensor, CameraSpec,
)
from falsify.sim.dynamics_state import DroneState


def _state(t: float = 0.0) -> DroneState:
    return DroneState(
        pos=Point.of(0.0, 0.0, 0.0, NED),
        vel=np.zeros(3),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        t=t,
    )


def test_state_sensor_exposes_frame_tagged_position_only():
    rig = SensorRig([StateSensor()])
    obs = rig.build(_state(0.5))
    pos = obs.require("state.pos")
    assert pos.frame is NED   # frame-tagged Point, not a bare ndarray
    # vel / quat / t are accessed via obs.state — never as bare dict entries.
    assert obs.state.t == 0.5
    np.testing.assert_array_equal(obs.state.vel, np.zeros(3))
    np.testing.assert_array_equal(obs.state.quat_xyzw, [0, 0, 0, 1])
    # Frame is uniquely discoverable through the state dataclass.
    assert obs.state.frame is NED
    with pytest.raises(KeyError):
        obs.require("state.vel")   # explicitly NOT a dotted key — no bare leak


def test_rig_asserts_coverage_and_fails_fast_on_missing():
    rig = SensorRig([StateSensor()])
    rig.assert_covers({"state.pos"})  # ok
    with pytest.raises(ValueError, match="missing required modalities"):
        rig.assert_covers({"images.forward"})


def test_rig_rejects_overlapping_keys():
    class A(Sensor):
        KEYS = frozenset({"shared.key"})

        @property
        def keys_provided(self):
            return self.KEYS

        def sense(self, state, builder):
            builder.set("shared.key", 1)

    class B(Sensor):
        KEYS = frozenset({"shared.key"})

        @property
        def keys_provided(self):
            return self.KEYS

        def sense(self, state, builder):
            builder.set("shared.key", 2)

    with pytest.raises(ValueError, match="conflict"):
        SensorRig([A(), B()])


def test_rig_rejects_unexpected_writes():
    class Misbehaved(Sensor):
        @property
        def keys_provided(self):
            return frozenset({"declared.key"})

        def sense(self, state, builder):
            builder.set("declared.key", 1)
            builder.set("undeclared.key", 2)

    rig = SensorRig([Misbehaved()])
    with pytest.raises(RuntimeError, match="unexpected keys"):
        rig.build(_state())


def test_observation_require_raises_on_missing_key():
    obs = Observation(state=_state(), data={"a": 1})
    obs.require("a")
    with pytest.raises(KeyError, match="not present"):
        obs.require("missing")


def test_camera_sensor_writes_image_key():
    # A stub renderer returning a deterministic image so we don't need FiGS.
    def renderer(pose, intrinsics):
        H, W = intrinsics["height"], intrinsics["width"]
        return np.full((H, W, 3), 42, dtype=np.uint8), None

    def body_to_world(state, body_from_cam):
        # Construct an arbitrary world-frame Pose. The renderer stub ignores it.
        return Pose(R=np.eye(3), t=state.pos.xyz, frame=state.pos.frame)

    spec = CameraSpec(
        name="forward",
        frame=Frame("cam_forward"),
        intrinsics={"height": 8, "width": 16, "fx": 1, "fy": 1, "cx": 8, "cy": 4},
        body_from_camera=SE3.identity(Frame("cam_forward"), Frame("cam_body")),
    )
    sensor = CameraSensor(spec, renderer=renderer, body_to_world=body_to_world)
    rig = SensorRig([StateSensor(), sensor])
    rig.assert_covers({"images.forward", "state.pos"})
    obs = rig.build(_state())
    img = obs.require("images.forward")
    assert img.shape == (8, 16, 3)
    assert (img == 42).all()


def test_prompt_sensor_sets_prompt():
    rig = SensorRig([StateSensor(), PromptSensor("fly through the gate")])
    obs = rig.build(_state())
    assert obs.prompt == "fly through the gate"
