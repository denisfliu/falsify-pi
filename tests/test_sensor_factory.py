"""Sensor-rig factory tests — modality → sensor wiring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from falsify.geometry import (
    CAM_BODY, CAM_FORWARD, NED, Frame, FrameGraph, SE3,
)
from falsify.io import build_frame_graph, load_yaml
from falsify.sensors import build_sensor_rig


def _scene_with_cameras() -> tuple[FrameGraph, dict]:
    g = FrameGraph()
    for f in (NED, CAM_BODY, CAM_FORWARD):
        g.register_frame(f)
    g.register_edge(SE3(R=np.eye(3), t=np.zeros(3), src=CAM_BODY, dst=CAM_FORWARD))

    frame_cfg = {
        "cameras": {
            "forward": {
                "frame": "cam_forward",
                "model": "pinhole",
                "intrinsics": {"width": 8, "height": 8, "fx": 1, "fy": 1, "cx": 4, "cy": 4},
            }
        }
    }
    return g, frame_cfg


def _stub_renderer(pose, intrinsics):
    H, W = intrinsics["height"], intrinsics["width"]
    return np.zeros((H, W, 3), dtype=np.uint8), None


def _stub_body_to_world(state, body_from_camera):
    from falsify.geometry import Pose
    return Pose(R=np.eye(3), t=state.pos.xyz, frame=state.pos.frame)


def test_no_camera_modalities_yields_state_only_rig():
    g, frame_cfg = _scene_with_cameras()
    rig = build_sensor_rig(
        frozenset(),
        frame_graph=g,
        frame_cfg=frame_cfg,
    )
    assert rig.keys_provided() == frozenset({"state.pos"})


def test_required_camera_wires_camera_sensor():
    g, frame_cfg = _scene_with_cameras()
    rig = build_sensor_rig(
        frozenset({"images.forward"}),
        frame_graph=g,
        frame_cfg=frame_cfg,
        renderer=_stub_renderer,
        body_to_world=_stub_body_to_world,
    )
    assert "images.forward" in rig.keys_provided()


def test_factory_fails_fast_without_renderer():
    g, frame_cfg = _scene_with_cameras()
    with pytest.raises(ValueError, match="no renderer"):
        build_sensor_rig(
            frozenset({"images.forward"}),
            frame_graph=g,
            frame_cfg=frame_cfg,
        )


def test_factory_rejects_unknown_camera_name():
    g, frame_cfg = _scene_with_cameras()
    with pytest.raises(ValueError, match="no such entry"):
        build_sensor_rig(
            frozenset({"images.rear_view"}),
            frame_graph=g,
            frame_cfg=frame_cfg,
            renderer=_stub_renderer,
            body_to_world=_stub_body_to_world,
        )


def test_factory_emits_depth_when_required():
    g, frame_cfg = _scene_with_cameras()
    rig = build_sensor_rig(
        frozenset({"images.forward", "depth.forward"}),
        frame_graph=g,
        frame_cfg=frame_cfg,
        renderer=_stub_renderer,
        body_to_world=_stub_body_to_world,
    )
    assert {"images.forward", "depth.forward"}.issubset(rig.keys_provided())


def test_prompt_passes_through():
    g, frame_cfg = _scene_with_cameras()
    rig = build_sensor_rig(
        frozenset(),
        frame_graph=g,
        frame_cfg=frame_cfg,
        prompt="fly through the gate",
    )
    from falsify.geometry import Point
    from falsify.sim.dynamics_state import DroneState
    state = DroneState(pos=Point.of(0, 0, 0, NED), vel=np.zeros(3),
                       quat_xyzw=np.array([0, 0, 0, 1.0]), t=0.0)
    obs = rig.build(state)
    assert obs.prompt == "fly through the gate"
