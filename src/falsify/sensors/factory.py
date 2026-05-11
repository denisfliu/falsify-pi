"""Sensor-rig factory.

Given a policy's `required_modalities` plus the active scene + drone-frame
configs, assemble a `SensorRig` that covers exactly what the policy needs —
no more, no less.

Camera sensors require a renderer callable. If the policy needs cameras and
no renderer is supplied, construction fails fast with a clear message.

This is the single wiring point that translates dotted-modality keys into
sensor instances. Adding a new modality namespace = a new branch here.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from falsify.geometry import FrameGraph, Pose
from falsify.policy.observation import ObservationBuilder
from falsify.sim.dynamics_state import DroneState
from .base import Sensor, SensorRig
from .camera import CameraSensor, make_camera_sensor_from_yaml
from .prompt import PromptSensor
from .state import StateSensor


# A renderer is anything callable with the camera's `(Pose, intrinsics)` and
# returning ``(rgb, depth | None)``. The simulator's `GSplatRenderer.render`
# is one example; tests use stubs.
RendererFn = Callable[[Pose, dict], tuple]

# A body→world hinge: takes the runtime DroneState plus a static body-from-
# camera SE3 and returns the camera-to-world `Pose` the renderer expects.
BodyToWorldFn = Callable[[DroneState, "object"], Pose]


def build_sensor_rig(
    required_modalities: frozenset[str],
    *,
    frame_graph: FrameGraph,
    frame_cfg: dict,
    renderer: Optional[RendererFn] = None,
    body_to_world: Optional[BodyToWorldFn] = None,
    prompt: Optional[str] = None,
    extra_sensors: Sequence[Sensor] = (),
) -> SensorRig:
    """Construct the rig.

    Parameters
    ----------
    required_modalities
        ``policy.required_modalities`` — drives which non-default sensors are
        attached.
    frame_graph
        Active scene `FrameGraph` (used to resolve body→camera extrinsics).
    frame_cfg
        Parsed drone-frame YAML; expected to contain a ``cameras`` block
        keyed by camera name (see ``configs/frames/carl_dual.yaml``).
    renderer, body_to_world
        Required iff `required_modalities` includes any ``images.*`` or
        ``depth.*`` key. The renderer takes ``(Pose, intrinsics_dict)`` and
        returns ``(rgb, depth)``. `body_to_world` composes the runtime
        state with the static body-from-camera extrinsic.
    prompt
        If non-None, adds a `PromptSensor` that emits this string each step.
    extra_sensors
        Caller-supplied sensors to append (e.g. for tests or custom
        modalities).
    """
    sensors: list[Sensor] = [StateSensor()]
    if prompt is not None:
        sensors.append(PromptSensor(prompt))

    cameras_block = frame_cfg.get("cameras", {})
    requested_cameras = _cameras_referenced(required_modalities)

    if requested_cameras and renderer is None:
        raise ValueError(
            f"policy requires camera modalities {sorted(requested_cameras)} "
            f"but no renderer was supplied"
        )
    if requested_cameras and body_to_world is None:
        raise ValueError(
            f"policy requires camera modalities {sorted(requested_cameras)} "
            f"but no body_to_world hinge was supplied"
        )

    for cam_name in sorted(requested_cameras):
        if cam_name not in cameras_block:
            raise ValueError(
                f"policy requires camera {cam_name!r} but frame config has no "
                f"such entry (available: {sorted(cameras_block)})"
            )
        cam_yaml = cameras_block[cam_name]
        emit_depth = f"depth.{cam_name}" in required_modalities
        sensor = make_camera_sensor_from_yaml(
            cam_name,
            cam_yaml,
            frame_graph,
            renderer=renderer,
            body_to_world=body_to_world,
            emit_depth=emit_depth,
        )
        sensors.append(sensor)

    sensors.extend(extra_sensors)
    rig = SensorRig(sensors)
    rig.assert_covers(required_modalities)
    return rig


def _cameras_referenced(modalities: frozenset[str]) -> set[str]:
    out: set[str] = set()
    for m in modalities:
        for ns in ("images.", "depth."):
            if m.startswith(ns):
                out.add(m[len(ns):])
    return out
