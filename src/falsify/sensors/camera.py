"""`CameraSensor` — one named camera mounted on the drone.

The sensor holds:
- a name (``forward``, ``downward``, …) that determines its observation keys
  (``images.<name>`` and optionally ``depth.<name>``);
- a `Frame` (the camera's optical frame, declared in the scene YAML);
- a static body→camera `SE3` extrinsic;
- a renderer callable that turns a *world-frame* `Pose` plus intrinsics into
  (RGB, depth) arrays. This callable abstracts over FiGS so tests/mocks can
  swap in a synthetic renderer.

The runtime body→world composition uses the drone state. The optional
``frame_graph`` is only consulted to convert camera-frame intermediates
into world-aligned frames when needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from falsify.geometry import Frame, FrameGraph, Pose, SE3
from falsify.policy.observation import ObservationBuilder
from falsify.sim.dynamics_state import DroneState
from .base import Sensor


# Renderer signature: takes a camera-to-world Pose (in some world frame
# acceptable to the renderer) plus an intrinsics dict, returns (rgb_uint8,
# depth_float32). Both arrays are H x W shaped.
RendererFn = Callable[[Pose, dict], tuple[np.ndarray, Optional[np.ndarray]]]


@dataclass(frozen=True)
class CameraSpec:
    """Static camera definition from the drone-frame YAML."""
    name: str
    frame: Frame
    intrinsics: dict
    body_from_camera: SE3   # SE3 with src=camera_frame, dst=cam_body
    model: str = "pinhole"


class CameraSensor(Sensor):
    """Render one camera every step.

    Parameters
    ----------
    spec
        Static camera definition.
    renderer
        Callable that performs the actual rendering. The orchestrator wires
        this with the simulator's GSPlat renderer; tests can pass a stub.
    body_to_world
        Callable mapping `DroneState` → camera-to-world `Pose` in the
        renderer's expected world frame. This is the runtime hinge that
        composes the drone's body→world pose with the static body→camera
        extrinsic.
    emit_depth
        When True, also writes ``depth.<name>``.
    """

    def __init__(
        self,
        spec: CameraSpec,
        renderer: RendererFn,
        body_to_world: Callable[[DroneState, SE3], Pose],
        emit_depth: bool = False,
    ) -> None:
        self.spec = spec
        self._render = renderer
        self._body_to_world = body_to_world
        self._emit_depth = bool(emit_depth)
        self._rgb_key = f"images.{spec.name}"
        self._depth_key = f"depth.{spec.name}"

    @property
    def keys_provided(self) -> frozenset[str]:
        if self._emit_depth:
            return frozenset({self._rgb_key, self._depth_key})
        return frozenset({self._rgb_key})

    def sense(self, state: DroneState, builder: ObservationBuilder) -> None:
        cam_pose_world = self._body_to_world(state, self.spec.body_from_camera)
        rgb, depth = self._render(cam_pose_world, self.spec.intrinsics)
        builder.set(self._rgb_key, rgb)
        if self._emit_depth:
            builder.set(self._depth_key, depth)


def make_camera_sensor_from_yaml(
    name: str,
    cam_yaml: dict,
    frame_graph: FrameGraph,
    renderer: RendererFn,
    body_to_world: Callable[[DroneState, SE3], Pose],
    *,
    emit_depth: bool = False,
) -> CameraSensor:
    """Build a `CameraSensor` from a drone-frame YAML entry.

    Expected ``cam_yaml`` schema (see ``configs/frames/carl_dual.yaml``)::

        frame: cam_forward          # already declared in scene's FrameGraph
        model: pinhole
        intrinsics: { width, height, fx, fy, cx, cy }

    The static body→camera SE3 is read from the FrameGraph (composed by
    `FrameGraph.transform("cam_body", cam_yaml["frame"])`), so adding or
    moving cameras is a YAML edit — no Python required.
    """
    cam_frame = frame_graph.frame(cam_yaml["frame"])
    body_frame = frame_graph.frame("cam_body")
    se3 = frame_graph.transform("cam_body", cam_yaml["frame"])
    if not isinstance(se3, SE3):
        raise TypeError(
            f"body→camera path produced a {type(se3).__name__}; expected SE3 "
            f"(camera extrinsics must be rigid)"
        )
    spec = CameraSpec(
        name=name,
        frame=cam_frame,
        intrinsics=dict(cam_yaml["intrinsics"]),
        body_from_camera=se3.inv(),  # we store body_from_camera by convention
        model=cam_yaml.get("model", "pinhole"),
    )
    return CameraSensor(spec, renderer=renderer, body_to_world=body_to_world, emit_depth=emit_depth)
