"""Runtime body↔world pose composition.

The static `FrameGraph` holds *fixed* extrinsics — body→camera, mocap↔colmap,
etc. The drone's instantaneous body→world pose is the runtime state, not a
static edge. This module owns the small piece of math that composes the
two: given a `DroneState` and a static ``body_from_camera`` SE3, produce a
camera-to-world `Pose` ready for the renderer.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as _R

from falsify.geometry import CAM_BODY, Frame, Pose, SE3
from .dynamics_state import DroneState


def body_to_world_se3(state: DroneState, *, body_frame: Frame = CAM_BODY) -> SE3:
    """Construct the SE3 mapping ``body_frame → state.pos.frame``.

    The rotation comes from `state.quat_xyzw`; the translation is `state.pos`.
    The result's ``src`` is the body frame and ``dst`` is the state's frame
    (usually ``"ned"``).
    """
    R = _R.from_quat(state.quat_xyzw).as_matrix()
    return SE3(R=R, t=state.pos.xyz, src=body_frame, dst=state.pos.frame)


def camera_to_world_pose(
    state: DroneState,
    body_from_camera: SE3,
    *,
    body_frame: Frame = CAM_BODY,
) -> Pose:
    """Compose body→world with the static body←camera extrinsic.

    Parameters
    ----------
    state
        Current drone state. Provides the runtime body→world rotation+translation.
    body_from_camera
        Static SE3 with ``src == <camera_frame>`` and ``dst == body_frame``.
        Conventionally stored on `CameraSpec.body_from_camera`.

    Returns
    -------
    A `Pose` whose frame is ``state.pos.frame`` (typically ``"ned"``). The
    rotation/translation describe the camera's pose in the world frame.
    """
    if body_from_camera.dst.name != body_frame.name:
        raise ValueError(
            f"body_from_camera.dst must equal body frame {body_frame.name!r}; "
            f"got {body_from_camera.dst.name!r}"
        )
    T_body_to_world = body_to_world_se3(state, body_frame=body_frame)
    T_cam_to_world = T_body_to_world @ body_from_camera
    return Pose(R=T_cam_to_world.R, t=T_cam_to_world.t, frame=state.pos.frame)
