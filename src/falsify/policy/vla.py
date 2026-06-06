"""VLA policy — OpenPI websocket client.

Connects to a remote OpenPI VLA server, sends an `Observation` (dual-camera
images + state in MOCAP), receives a chunk of position-delta actions, and
integrates them into a NED `Trajectory` that the simulator follows.

Frame contract
--------------
- Input: ``Observation.state.pos`` in ``"ned"``; images keyed by camera name.
- The VLA was trained in MOCAP-Z-up (FiGS perm5: ``R_mocap_from_ned = diag(1,-1,-1)``).
  Positions are converted NED → MOCAP via the active `FrameGraph` before the
  query; the integrated MOCAP-frame waypoints are converted back to NED on
  exit. The conversion happens exactly once per chunk, inside this module —
  no frame leaks.
- The yaw component of the state vector is **negated** (``-yaw_ned``) because
  NED z-down and MOCAP z-up induce opposite-sign yaw conventions.

OpenPI payload (matches SousVide / the moraband server's training distribution):

  observation/image          — forward camera, uint8 (256, 256, 3)
  observation/wrist_image    — downward camera, uint8 (256, 256, 3)
  observation/3pov_1         — static third-person, uint8 (256, 256, 3) — zeros by default
  observation/state          — float32 shape (7,) = [px, py, pz, -yaw, 0, 0, 0]
  prompt                     — str

Action chunk
------------
The server returns ``actions`` of shape ``(N, ≥3)`` — position deltas in
MOCAP per timestep, optionally with a yaw delta at index 3. We integrate
them cumulatively starting from the current MOCAP position/yaw, prepend the
current pose (so the chunk starts where the drone is), assign timestamps at
``hz``, and convert back to NED. The simulator follows the resulting
`Trajectory["ned"]` waypoint by waypoint until exhausted, then re-queries.

Debug recording
---------------
If ``VLAPolicyConfig.record_dir`` is set, each query saves a directory
``query_<NNNN>_step_<KKKKK>/`` containing:

  rgb_fwd.png, rgb_dwn.png         — native-resolution renders
  obs_front.png, obs_down.png      — the post-resize images sent to the VLA
  obs_3pov.png                     — the third-person channel (zeros by default)
  data.txt                         — state_ned, state_to_vla, prompt
  actions.npy                      — raw action array (N, M)
  waypoints_ned.npy                — integrated NED waypoints

Lazy imports
------------
``openpi_client`` is imported on first connection so the rest of falsify
loads cleanly on machines without it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from falsify.geometry import (
    FrameGraph, Point, Trajectory, assert_frame,
)
from .base import Policy
from .camera_postprocess import CameraPostprocess
from .observation import Observation


# ---------------------------------------------------------------------------
# Host alias registry
# ---------------------------------------------------------------------------


_POLICY_HOSTS: dict[str, str] = {
    "moraband":  "moraband.stanford.edu",
    "manaan":    "SOE-50TJK74.stanford.edu",
    "coruscant": "coruscant.stanford.edu",
    "endor":     "localhost",
}


def register_policy_host(alias: str, fqdn: str) -> None:
    _POLICY_HOSTS[alias] = fqdn


def _resolve_host(name: str) -> str:
    return _POLICY_HOSTS.get(name, name)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class VLAPolicyConfig:
    host: str = "moraband"
    port: int = 8000
    prompt: str = ""

    # Control / chunking
    hz: int = 10
    actions_per_chunk: int = 50

    # Image preprocessing
    image_size: int = 256
    forward_camera: str = "forward"
    downward_camera: str = "downward"

    # Optional per-camera RGBA overlay paths (e.g. wrist gripper).
    # Composited as the last step of preprocess so train/eval see
    # identical bytes. Keys are camera names ("forward", "downward");
    # missing entries → no overlay for that camera.
    gripper_overlay_paths: dict[str, str] = field(default_factory=dict)

    # Frame in which the VLA was trained. Server outputs are converted from
    # this frame into NED on the way out.
    server_frame: str = "mocap"

    # Optional static third-person image. None → zero-filled 3pov channel.
    third_person_image_path: Optional[str] = None

    # Debug recording.
    record_dir: Optional[Path] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resize_image(rgb: np.ndarray, size: int) -> np.ndarray:
    """Square-resize an HxWx3 uint8 image (PIL bilinear). No-op if already sq."""
    from PIL import Image as _Image
    if rgb.shape[0] == size and rgb.shape[1] == size:
        return rgb
    img = _Image.fromarray(rgb)
    return np.asarray(img.resize((size, size), _Image.BILINEAR))


# Re-exports from `falsify.geometry.yaw` — kept under their legacy
# underscore-prefixed names because external code (e.g.
# `training.exporter`) historically imported them from here.
_quat_to_yaw_xyzw = None  # populated just below
_yaw_to_quat_xyzw = None
from falsify.geometry import quat_to_yaw_xyzw as _quat_to_yaw_xyzw  # noqa: E402, F811
from falsify.geometry import yaw_to_quat_xyzw as _yaw_to_quat_xyzw  # noqa: E402, F811


def _load_third_person(path: Optional[str], size: int) -> np.ndarray:
    if path is None:
        return np.zeros((size, size, 3), dtype=np.uint8)
    from PIL import Image as _Image
    img = np.asarray(_Image.open(path).convert("RGB"))
    return _resize_image(img, size)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


class VLAPolicy(Policy):
    """OpenPI-backed VLA policy.

    Declares the camera modalities it needs; the orchestrator's sensor
    factory wires matching `CameraSensor`s. The policy does the frame
    conversions (NED → MOCAP → NED) and the OpenPI handshake.
    """

    def __init__(self, cfg: VLAPolicyConfig, frame_graph: FrameGraph) -> None:
        self.cfg = cfg
        self.frame_graph = frame_graph
        self._client = None
        self._connected = False
        self._query_count = 0
        self._step_count = 0
        self._third_person_cache: Optional[np.ndarray] = None
        # Shared with `PiGatewayPolicy` and `TrainingDataExporter`: same
        # resize → channel swap → overlay pipeline so a policy YAML and
        # an embodiment YAML pointing at the same overlay produce
        # byte-identical preprocess.
        self._postprocess: dict[str, CameraPostprocess] = {
            cam_name: CameraPostprocess.from_paths(
                image_size=cfg.image_size,
                channel_order="RGB",   # OpenPI server expects RGB pixels
                overlay_path=cfg.gripper_overlay_paths.get(cam_name),
            )
            for cam_name in (cfg.forward_camera, cfg.downward_camera)
        }

    @property
    def required_modalities(self) -> frozenset[str]:
        return frozenset({
            f"images.{self.cfg.forward_camera}",
            f"images.{self.cfg.downward_camera}",
        })

    def reset(self) -> None:
        # Connection persists across episodes; per-episode counters reset.
        self._query_count = 0
        self._step_count = 0

    # ---- OpenPI handshake ---------------------------------------------

    def _ensure_connected(self) -> None:
        if self._connected:
            return
        from openpi_client import websocket_client_policy  # type: ignore
        host = _resolve_host(self.cfg.host)
        self._client = websocket_client_policy.WebsocketClientPolicy(
            host=host, port=self.cfg.port,
        )
        self._connected = True

    def close(self) -> None:
        if self._client is not None and hasattr(self._client, "_ws"):
            try:
                self._client._ws.close()
            except Exception:
                pass
        self._connected = False

    # ---- main loop ----------------------------------------------------

    def observe(self, obs: Observation) -> Trajectory:
        # 1. Convert state to server frame (MOCAP).
        pos_state = obs.state.pos
        assert_frame(pos_state, "ned")
        pos_mocap = self.frame_graph.convert(pos_state, to=self.cfg.server_frame).xyz
        # NED z-down vs. MOCAP z-up flips the yaw sign convention.
        yaw_ned = _quat_to_yaw_xyzw(obs.state.quat_xyzw)
        yaw_to_vla = -yaw_ned

        # 2. Build the OpenPI observation dict.
        sz = self.cfg.image_size
        rgb_fwd_native = obs.require(f"images.{self.cfg.forward_camera}")
        rgb_dwn_native = obs.require(f"images.{self.cfg.downward_camera}")
        rgb_fwd = self._postprocess[self.cfg.forward_camera].apply(rgb_fwd_native)
        rgb_dwn = self._postprocess[self.cfg.downward_camera].apply(rgb_dwn_native)
        if self._third_person_cache is None:
            self._third_person_cache = _load_third_person(
                self.cfg.third_person_image_path, sz,
            )
        rgb_3pov = self._third_person_cache

        state_vec = np.zeros(7, dtype=np.float32)
        state_vec[:3] = pos_mocap
        state_vec[3] = yaw_to_vla

        prompt = obs.prompt or self.cfg.prompt

        payload = {
            "observation/image":       rgb_fwd,
            "observation/wrist_image": rgb_dwn,
            "observation/3pov_1":      rgb_3pov,
            "observation/state":       state_vec,
            "prompt":                  prompt,
        }

        # 3. Query the server.
        self._ensure_connected()
        t0 = time.time()
        result = self._client.infer(payload)
        infer_s = time.time() - t0
        actions = np.asarray(result["actions"], dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] < 3:
            raise ValueError(
                f"VLA server returned actions shape {actions.shape}; expected (N, ≥3)"
            )

        # 4. Integrate position + yaw deltas in MOCAP.
        n = min(actions.shape[0], self.cfg.actions_per_chunk)
        deltas = actions[:n, :3]
        positions_mocap = np.cumsum(deltas, axis=0) + pos_mocap[None, :]
        positions_mocap = np.concatenate(
            [pos_mocap[None, :], positions_mocap], axis=0,
        )  # length n+1, starts at current pose

        # Yaw integration: deltas in MOCAP-yaw add to the current MOCAP-yaw.
        # We carry NED yaw through (sign flip at the boundary), so subtract
        # the MOCAP delta from NED yaw (mirror of the state-vec sign flip).
        yaws_ned = np.zeros(n + 1)
        yaws_ned[0] = yaw_ned
        if actions.shape[1] >= 4:
            yaw_deltas_mocap = actions[:n, 3]
            for i, dy in enumerate(yaw_deltas_mocap):
                yaws_ned[i + 1] = yaws_ned[i] - float(dy)
        else:
            yaws_ned[:] = yaw_ned
        quats_xyzw = np.stack([_yaw_to_quat_xyzw(y) for y in yaws_ned], axis=0)

        times = np.arange(positions_mocap.shape[0]) / self.cfg.hz + obs.state.t

        # 5. Convert positions back to NED. Quaternions are integrated and
        #    attached in the NED frame directly — going through
        #    `frame_graph.convert` would apply perm5 (a 180° flip about x)
        #    to each quaternion, which is the wrong transformation for a
        #    body orientation (NED yaw and MOCAP yaw differ by a sign, not
        #    a 180° tilt). Attaching NED-frame quats to a NED-frame
        #    trajectory keeps the orientation consistent with NED yaw
        #    convention everywhere downstream.
        server_frame = self.frame_graph.frame(self.cfg.server_frame)
        traj_mocap_pos = Trajectory(
            times=times,
            positions=positions_mocap,
            frame=server_frame,
        )
        traj_ned_pos = self.frame_graph.convert(traj_mocap_pos, to="ned")
        ned_frame = self.frame_graph.frame("ned")
        traj_ned = Trajectory(
            times=times,
            positions=traj_ned_pos.positions,
            frame=ned_frame,
            quaternions=quats_xyzw,
        )
        assert_frame(traj_ned, "ned")

        # 6. Debug recording.
        if self.cfg.record_dir is not None:
            self._record_query(
                pos_state.xyz, yaw_ned, pos_mocap, state_vec, prompt,
                rgb_fwd_native, rgb_dwn_native, rgb_fwd, rgb_dwn, rgb_3pov,
                actions, traj_ned, infer_s,
            )
        self._query_count += 1
        self._step_count += n
        return traj_ned

    # ---- debug recording ----------------------------------------------

    def _record_query(
        self,
        pos_ned_xyz: np.ndarray,
        yaw_ned: float,
        pos_mocap: np.ndarray,
        state_vec_to_vla: np.ndarray,
        prompt: str,
        rgb_fwd_native: np.ndarray,
        rgb_dwn_native: np.ndarray,
        rgb_fwd_resized: np.ndarray,
        rgb_dwn_resized: np.ndarray,
        rgb_3pov: np.ndarray,
        actions: np.ndarray,
        traj_ned: Trajectory,
        infer_seconds: float,
    ) -> None:
        from PIL import Image as _Image
        root = Path(self.cfg.record_dir)
        qdir = root / f"query_{self._query_count:04d}_step_{self._step_count:05d}"
        qdir.mkdir(parents=True, exist_ok=True)

        _Image.fromarray(rgb_fwd_native).save(qdir / "rgb_fwd.png")
        _Image.fromarray(rgb_dwn_native).save(qdir / "rgb_dwn.png")
        _Image.fromarray(rgb_fwd_resized).save(qdir / "obs_front.png")
        _Image.fromarray(rgb_dwn_resized).save(qdir / "obs_down.png")
        _Image.fromarray(rgb_3pov).save(qdir / "obs_3pov.png")

        np.save(qdir / "actions.npy", actions)
        np.save(qdir / "waypoints_ned.npy", traj_ned.positions)

        with (qdir / "data.txt").open("w") as f:
            f.write(f"query_index: {self._query_count}\n")
            f.write(f"step_count_so_far: {self._step_count}\n")
            f.write(f"infer_seconds: {infer_seconds:.4f}\n")
            f.write(f"prompt: {prompt!r}\n")
            f.write(f"state_ned_pos: {pos_ned_xyz.tolist()}\n")
            f.write(f"state_ned_yaw_rad: {yaw_ned}\n")
            f.write(f"pos_mocap: {pos_mocap.tolist()}\n")
            f.write(f"state_vec_to_vla: {state_vec_to_vla.tolist()}\n")
            f.write(f"actions_shape: {actions.shape}\n")
