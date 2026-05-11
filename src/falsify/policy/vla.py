"""VLA policy — OpenPI websocket client.

Connects to a remote OpenPI VLA server, sends an `Observation` (dual-camera
images + state in MOCAP), receives a chunk of position-delta actions, and
integrates them into a NED `Trajectory` that the simulator follows.

Frame contract
--------------
- Input: `Observation.state.pos` in ``"ned"``; images keyed by camera name.
- The VLA was trained in MOCAP-Z-up, so positions are converted NED → MOCAP
  via the active `FrameGraph` before query, and the returned MOCAP-frame
  waypoints are converted back to NED on exit. The conversion happens
  exactly once per chunk, inside this module — no frame leaks.

Lazy imports
------------
``openpi_client`` is imported on first connection so the rest of falsify
loads cleanly on machines without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from falsify.geometry import (
    FrameGraph, Point, Trajectory, assert_frame, frame_by_name,
)
from .base import Policy
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

    # Frame in which the VLA was trained. Server outputs are converted from
    # this frame into NED on the way out.
    server_frame: str = "mocap"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resize_image(rgb: np.ndarray, size: int) -> np.ndarray:
    """Square-resize an HxWx3 uint8 image. Uses PIL to avoid an opencv dep."""
    from PIL import Image as _Image
    if rgb.shape[0] == size and rgb.shape[1] == size:
        return rgb
    img = _Image.fromarray(rgb)
    return np.asarray(img.resize((size, size), _Image.BILINEAR))


def _quat_to_yaw_xyzw(q: np.ndarray) -> float:
    qx, qy, qz, qw = q
    return float(np.arctan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    ))


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


class VLAPolicy(Policy):
    """OpenPI-backed VLA policy.

    Declares the camera modalities it needs; the orchestrator's sensor
    factory wires matching `CameraSensor`s. The policy itself does the
    frame conversions (NED → server frame → NED) and OpenPI handshake.
    """

    def __init__(self, cfg: VLAPolicyConfig, frame_graph: FrameGraph) -> None:
        self.cfg = cfg
        self.frame_graph = frame_graph
        self._client = None
        self._connected = False

    @property
    def required_modalities(self) -> frozenset[str]:
        return frozenset({
            f"images.{self.cfg.forward_camera}",
            f"images.{self.cfg.downward_camera}",
        })

    def reset(self) -> None:
        # Nothing per-episode; connection is reused across episodes.
        return None

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
        # 1. Convert state to server frame.
        pos_state = obs.state.pos
        assert_frame(pos_state, "ned")
        pos_server = self.frame_graph.convert(pos_state, to=self.cfg.server_frame).xyz
        yaw_server = _quat_to_yaw_xyzw(obs.state.quat_xyzw)

        # 2. Pack the OpenPI observation dict.
        rgb_forward = _resize_image(
            obs.require(f"images.{self.cfg.forward_camera}"), self.cfg.image_size,
        )
        rgb_downward = _resize_image(
            obs.require(f"images.{self.cfg.downward_camera}"), self.cfg.image_size,
        )
        prompt = obs.prompt or self.cfg.prompt
        payload = {
            "observation/image_forward": rgb_forward,
            "observation/image_downward": rgb_downward,
            "observation/state": np.concatenate([pos_server, [yaw_server]]).astype(np.float32),
            "prompt": prompt,
        }

        # 3. Query the server.
        self._ensure_connected()
        result = self._client.infer(payload)
        actions = np.asarray(result["actions"], dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] < 3:
            raise ValueError(
                f"VLA server returned actions shape {actions.shape}; expected (N, ≥3)"
            )

        # 4. Integrate position deltas in the server frame.
        n = min(actions.shape[0], self.cfg.actions_per_chunk)
        deltas = actions[:n, :3]
        positions_server = np.cumsum(deltas, axis=0) + pos_server[None, :]
        # Prepend the current position so the chunk starts where the drone is.
        positions_server = np.concatenate([pos_server[None, :], positions_server], axis=0)
        times = np.arange(positions_server.shape[0]) / self.cfg.hz + obs.state.t

        server_frame = self.frame_graph.frame(self.cfg.server_frame)
        traj_server = Trajectory(
            times=times,
            positions=positions_server,
            frame=server_frame,
        )
        # 5. Convert back to NED.
        traj_ned = self.frame_graph.convert(traj_server, to="ned")
        assert_frame(traj_ned, "ned")
        return traj_ned
