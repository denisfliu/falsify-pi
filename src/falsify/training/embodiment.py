"""Embodiment specification — declarative training-data schema config.

An embodiment YAML decides:
- what features the parquet emits (image columns, state vector layout, etc.)
- which scene camera each image column reads from
- the channel order written into the PNG bytes (BGR for cv2-collected
  datasets, RGB for PIL-collected ones)
- how state and action vectors are assembled per frame

Adding a new embodiment is a YAML edit — code only needs to change when a
genuinely new modality is introduced (e.g., depth, lidar).

The exporter only consumes ``EmbodimentSpec``; the YAML loader is
``load_embodiment(path)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import yaml


@dataclass(frozen=True)
class CameraSpec:
    """One image column in the parquet."""
    column: str                       # parquet column name (e.g. "image", "wrist_image", "3pov_1")
    source: Literal["render", "zeros", "static"]  # how to populate this image
    camera_name: Optional[str] = None  # required when source == "render"
    static_path: Optional[str] = None  # path to a static PNG, when source == "static"
    channel_order: Literal["RGB", "BGR"] = "BGR"
    image_size: int = 256             # square output edge


@dataclass(frozen=True)
class StateField:
    """One slot in the state vector."""
    name: str   # logical name of the value, used by the exporter
    # `name` understands:
    #   x_mocap, y_mocap, z_mocap      — drone position components in MOCAP
    #   yaw_mocap                      — MOCAP yaw (= -yaw_ned)
    #   x_ned, y_ned, z_ned, yaw_ned   — NED equivalents
    #   gripper                        — gripper state (currently always 0.0)
    #   zero                           — constant 0.0 padding


@dataclass(frozen=True)
class ActionField:
    """One slot in the action vector — typically a delta of a state field."""
    name: str
    # `name` understands the delta of any state field:
    #   d_x_mocap, d_y_mocap, d_z_mocap, d_yaw_mocap, d_gripper, zero, ...


@dataclass(frozen=True)
class EmbodimentSpec:
    name: str
    fps: int
    image_size: int
    cameras: tuple[CameraSpec, ...]
    state: tuple[StateField, ...]
    actions: tuple[ActionField, ...]
    # Yaw delta wrapping behavior. "pi" wraps to [-π, π]; "none" leaves
    # raw subtraction (callers should ensure yaws stay continuous).
    yaw_wrap: Literal["pi", "none"] = "pi"
    # First-row action convention. "zeros" matches DroneVLA2.0's collector
    # (the first action is undefined since there is no previous state).
    first_action: Literal["zeros", "forward"] = "zeros"
    # robot_type metadata kept for HF / LeRobot compatibility, never used
    # by the exporter directly.
    robot_type: str = "panda"
    # Free-form description of the embodiment (informational).
    notes: str = ""

    def state_dim(self) -> int:
        return len(self.state)

    def action_dim(self) -> int:
        return len(self.actions)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_embodiment(path: str | Path) -> EmbodimentSpec:
    """Parse an embodiment YAML into an ``EmbodimentSpec``."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    cameras = tuple(
        CameraSpec(
            column=c["column"],
            source=c["source"],
            camera_name=c.get("camera_name"),
            static_path=c.get("static_path"),
            channel_order=c.get("channel_order", "BGR"),
            image_size=int(c.get("image_size", cfg.get("image_size", 256))),
        )
        for c in cfg["cameras"]
    )
    state = tuple(StateField(name=s["name"]) for s in cfg["state"])
    actions = tuple(ActionField(name=a["name"]) for a in cfg["actions"])
    return EmbodimentSpec(
        name=cfg["name"],
        fps=int(cfg["fps"]),
        image_size=int(cfg.get("image_size", 256)),
        cameras=cameras,
        state=state,
        actions=actions,
        yaw_wrap=cfg.get("yaw_wrap", "pi"),
        first_action=cfg.get("first_action", "zeros"),
        robot_type=cfg.get("robot_type", "panda"),
        notes=cfg.get("notes", ""),
    )
