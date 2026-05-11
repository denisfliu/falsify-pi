"""`Frame` value object plus a small registry of canonical defaults.

Frames are *data*, not enum members. Adding a new frame is a one-line config
edit — never a library edit. The canonical defaults below are useful for
typical drone-in-gsplat scenes but are not privileged in any way; library code
must look up frames by name through a `FrameGraph`, not by importing constants
from here.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import conventions as conv


@dataclass(frozen=True)
class Frame:
    """A named coordinate frame.

    ``name`` is the canonical identifier used everywhere (configs, type tags,
    `FrameGraph` lookups). ``convention`` and ``notes`` are informational and
    surface in `FrameGraph.describe()` for debugging.
    """

    name: str
    convention: str = conv.RIGHT_HANDED
    notes: str = ""

    def __hash__(self) -> int:  # frozen + str-based identity
        return hash(self.name)

    def __str__(self) -> str:
        return self.name


# ---------------------------------------------------------------------------
# Canonical defaults — convenient constants, but not privileged.
# Scenes are free to declare additional frames in their YAML.
# ---------------------------------------------------------------------------

NED = Frame("ned", notes=f"FiGS dynamics; {conv.NED_AXES}")
MOCAP = Frame("mocap", notes=f"Motion-capture world; {conv.ZUP_AXES}")
COLMAP = Frame("colmap", notes="Raw SfM frame")
NS = Frame("ns", notes="Nerfstudio-internal (dataparser-transformed, scaled)")
CAM_BODY = Frame("cam_body", notes="Body frame attached to the drone IMU origin")
CAM_FORWARD = Frame(
    "cam_forward",
    notes=f"Forward-facing camera optical frame; {conv.OPENCV_CAMERA_AXES}",
)
CAM_DOWNWARD = Frame(
    "cam_downward",
    notes=f"Downward-facing camera optical frame; {conv.OPENCV_CAMERA_AXES}",
)


CANONICAL_FRAMES: tuple[Frame, ...] = (
    NED, MOCAP, COLMAP, NS, CAM_BODY, CAM_FORWARD, CAM_DOWNWARD,
)


def frame_by_name(name: str, *extras: Frame) -> Frame:
    """Look up a `Frame` by name from the canonical defaults, falling back to
    ``extras`` (typically the frames declared in the active scene YAML)."""
    for f in (*CANONICAL_FRAMES, *extras):
        if f.name == name:
            return f
    raise KeyError(f"unknown frame: {name!r}")
