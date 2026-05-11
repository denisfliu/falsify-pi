"""Frame-tagged geometry layer.

See ``CLAUDE.md`` in this directory for the frame contract and the recipe
for adding new frames or transform types.

Public API:
- `Frame` value object + canonical defaults (NED, MOCAP, COLMAP, NS, CAM_BODY, …)
- `Point`, `Pose`, `Trajectory`, `PointCloud` — frame-tagged geometry types
- `assert_frame(value, expected)` — call-site guard
- `SE3`, `Sim3` — frame-aware rigid / similarity transforms with `@` overload
- `FrameGraph` — runtime registry + BFS-composing convert(value, to)
- `register_loader(type_name, fn)` — add new YAML transform types
"""

from .frames import (
    Frame,
    NED, MOCAP, COLMAP, NS, CAM_BODY, CAM_FORWARD, CAM_DOWNWARD,
    CANONICAL_FRAMES, frame_by_name,
)
from .types import Point, Pose, Trajectory, PointCloud, assert_frame
from .transforms import SE3, Sim3
from .frame_graph import FrameGraph
from .loaders import register_loader, get_loader, available_loaders
from .presets import axis_permutation, available_presets

__all__ = [
    "Frame",
    "NED", "MOCAP", "COLMAP", "NS", "CAM_BODY", "CAM_FORWARD", "CAM_DOWNWARD",
    "CANONICAL_FRAMES", "frame_by_name",
    "Point", "Pose", "Trajectory", "PointCloud", "assert_frame",
    "SE3", "Sim3",
    "FrameGraph",
    "register_loader", "get_loader", "available_loaders",
    "axis_permutation", "available_presets",
]
