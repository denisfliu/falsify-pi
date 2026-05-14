"""Trajectory planning from waypoint courses.

See ``CLAUDE.md`` for the workflow:
  course YAML → Course → plan_*(course, frame_graph) → Trajectory NPZ → ...

Today's planners:
- ``plan_spline``: cubic spline through positions; yaw per ``yaw_mode``.
  Adequate baseline for renderer / dataset bring-up.

Planners on the roadmap:
- ``plan_mpc``: FiGS VehicleRateMPC over a feasible reference. Stub.
- ``plan_splatnav``: SplatNav A* + spline through the gsplat. Stub.
"""

from .waypoints import Course, Waypoint, load_course, save_course
from .spline import plan_spline
from .perturbations import (
    CourseVariant, Direction, perturb_waypoint, sample_variants,
)

__all__ = [
    "Course",
    "Waypoint",
    "load_course",
    "save_course",
    "plan_spline",
    "CourseVariant",
    "Direction",
    "perturb_waypoint",
    "sample_variants",
]
