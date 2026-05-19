"""Trajectory planning from waypoint courses.

See ``CLAUDE.md`` for the workflow:
  course YAML → Course → plan_*(course, frame_graph) → Trajectory NPZ → ...

Today's planners:
- ``plan_spline``: cubic spline through positions; yaw per ``yaw_mode``.
  Adequate baseline for renderer / dataset bring-up.
- ``plan_mpc``: FiGS ``VehicleRateMPC`` tracking a min-time-snap
  reference through the course's waypoints, integrated with an
  ``acados`` IRK solver. Dynamically feasible. Heavy first-call cost
  due to acados JIT (~30 s); the produced shared library is isolated
  per-call in a tempfile so concurrent planners don't collide.

Planner on the roadmap:
- ``plan_splatnav``: SplatNav A* + spline through the gsplat. Stub.
"""

from .waypoints import (
    Course,
    CorrectivePerturbation,
    Perturbation,
    TrajectoryPerturbation,
    Waypoint,
    load_course,
    save_course,
)
from .spline import plan_spline
from .mpc import plan_mpc
from .perturbations import (
    CourseVariant,
    Direction,
    perturb_waypoint,
    sample_stochastic_variants,
    sample_variants,
)

__all__ = [
    "Course",
    "CorrectivePerturbation",
    "Perturbation",
    "TrajectoryPerturbation",
    "Waypoint",
    "load_course",
    "save_course",
    "plan_spline",
    "plan_mpc",
    "CourseVariant",
    "Direction",
    "perturb_waypoint",
    "sample_stochastic_variants",
    "sample_variants",
]
