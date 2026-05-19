"""Recovery planners.

- ``SplatNavPlanner`` — collision-free A*+spline through a Gaussian splat
  (NED in, NED out). Used when the recovery must compute a free-space
  path around obstacles.
- ``CoursedMpcPlanner`` — dynamically-feasible MPC tracking a course
  YAML, started from ``last_safe_state``. Used in the falsification
  pipeline to "snap back" onto the nominal course after a VLA miss.
"""

from .planner import (
    RecoveryConfig, RecoveryResult, SplatNavPlanner, PlannerBackend,
)
from .coursed_mpc import CoursedMpcPlanner
from .seed_sampling import sample_recovery_seed, bias_for

__all__ = [
    "RecoveryConfig", "RecoveryResult", "SplatNavPlanner", "PlannerBackend",
    "CoursedMpcPlanner",
    "sample_recovery_seed", "bias_for",
]
