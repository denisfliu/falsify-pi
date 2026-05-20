"""Three perturbation surfaces (observation / action / environment) + a suite.

See ``CLAUDE.md`` in this directory for the design.
"""

from .base import (
    Perturbation, ObservationPerturbation, ActionPerturbation, EnvironmentPerturbation,
    PerturbationSuite,
)
from .action import PositionNoise, PositionBias, VelocityScale
from .observation import ImageGaussianNoise, ImageBlur, StateNoise
from .environment import StubEnvironmentPerturbation, GateRigidPerturbation
from .gate_obstacle_check import (
    is_perturbation_obstacle_safe,
    sample_obstacle_safe_perturbation,
)

__all__ = [
    "Perturbation",
    "ObservationPerturbation", "ActionPerturbation", "EnvironmentPerturbation",
    "PerturbationSuite",
    "PositionNoise", "PositionBias", "VelocityScale",
    "ImageGaussianNoise", "ImageBlur", "StateNoise",
    "StubEnvironmentPerturbation", "GateRigidPerturbation",
    "is_perturbation_obstacle_safe", "sample_obstacle_safe_perturbation",
]
