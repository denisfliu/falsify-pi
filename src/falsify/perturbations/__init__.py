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

__all__ = [
    "Perturbation",
    "ObservationPerturbation", "ActionPerturbation", "EnvironmentPerturbation",
    "PerturbationSuite",
    "PositionNoise", "PositionBias", "VelocityScale",
    "ImageGaussianNoise", "ImageBlur", "StateNoise",
    "StubEnvironmentPerturbation", "GateRigidPerturbation",
]
