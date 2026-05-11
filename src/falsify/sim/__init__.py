"""Simulator wrapper + frame-tagged drone state.

Lazy-imports the FiGS rendering stack inside `renderer.GSplatRenderer`, so
the rest of falsify is importable without CUDA / acados.
"""

from .dynamics_state import DroneState
from .simulator import Simulator, SimulatorConfig, EpisodeTrace
# Note: ``renderer.GSplatRenderer`` is imported on demand to avoid pulling
# in the FiGS / nerfstudio stack at falsify import time.

__all__ = ["DroneState", "Simulator", "SimulatorConfig", "EpisodeTrace"]
