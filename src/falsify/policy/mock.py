"""Mock policies for smoke tests and recovery exercising.

These declare no modality requirements — they ignore images entirely — so
rollouts can run without invoking gsplat rendering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from falsify.geometry import Trajectory, Point
from .base import Policy
from .observation import Observation


@dataclass
class MockStraightLineConfig:
    goal: Point                      # in the same frame as the drone state (typically "ned")
    speed: float = 1.0               # m/s along the straight line
    horizon_s: float = 5.0           # length of the emitted reference trajectory
    n_waypoints: int = 50            # samples in the emitted trajectory


class MockStraightLine(Policy):
    """Emit a straight-line trajectory from the current state toward the goal."""

    required_modalities = frozenset()

    def __init__(self, cfg: MockStraightLineConfig) -> None:
        self.cfg = cfg

    def observe(self, obs: Observation) -> Trajectory:
        start = obs.state.pos
        if start.frame.name != self.cfg.goal.frame.name:
            raise ValueError(
                f"goal frame {self.cfg.goal.frame.name!r} != state frame "
                f"{start.frame.name!r}; convert the goal at construction time"
            )
        direction = self.cfg.goal.xyz - start.xyz
        dist = float(np.linalg.norm(direction))
        if dist < 1e-9:
            unit = np.zeros(3)
        else:
            unit = direction / dist
        max_reach = self.cfg.speed * self.cfg.horizon_s
        reach = min(dist, max_reach)
        end = start.xyz + unit * reach
        t0 = obs.state.t
        times = np.linspace(t0, t0 + self.cfg.horizon_s, self.cfg.n_waypoints)
        positions = np.linspace(start.xyz, end, self.cfg.n_waypoints)
        velocities = np.tile(unit * self.cfg.speed, (self.cfg.n_waypoints, 1))
        return Trajectory(
            times=times,
            positions=positions,
            frame=start.frame,
            velocities=velocities,
        )


@dataclass
class MockNoisyConfig:
    goal: Point
    speed: float = 1.0
    horizon_s: float = 5.0
    n_waypoints: int = 50
    position_noise_std: float = 0.05
    seed: int | None = None


class MockNoisy(Policy):
    """Straight-line toward goal with bounded random position perturbations.

    Used to exercise the failure detector + SplatNav recovery without a real
    VLA server.
    """

    required_modalities = frozenset()

    def __init__(self, cfg: MockNoisyConfig) -> None:
        self.cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.cfg.seed)

    def observe(self, obs: Observation) -> Trajectory:
        base = MockStraightLine(MockStraightLineConfig(
            goal=self.cfg.goal,
            speed=self.cfg.speed,
            horizon_s=self.cfg.horizon_s,
            n_waypoints=self.cfg.n_waypoints,
        )).observe(obs)
        noise = self._rng.normal(
            scale=self.cfg.position_noise_std, size=base.positions.shape,
        )
        return Trajectory(
            times=base.times,
            positions=base.positions + noise,
            frame=base.frame,
            velocities=base.velocities,
        )
