"""Action-side perturbations: mutate the policy's emitted `Trajectory`.

We work in waypoint space (positions/velocities) since the orchestrator's
v0 integrator follows the trajectory directly. When the FiGS-MPC integrator
lands, additional perturbations (thrust noise, body-rate bias) can be added
here without changing the `Trajectory` interface — the simulator will apply
them between MPC output and the dynamics step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from falsify.geometry import FrameGraph, Trajectory
from .base import ActionPerturbation, _jsonable


@dataclass
class PositionNoise(ActionPerturbation):
    """Add zero-mean Gaussian noise to every waypoint position.

    The noise is **isotropic in the input trajectory's frame**. In normal
    orchestrator flow that frame is always NED (the policy contract), so
    ``std`` is in NED meters per axis. If a future caller hand-feeds a
    non-NED trajectory, the noise applies in that frame.
    """
    std: float = 0.05
    name: str = "position_noise"

    def apply(
        self,
        traj: Trajectory,
        *,
        rng: np.random.Generator,
        frame_graph: Optional[FrameGraph] = None,
    ) -> Trajectory:
        noise = rng.normal(scale=self.std, size=traj.positions.shape)
        return Trajectory(
            times=traj.times,
            positions=traj.positions + noise,
            frame=traj.frame,
            velocities=traj.velocities,
            quaternions=traj.quaternions,
        )

    def manifest(self) -> dict:
        return {"name": self.name, "type": "PositionNoise", "std": _jsonable(self.std)}


@dataclass
class PositionBias(ActionPerturbation):
    """Add a constant 3-vector bias to every waypoint position.

    The bias is applied in the **input trajectory's frame**. Same convention
    as `PositionNoise`: in normal orchestrator flow this is NED.
    """
    bias_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    name: str = "position_bias"

    def apply(
        self,
        traj: Trajectory,
        *,
        rng: np.random.Generator,
        frame_graph: Optional[FrameGraph] = None,
    ) -> Trajectory:
        bias = np.asarray(self.bias_xyz, dtype=np.float64)
        return Trajectory(
            times=traj.times,
            positions=traj.positions + bias[None, :],
            frame=traj.frame,
            velocities=traj.velocities,
            quaternions=traj.quaternions,
        )

    def manifest(self) -> dict:
        return {
            "name": self.name, "type": "PositionBias",
            "bias_xyz": _jsonable(self.bias_xyz),
        }


@dataclass
class VelocityScale(ActionPerturbation):
    """Scale waypoint velocities (e.g. simulate underpowered motors)."""
    scale: float = 1.0
    name: str = "velocity_scale"

    def apply(
        self,
        traj: Trajectory,
        *,
        rng: np.random.Generator,
        frame_graph: Optional[FrameGraph] = None,
    ) -> Trajectory:
        new_v = None if traj.velocities is None else traj.velocities * self.scale
        return Trajectory(
            times=traj.times,
            positions=traj.positions,
            frame=traj.frame,
            velocities=new_v,
            quaternions=traj.quaternions,
        )

    def manifest(self) -> dict:
        return {"name": self.name, "type": "VelocityScale", "scale": _jsonable(self.scale)}
