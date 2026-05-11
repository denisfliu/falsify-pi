"""Perturbation framework — three surfaces, one suite.

Three stages, executed in order each control tick (or once per episode for
environment perturbations):

- ``ObservationPerturbation`` — mutates the `Observation` *after* sensors
  fill it and *before* the policy reads it. Used for image noise/blur/
  occlusion, fake state-estimate noise, etc.
- ``ActionPerturbation`` — mutates the `Trajectory` returned by the policy
  before the simulator follows it. Used for waypoint noise, scale, bias.
- ``EnvironmentPerturbation`` — mutates the gsplat scene (Splat-MOVER).
  Applied between episodes (and optionally each step).

Each perturbation declares its parameters in JSON-serializable form so the
suite can write a manifest with the exact perturbation set used per episode.

Frame contract
--------------
- `ObservationPerturbation` receives a frame-tagged `Observation` and returns
  a frame-tagged one — positions / poses inside `obs.data` keep their tags.
- `ActionPerturbation` operates on frame-tagged `Trajectory`. Position-space
  noise is applied in the trajectory's own frame; perturbations that need
  a specific frame convert internally via the `FrameGraph` passed at
  construction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Iterable, List, Optional

import numpy as np

from falsify.geometry import FrameGraph, Trajectory
from falsify.policy.observation import Observation


# ---------------------------------------------------------------------------
# Manifest serialization helper
# ---------------------------------------------------------------------------


def _jsonable(v: Any) -> Any:
    """Convert numpy / tuple / dataclass / dict values to JSON-friendly types."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, (tuple, list)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if is_dataclass(v):
        return {f.name: _jsonable(getattr(v, f.name)) for f in fields(v)}
    return v


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------


class Perturbation(ABC):
    """Base for any perturbation. Subclasses override `apply()` per stage."""

    name: str = ""

    def reset(self, rng: np.random.Generator) -> None:
        """Called at episode start. Default: no-op."""
        return None

    def manifest(self) -> dict:
        """Return a JSON-serializable summary of this perturbation's config."""
        return {"name": self.name or type(self).__name__, "type": type(self).__name__}


class ObservationPerturbation(Perturbation):
    @abstractmethod
    def apply(self, obs: Observation, *, rng: np.random.Generator) -> Observation:
        ...


class ActionPerturbation(Perturbation):
    @abstractmethod
    def apply(
        self,
        traj: Trajectory,
        *,
        rng: np.random.Generator,
        frame_graph: Optional[FrameGraph] = None,
    ) -> Trajectory:
        ...


class EnvironmentPerturbation(Perturbation):
    """Mutates the gsplat scene. Stub for v0 — concrete impls land when the
    Splat-MOVER integration is wired (Phase 6 / new gsplat asset)."""

    @abstractmethod
    def apply(self, gsplat: Any) -> None:
        ...


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------


class PerturbationSuite:
    """Holds all perturbations and dispatches them at the right stage.

    The suite owns a single numpy `Generator`. Each perturbation gets the
    same generator so the manifest + seed are sufficient to reproduce an
    episode. Sub-RNGs can be derived if needed.
    """

    def __init__(
        self,
        observation: Iterable[ObservationPerturbation] = (),
        action: Iterable[ActionPerturbation] = (),
        environment: Iterable[EnvironmentPerturbation] = (),
        seed: int | None = None,
    ) -> None:
        self.observation_perts: tuple[ObservationPerturbation, ...] = tuple(observation)
        self.action_perts: tuple[ActionPerturbation, ...] = tuple(action)
        self.environment_perts: tuple[EnvironmentPerturbation, ...] = tuple(environment)
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    # ---- lifecycle -----------------------------------------------------

    def reset(self) -> None:
        """Re-seed the RNG and call each perturbation's reset hook."""
        self._rng = np.random.default_rng(self._seed)
        for p in self._all():
            p.reset(self._rng)

    # ---- stage hooks ---------------------------------------------------

    def apply_observation(self, obs: Observation) -> Observation:
        for p in self.observation_perts:
            obs = p.apply(obs, rng=self._rng)
        return obs

    def apply_action(
        self,
        traj: Trajectory,
        frame_graph: Optional[FrameGraph] = None,
    ) -> Trajectory:
        for p in self.action_perts:
            traj = p.apply(traj, rng=self._rng, frame_graph=frame_graph)
        return traj

    def apply_environment(self, gsplat: Any) -> None:
        for p in self.environment_perts:
            p.apply(gsplat)

    # ---- manifest ------------------------------------------------------

    def manifest(self) -> dict:
        return {
            "seed": self._seed,
            "observation": [p.manifest() for p in self.observation_perts],
            "action": [p.manifest() for p in self.action_perts],
            "environment": [p.manifest() for p in self.environment_perts],
        }

    # ---- internals -----------------------------------------------------

    def _all(self) -> List[Perturbation]:
        return list(self.observation_perts) + list(self.action_perts) + list(self.environment_perts)

    def __len__(self) -> int:
        return len(self._all())
