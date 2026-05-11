"""Environment perturbations — Splat-MOVER scene edits.

v0 ships only a documented stub. Concrete edits (rigid transform of a
labeled object group, opacity changes, color shifts) land once the new
gsplat asset is available and we wire `external/Splat-MOVER` into the
renderer.

When implementing a concrete edit:
- Accept a Gaussian-mask selection (path to a saved boolean mask, or a CLIP
  query) so selection is config-driven.
- Mutate ``model.means.data`` / ``model.opacities.data`` etc. in place.
- Store a backup at `reset(rng)` time so `restore()` returns the gsplat to
  its original state between episodes (the suite calls `apply` once per
  episode start).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import EnvironmentPerturbation


@dataclass
class StubEnvironmentPerturbation(EnvironmentPerturbation):
    """Placeholder until the Splat-MOVER pipeline is wired in.

    Construct with ``description`` only; calling `apply` raises so that
    forgetting to remove the stub from a real run fails loudly.
    """
    description: str = "stub environment perturbation"
    name: str = "env_stub"

    def apply(self, gsplat: Any) -> None:
        raise NotImplementedError(
            f"EnvironmentPerturbation not yet implemented: {self.description}. "
            f"Concrete impls land once a new gsplat asset + Splat-MOVER integration are ready."
        )

    def manifest(self) -> dict:
        return {"name": self.name, "type": "StubEnvironmentPerturbation",
                "description": self.description}
