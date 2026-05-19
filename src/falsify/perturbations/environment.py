"""Environment perturbations — gsplat-level scene edits sampled per episode.

`GateRigidPerturbation` is the v0 concrete impl. It samples a random xyz
offset and a random yaw delta from configured uniform bounds and applies the
resulting rigid transform to the gate Gaussians of the loaded gsplat. The
selection AABB comes from the scene YAML's published ``gate_region:`` block
so the same metadata feeds the per-episode perturbation, the static
``scene_edits`` (in ``center_gate.yaml``), and any future falsification
sweep.

The classic Splat-MOVER-style multi-object / opacity / color edits stay
deferred until that integration lands; their slot is the same
`EnvironmentPerturbation` ABC, so adding them later won't churn the wiring.

Wiring contract
---------------
- Constructed by ``smoke_test.build_perturbations_factory`` from the
  perturbations YAML.
- Receives the scene YAML at construction time (the factory hands it in)
  so it can read ``gate_region:``. Without that, the perturbation can't
  know which Gaussians to move and raises at construction.
- At episode start the orchestrator calls
  ``suite.apply_environment(renderer)`` — the GSplatRenderer is the
  ``gsplat`` parameter advertised on the ABC. The perturbation calls
  ``renderer.apply_dynamic_edits([...])`` which restores baseline first,
  so per-episode perturbations don't compound.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from .base import EnvironmentPerturbation, _jsonable


# ---------------------------------------------------------------------------
# Stubs kept for back-compat — replaced by concrete impls below.
# ---------------------------------------------------------------------------


@dataclass
class StubEnvironmentPerturbation(EnvironmentPerturbation):
    """Placeholder retained so existing imports don't break. Prefer
    `GateRigidPerturbation` for gate jitter."""
    description: str = "stub environment perturbation"
    name: str = "env_stub"

    def apply(self, gsplat: Any) -> None:
        raise NotImplementedError(
            f"EnvironmentPerturbation not implemented: {self.description}."
        )

    def manifest(self) -> dict:
        return {"name": self.name, "type": "StubEnvironmentPerturbation",
                "description": self.description}


# ---------------------------------------------------------------------------
# GateRigidPerturbation
# ---------------------------------------------------------------------------


def _load_gate_region(scene_cfg: dict) -> dict:
    region = scene_cfg.get("gate_region")
    if not region:
        raise ValueError(
            "GateRigidPerturbation requires a `gate_region:` block in the "
            "scene YAML — declare it once per scene (see configs/scenes/"
            "left_gate.yaml for the canonical example)."
        )
    required = ("aabb_frame", "aabb_min", "aabb_max", "anchor", "normal")
    missing = [k for k in required if k not in region]
    if missing:
        raise ValueError(
            f"scene_cfg.gate_region is missing required keys {missing}; "
            f"got {sorted(region)}"
        )
    return region


@dataclass
class GateRigidPerturbation(EnvironmentPerturbation):
    """Per-episode random rigid translation + yaw of the gate Gaussians.

    Bounds are uniform half-widths around the gate's published anchor /
    normal. Default yaw bounds are zero (translation-only) — set
    ``yaw_half_width_rad`` to enable rotational jitter.

    The perturbation is constructed with the parsed scene YAML so it can
    read the `gate_region:` block; the YAML deserializer in
    ``smoke_test.build_perturbations_factory`` plumbs that through.
    """

    offset_half_widths: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw_half_width_rad: float = 0.0
    scene_cfg: Optional[dict] = None
    name: str = "gate_rigid_perturbation"

    # Sampled per-episode state; refreshed in `reset(rng)`.
    _delta_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    _delta_yaw: float = 0.0
    # Replay path: when set, `reset(rng)` skips sampling and uses these
    # absolute values verbatim. Driven by the orchestrator's
    # `perturbation_overrides` (i.e. trial cards).
    _absolute_delta_xyz: Optional[np.ndarray] = None
    _absolute_delta_yaw: Optional[float] = None

    def __post_init__(self):
        hw = np.asarray(self.offset_half_widths, dtype=np.float64)
        if hw.shape != (3,):
            raise ValueError(
                f"offset_half_widths must be (3,); got {hw.shape}"
            )
        if (hw < 0).any() or self.yaw_half_width_rad < 0:
            raise ValueError(
                "bounds must be non-negative half-widths "
                f"(got offsets={hw.tolist()}, yaw={self.yaw_half_width_rad})"
            )
        object.__setattr__(self, "offset_half_widths", tuple(hw.tolist()))
        if self.scene_cfg is None:
            raise ValueError("GateRigidPerturbation requires scene_cfg=...")
        # Eager-validate gate_region so construction failures are caught at
        # factory time, not buried inside the first apply().
        _load_gate_region(self.scene_cfg)

    # ---- lifecycle -----------------------------------------------------

    def reset(self, rng: np.random.Generator) -> None:
        if self._absolute_delta_xyz is not None and self._absolute_delta_yaw is not None:
            # Replay path: use the trial card's absolute deltas.
            self._delta_xyz = np.asarray(self._absolute_delta_xyz, dtype=np.float64)
            self._delta_yaw = float(self._absolute_delta_yaw)
            return
        hw = np.asarray(self.offset_half_widths, dtype=np.float64)
        self._delta_xyz = rng.uniform(low=-hw, high=hw, size=(3,))
        self._delta_yaw = float(
            rng.uniform(-self.yaw_half_width_rad, self.yaw_half_width_rad)
        )

    def set_absolute_deltas(
        self,
        delta_xyz: Sequence[float] | np.ndarray,
        delta_yaw_rad: float,
    ) -> None:
        """Switch this perturbation into replay mode for the next ``reset()``.

        Trial-card-driven evaluations call this so every policy sees the
        same exact gate pose. ``delta_xyz`` and ``delta_yaw_rad`` are in the
        same MOCAP frame the bounds-driven sampler uses.
        """
        arr = np.asarray(delta_xyz, dtype=np.float64).reshape(-1)
        if arr.shape != (3,):
            raise ValueError(f"delta_xyz must be (3,); got {arr.shape}")
        self._absolute_delta_xyz = arr
        self._absolute_delta_yaw = float(delta_yaw_rad)

    # ---- application ---------------------------------------------------

    def apply(self, gsplat: Any) -> None:
        """Apply the sampled delta to the renderer's gsplat.

        ``gsplat`` is the `GSplatRenderer` passed by the orchestrator. The
        perturbation builds a single `RigidTransformAABB` scene edit that
        translates the gate's anchor by ``_delta_xyz`` and rotates its
        normal in the xy-plane by ``_delta_yaw``, then hands it to
        ``renderer.apply_dynamic_edits([edit])`` which takes care of the
        baseline restore.
        """
        if not hasattr(gsplat, "apply_dynamic_edits"):
            raise TypeError(
                "GateRigidPerturbation.apply expected a GSplatRenderer "
                f"(with `apply_dynamic_edits`), got {type(gsplat).__name__}. "
                "Wire the renderer into run_episode and call "
                "suite.apply_environment(renderer)."
            )
        edit = self._build_edit()
        n = gsplat.apply_dynamic_edits([edit])
        # Don't spam stdout for zero-delta no-ops; otherwise print one line.
        if np.allclose(self._delta_xyz, 0.0) and abs(self._delta_yaw) < 1e-9:
            return
        print(
            f"[gate_perturb] Δxyz={self._delta_xyz.round(3).tolist()} "
            f"Δyaw={np.degrees(self._delta_yaw):.1f}° → {n} gaussians moved"
        )

    def _build_edit(self):
        # Local import — keeps the perturbations package importable without
        # the sim package's heavy deps (torch / FiGS) loaded.
        from falsify.sim.scene_edits import RigidTransformAABB, _Box

        region = _load_gate_region(self.scene_cfg)
        aabb_frame = region["aabb_frame"]
        anchor = np.asarray(region["anchor"], dtype=np.float64)
        normal = np.asarray(region["normal"], dtype=np.float64)
        normal_xy = normal.copy()
        normal_xy[2] = 0.0
        n_norm = np.linalg.norm(normal_xy[:2])
        if n_norm < 1e-9:
            raise ValueError(
                f"gate_region.normal must have non-zero xy component; got {normal.tolist()}"
            )
        normal_xy[:2] /= n_norm

        # Rotate the normal by _delta_yaw about +z to get the target normal.
        c, s = np.cos(self._delta_yaw), np.sin(self._delta_yaw)
        target_normal = np.array([
            c * normal_xy[0] - s * normal_xy[1],
            s * normal_xy[0] + c * normal_xy[1],
            0.0,
        ])
        target_anchor = anchor + self._delta_xyz

        include_boxes = tuple(
            _Box(min=e["min"], max=e["max"])
            for e in (region.get("include_aabbs") or [])
        )
        exclude_boxes = tuple(
            _Box(min=e["min"], max=e["max"])
            for e in (region.get("exclude_aabbs") or [])
        )

        return RigidTransformAABB(
            name=f"{self.name}_edit",
            target_aabb_frame=aabb_frame,
            target_aabb_min=np.asarray(region["aabb_min"], dtype=np.float64),
            target_aabb_max=np.asarray(region["aabb_max"], dtype=np.float64),
            source_anchor=anchor,
            target_anchor=target_anchor,
            source_normal=normal_xy,
            target_normal=target_normal,
            transform_frame=aabb_frame,
            applies_to_scene_objects=tuple(
                region.get("applies_to_scene_objects", []) or []
            ),
            include_aabbs=include_boxes,
            exclude_aabbs=exclude_boxes,
        )

    # ---- manifest ------------------------------------------------------

    def manifest(self) -> dict:
        return {
            "name": self.name,
            "type": "GateRigidPerturbation",
            "offset_half_widths": _jsonable(self.offset_half_widths),
            "yaw_half_width_rad": _jsonable(self.yaw_half_width_rad),
            "sampled_delta_xyz": _jsonable(self._delta_xyz),
            "sampled_delta_yaw_rad": _jsonable(self._delta_yaw),
        }
