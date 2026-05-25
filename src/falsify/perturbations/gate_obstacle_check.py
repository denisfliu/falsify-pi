"""Reject gate perturbations that push the gate into a scene obstacle.

Used by the bundle generator (`scripts/eval/generate_eval_bundles.py`) and the
runtime `GateRigidPerturbation` to ensure sampled Δxyz / Δyaw don't
plausibly merge the gate Gaussians into the table (or any other obstacle
declared under `scene_cfg["obstacles"]`).

Algorithm
---------
Given the perturbed gate's 8 AABB corners (rotated about the gate anchor
by Δyaw, then translated by Δxyz, all in MOCAP), we re-axis-align them
to a perturbed AABB and compute the intersection volume with each
obstacle AABB. The sample is accepted iff for every obstacle:

    perturbed_overlap_volume <= max_growth_factor * nominal_overlap_volume

where ``nominal_overlap_volume`` is the unperturbed intersection (zero
when the gate and obstacle nominally don't touch). ``max_growth_factor``
defaults to 1.5 — allowing a 50% growth in any existing overlap. For
obstacle pairs that don't touch at nominal (e.g. center_gate vs the
left_table), the rule reduces to "no new overlap allowed".

Frame contract
--------------
All AABBs are in MOCAP. Scene authors publish the gate's AABB under
`gate_region` and obstacle AABBs under `obstacles`. Both use the same
`aabb_frame: mocap` convention.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def aabb_overlap_volume(
    a_min: np.ndarray, a_max: np.ndarray,
    b_min: np.ndarray, b_max: np.ndarray,
) -> float:
    """Axis-aligned-box intersection volume in 3D. 0 when disjoint."""
    lo = np.maximum(a_min, b_min)
    hi = np.minimum(a_max, b_max)
    if (hi <= lo).any():
        return 0.0
    return float(np.prod(hi - lo))


def perturbed_gate_aabb(
    gate_aabb_min: np.ndarray,
    gate_aabb_max: np.ndarray,
    gate_anchor: np.ndarray,
    delta_xyz: np.ndarray,
    delta_yaw_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply (Δxyz, Δyaw) to the gate's AABB and return a new axis-aligned
    bracket of the rotated+translated corners. Yaw is about the gate
    anchor's z-axis (gates are vertical — pitch/roll not supported)."""
    mn = np.asarray(gate_aabb_min, dtype=np.float64)
    mx = np.asarray(gate_aabb_max, dtype=np.float64)
    anchor = np.asarray(gate_anchor, dtype=np.float64)
    dxyz = np.asarray(delta_xyz, dtype=np.float64).reshape(3)
    yaw = float(delta_yaw_rad)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    # 8 corners of the gate AABB.
    corners = np.array([
        [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
        [mn[0], mx[1], mn[2]], [mx[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
        [mn[0], mx[1], mx[2]], [mx[0], mx[1], mx[2]],
    ])
    # Rigid transform about the anchor: rotate, then translate.
    moved = (R @ (corners - anchor).T).T + anchor + dxyz
    return moved.min(axis=0), moved.max(axis=0)


def is_perturbation_obstacle_safe(
    scene_cfg: dict,
    delta_xyz: np.ndarray,
    delta_yaw_rad: float,
    *,
    max_growth_factor: float = 1.5,
    absolute_max_new_overlap_m3: float = 0.01,
) -> bool:
    """Return ``True`` when the perturbed gate AABB doesn't intrude into
    any declared scene obstacle beyond a tolerable threshold.

    The check is permissive when the scene declares no ``gate_region``
    or no ``obstacles`` — those are configurations where the sampler
    has nothing to reject against.

    The rule per obstacle:

    - If nominal overlap is **zero**: any growth in overlap is bounded
      by ``absolute_max_new_overlap_m3`` (default 1 cm³ — essentially
      zero, with a tiny float-slop tolerance).
    - If nominal overlap is **positive** (gate already touches the
      obstacle slightly at nominal — common for left/right_gate posts
      near the table edge): the perturbation may only grow that overlap
      by ``max_growth_factor`` (default 1.5×).
    """
    region = scene_cfg.get("gate_region")
    obstacles = scene_cfg.get("obstacles") or []
    if not region or not obstacles:
        return True

    gate_min_raw = np.asarray(region["aabb_min"], dtype=np.float64)
    gate_max_raw = np.asarray(region["aabb_max"], dtype=np.float64)
    anchor = np.asarray(region["anchor"], dtype=np.float64)

    # Nominal: zero-perturbation AABB (same as raw — provided as a
    # safety check against future refactors that change `perturbed_*`).
    nom_min, nom_max = perturbed_gate_aabb(
        gate_min_raw, gate_max_raw, anchor,
        delta_xyz=np.zeros(3), delta_yaw_rad=0.0,
    )
    pert_min, pert_max = perturbed_gate_aabb(
        gate_min_raw, gate_max_raw, anchor,
        delta_xyz=delta_xyz, delta_yaw_rad=delta_yaw_rad,
    )

    for ob in obstacles:
        if ob.get("aabb_frame", "mocap") != "mocap":
            raise NotImplementedError(
                f"obstacle {ob.get('name')!r}: only aabb_frame='mocap' supported"
            )
        ob_min = np.asarray(ob["aabb_min"], dtype=np.float64)
        ob_max = np.asarray(ob["aabb_max"], dtype=np.float64)
        nom_overlap = aabb_overlap_volume(nom_min, nom_max, ob_min, ob_max)
        pert_overlap = aabb_overlap_volume(pert_min, pert_max, ob_min, ob_max)
        if nom_overlap <= 1e-12:
            if pert_overlap > absolute_max_new_overlap_m3:
                return False
        else:
            if pert_overlap > max_growth_factor * nom_overlap:
                return False
    return True


def sample_obstacle_safe_perturbation(
    scene_cfg: dict,
    offset_half_widths: Sequence[float],
    yaw_half_width_rad: float,
    rng: np.random.Generator,
    *,
    max_tries: int = 200,
    max_growth_factor: float = 1.5,
) -> tuple[np.ndarray, float]:
    """Rejection-sample a (Δxyz, Δyaw) that doesn't push the gate into
    any declared obstacle. Falls back to the last sample after
    ``max_tries`` to avoid infinite loops on over-tight configs.

    Returns
    -------
    ``(delta_xyz: (3,), delta_yaw_rad: float)`` — both authored in MOCAP.
    """
    half_xyz = np.asarray(offset_half_widths, dtype=np.float64)
    half_yaw = float(yaw_half_width_rad)
    last_xyz = np.zeros(3)
    last_yaw = 0.0
    for _ in range(max_tries):
        dxyz = rng.uniform(low=-half_xyz, high=+half_xyz, size=(3,))
        dyaw = float(rng.uniform(-half_yaw, +half_yaw))
        if is_perturbation_obstacle_safe(
            scene_cfg, dxyz, dyaw, max_growth_factor=max_growth_factor,
        ):
            return dxyz, dyaw
        last_xyz, last_yaw = dxyz, dyaw
    # All tries rejected — log loudly. Returning the last sample so
    # callers still get *something*; in practice we'd rather rework the
    # half-widths than silently accept a bad sample.
    print(
        f"[gate_obstacle_check] WARNING: {max_tries} samples all rejected "
        f"against scene obstacles. Half-widths xyz={half_xyz.tolist()} "
        f"yaw={half_yaw:.3f} rad may be too large for this scene's clearance."
    )
    return last_xyz, last_yaw
