"""Per-trial card sampling — shared between bundle generation and
streaming collection.

The eval-bundle generator (``scripts/generate_eval_bundles.py``) and the
recovery-trajectory collector (``scripts/collect_recovery_trajectories.py``)
both draw the same kind of trial: a start-position jitter around the
scene's nominal start, plus an optional gate-rigid-perturbation Δxyz/Δyaw
sampled within the scenario recipe's half-widths (rejection-sampled
against scene obstacles when declared).

Keeping these three helpers in one place guarantees both callers see
byte-identical cards for a given ``(master_seed, scenario_name,
scene_key, trial_index)`` tuple — the determinism contract that makes
A/B comparisons across runs meaningful.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def seed_for(master_seed: int, scenario_name: str, scene_key: str,
             trial_index: int) -> int:
    """Derive a stable per-trial seed.

    Hashing ``(scenario, scene, trial_index)`` together so adding a new
    scene or scenario can't shift the random draws of unrelated trials.
    """
    h = hash((master_seed, scenario_name, scene_key, trial_index))
    return abs(h) % (2**32)


def sample_start_mocap(scene_cfg: dict, rng: np.random.Generator,
                       enabled: bool) -> list[float]:
    """Uniform ±half_widths offset around ``scene_cfg.start_position_mocap``.

    Returns the nominal point unmodified when ``enabled`` is False or the
    scene declares no ``start_randomization.half_widths_mocap``.
    """
    nominal = np.asarray(scene_cfg["start_position_mocap"], dtype=np.float64)
    if not enabled:
        return nominal.tolist()
    half = (scene_cfg.get("start_randomization") or {}).get("half_widths_mocap")
    if not half:
        return nominal.tolist()
    half_arr = np.asarray(half, dtype=np.float64)
    offset = rng.uniform(-half_arr, +half_arr)
    return (nominal + offset).tolist()


def sample_gate_perturbation(
    recipe: dict,
    rng: np.random.Generator,
    *,
    scene_cfg: Optional[dict] = None,
) -> Optional[dict]:
    """Sample a Δxyz + Δyaw for the gate, rejecting draws that would push
    the gate into any declared scene obstacle (``scene_cfg["obstacles"]``).

    Without ``scene_cfg`` (legacy callers / scenes lacking ``obstacles``),
    falls back to plain uniform sampling. Z is always pinned to 0 — the
    gates don't levitate.
    """
    gp = recipe.get("gate_perturbation") or {}
    if not gp.get("enabled", False):
        return None
    half_xyz = list(gp.get("offset_half_widths", [0.0, 0.0, 0.0]))
    half_yaw = float(gp.get("yaw_half_width_rad", 0.0))
    half_xyz[2] = 0.0

    if scene_cfg is not None and (scene_cfg.get("obstacles") or []):
        from falsify.perturbations import sample_obstacle_safe_perturbation
        dxyz, dyaw = sample_obstacle_safe_perturbation(
            scene_cfg, half_xyz, half_yaw, rng,
            max_tries=int(gp.get("max_tries", 200)),
            max_growth_factor=float(gp.get("max_overlap_growth_factor", 1.5)),
        )
    else:
        half_xyz_arr = np.asarray(half_xyz, dtype=np.float64)
        dxyz = rng.uniform(low=-half_xyz_arr, high=+half_xyz_arr, size=(3,))
        dyaw = float(rng.uniform(-half_yaw, +half_yaw))
    dxyz = np.asarray(dxyz, dtype=np.float64)
    dxyz[2] = 0.0

    return {
        "name": "gate_rigid_perturbation",
        "delta_xyz": dxyz.tolist(),
        "delta_yaw_rad": float(dyaw),
    }
