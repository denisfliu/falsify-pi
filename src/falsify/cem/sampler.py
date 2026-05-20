"""Trial-card authoring for CEM iterations.

The CEM loop and the eval-suite bundle generator both need to take a 6-d
``theta`` and write a trial-card JSON in the shape that
``run_eval_campaign.py`` consumes. This module owns that conversion in
one place so the two callers can't drift.

Trial-card schema (kept in sync with ``scripts/generate_eval_bundles.py``)::

    {
      "scenario":            <str>,
      "scene":               <repo-relative path to scene YAML>,
      "scene_key":           <str>,
      "safety":              <repo-relative path to safety YAML>,
      "recovery":            <repo-relative path | null>,
      "prompt_name":         <str>,
      "prompt":              <str>,
      "trial_index":         <int>,
      "master_seed":         <int>,
      "trial_seed":          <int>,
      "start_position_mocap":[x, y, z],
      "start_ned":           [n, e, d],
      "gate_perturbation": {
        "name": "gate_rigid_perturbation",
        "delta_xyz":    [dx, dy, 0.0],
        "delta_yaw_rad": dyaw
      },
      "cem_provenance": {                    # only on CEM cards
        "distribution_path": <repo-relative path | abs path>,
        "iter_index":        <int | null>,
        "param_names":       [6 strings],
        "theta":             [6 floats],
        "target_failure_type": <str | null>
      }
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from falsify.geometry import Point
from falsify.io import build_frame_graph, load_yaml

from falsify.cem.distribution import (
    GaussianBoxDistribution,
    PARAM_NAMES,
)


def theta_to_start_and_gate(
    theta: np.ndarray,
    scene_cfg: dict,
) -> tuple[list[float], dict]:
    """Turn a 6-d θ into ``(start_position_mocap, gate_perturbation)``.

    ``theta`` is an *offset* around the scene's nominal start; the
    gate-perturbation z-component is always 0 by construction.
    """
    unpacked = GaussianBoxDistribution.unpack(theta)
    nominal = np.asarray(scene_cfg["start_position_mocap"], dtype=np.float64)
    start_mocap = (nominal + np.asarray(unpacked["start_delta_mocap"])).tolist()
    gate_pert = {
        "name": "gate_rigid_perturbation",
        "delta_xyz":     list(unpacked["gate_delta_xyz"]),
        "delta_yaw_rad": float(unpacked["gate_delta_yaw_rad"]),
    }
    return start_mocap, gate_pert


def mocap_start_to_ned(scene_yaml: Path, start_mocap: list[float]) -> list[float]:
    scene_cfg = load_yaml(scene_yaml)
    fg = build_frame_graph(scene_cfg, base_path=scene_yaml.parent)
    start_pt = Point.of(*start_mocap, fg.frame("mocap"))
    ned = fg.convert(start_pt, to="ned")
    return list(map(float, ned.xyz))


def write_cem_trial_cards(
    *,
    distribution: GaussianBoxDistribution,
    rng: np.random.Generator,
    n_samples: int,
    scenario_name: str,
    scene_yaml: Path,
    scene_key: str,
    safety_yaml: Path,
    prompt_name: str,
    prompt_text: str,
    master_seed: int,
    iter_index: int,
    distribution_path: Path,
    repo_root: Path,
    out_dir: Path,
    recovery_yaml: Optional[Path] = None,
) -> list[Path]:
    """Sample ``n_samples`` θs and write trial cards under ``out_dir``.

    The cards are written as ``trial_000.json``, ``trial_001.json``, ...
    Returns the list of written paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_cfg = load_yaml(scene_yaml)
    thetas = distribution.sample(rng, n=n_samples)

    def _relpath(p: Path) -> str:
        return str(p.relative_to(repo_root)) if p.is_relative_to(repo_root) else str(p)

    paths: list[Path] = []
    for i, theta in enumerate(thetas):
        start_mocap, gate_pert = theta_to_start_and_gate(theta, scene_cfg)
        start_ned = mocap_start_to_ned(scene_yaml, start_mocap)
        card = {
            "scenario": scenario_name,
            "scene":      _relpath(scene_yaml),
            "scene_key":  scene_key,
            "safety":     _relpath(safety_yaml),
            "recovery":   _relpath(recovery_yaml) if recovery_yaml is not None else None,
            "prompt_name": prompt_name,
            "prompt":      prompt_text,
            "trial_index": i,
            "master_seed": int(master_seed),
            # No deterministic seed: θ was already drawn from the rng.
            # Record the position of this card in the iteration's draw
            # order so a re-run can be hashed if needed.
            "trial_seed":  None,
            "start_position_mocap": start_mocap,
            "start_ned":            start_ned,
            "gate_perturbation":    gate_pert,
            "cem_provenance": {
                "distribution_path":   _relpath(distribution_path),
                "iter_index":          int(iter_index),
                "param_names":         list(PARAM_NAMES),
                "theta":               theta.tolist(),
                "target_failure_type": distribution.target_failure_type,
            },
        }
        p = out_dir / f"trial_{i:03d}.json"
        p.write_text(json.dumps(card, indent=2))
        paths.append(p)
    return paths
