"""Generate per-trial JSON bundles ("trial cards") from a scenario YAML.

A scenario YAML (under ``configs/eval_suite/``) declares the scenes to
evaluate, the prompt for each, the safety config, the recipe (start
jitter on/off, gate perturbation bounds), the number of trials, and a
master seed.

This script samples ``n_trials`` *absolute* perturbation values per
(scenario, scene) and writes them to
``runs/eval_bundles/<scenario>/<scene>/trial_NNN.json``. Every value the
campaign runner needs is captured in the card — so two policies running
on the same card see byte-identical conditions even if the orchestrator's
sampler implementations later change.

Idempotent given the same ``--scenario`` YAML + master_seed in that YAML.

Usage:

    PYTHONPATH=src python scripts/generate_eval_bundles.py \\
        --scenario configs/eval_suite/pure.yaml
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from falsify.geometry import Point
from falsify.io import build_frame_graph, load_yaml


REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve(p: str | Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (REPO_ROOT / pp).resolve()


@dataclass
class SceneEntry:
    scene_yaml: Path
    prompt_name: str
    safety_yaml: Path
    # Per-entry overrides — let one scene YAML serve multiple eval entries
    # (e.g. center_gate.yaml under two different prompts so we evaluate
    # center-from-left and center-from-right against the same scene).
    scene_key_override: Optional[str] = None
    n_trials_override: Optional[int] = None
    # Optional recovery YAML. When set, the campaign runner wires a
    # CoursedMpcPlanner for this scene; a failing trial produces a
    # recovery_trajectory.npz that can be exported to a training parquet.
    # When absent, trials run eval-only (failures recorded, no replan).
    recovery_yaml: Optional[Path] = None
    # Optional per-scene CEM distribution override. When the scenario's
    # `recipe.cem_distribution` is set, a per-scene override here lets a
    # single scenario YAML wire up one converged distribution *per scene*
    # — necessary because each scene's CEM loop emits its own JSON.
    cem_distribution: Optional[dict] = None


def _load_scenes(scenario: dict) -> list[SceneEntry]:
    out: list[SceneEntry] = []
    for s in scenario["scenes"]:
        out.append(SceneEntry(
            scene_yaml=_resolve(s["scene"]),
            prompt_name=s["prompt_name"],
            safety_yaml=_resolve(s["safety"]),
            scene_key_override=s.get("scene_key_override"),
            n_trials_override=(
                int(s["n_trials_override"])
                if s.get("n_trials_override") is not None else None
            ),
            recovery_yaml=_resolve(s["recovery"]) if s.get("recovery") else None,
            cem_distribution=s.get("cem_distribution"),
        ))
    return out


def _seed_for(master_seed: int, scenario_name: str, scene_key: str,
              trial_index: int) -> int:
    """Derive a stable per-trial seed.

    Hashing ``(scenario, scene, trial_index)`` together so adding a new
    scene or scenario can't shift the random draws of unrelated trials.
    """
    h = hash((master_seed, scenario_name, scene_key, trial_index))
    # numpy Generator expects an unsigned 32/64 bit int; squeeze hash.
    return abs(h) % (2**32)


def _sample_start_mocap(scene_cfg: dict, rng: np.random.Generator,
                        enabled: bool) -> list[float]:
    nominal = np.asarray(scene_cfg["start_position_mocap"], dtype=np.float64)
    if not enabled:
        return nominal.tolist()
    half = (scene_cfg.get("start_randomization") or {}).get("half_widths_mocap")
    if not half:
        return nominal.tolist()
    half_arr = np.asarray(half, dtype=np.float64)
    offset = rng.uniform(-half_arr, +half_arr)
    return (nominal + offset).tolist()


def _sample_gate_perturbation(
    recipe: dict,
    rng: np.random.Generator,
    *,
    scene_cfg: Optional[dict] = None,
) -> Optional[dict]:
    """Sample a Δxyz + Δyaw for the gate, rejecting draws that would push
    the gate into any declared scene obstacle (`scene_cfg["obstacles"]`).

    Without `scene_cfg` (legacy callers / scenes lacking `obstacles`),
    falls back to plain uniform sampling.
    """
    gp = recipe.get("gate_perturbation") or {}
    if not gp.get("enabled", False):
        return None
    half_xyz = list(gp.get("offset_half_widths", [0.0, 0.0, 0.0]))
    half_yaw = float(gp.get("yaw_half_width_rad", 0.0))
    # Pin gate_dz to 0 in the sampling envelope — gates don't levitate.
    half_xyz[2] = 0.0

    if scene_cfg is not None and (scene_cfg.get("obstacles") or []):
        # Rejection-sample against declared obstacles. Tunables can be
        # promoted to the recipe later if a scene needs custom tolerance.
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
    dxyz[2] = 0.0   # belt-and-braces — never let z drift through

    return {
        "name": "gate_rigid_perturbation",
        "delta_xyz": dxyz.tolist(),
        "delta_yaw_rad": float(dyaw),
    }


def _theta_to_card_fields(
    theta: np.ndarray,
    scene_cfg: dict,
) -> tuple[list[float], dict]:
    """Turn a 6-d CEM sample into ``(start_mocap, gate_perturbation)``.

    The 6-d ordering is the canonical CEM parameter layout — see
    ``falsify.cem.distribution.PARAM_NAMES``. ``theta`` is an offset
    around the scene's nominal start; ``gate_perturbation``'s z-component
    is always zero by construction.
    """
    from falsify.cem.distribution import GaussianBoxDistribution
    unpacked = GaussianBoxDistribution.unpack(theta)
    nominal = np.asarray(scene_cfg["start_position_mocap"], dtype=np.float64)
    start_mocap = (nominal + np.asarray(unpacked["start_delta_mocap"])).tolist()
    gate_pert = {
        "name": "gate_rigid_perturbation",
        "delta_xyz": list(unpacked["gate_delta_xyz"]),
        "delta_yaw_rad": float(unpacked["gate_delta_yaw_rad"]),
    }
    return start_mocap, gate_pert


def _sample_from_cem_distribution(
    recipe: dict,
    scene_cfg: dict,
    rng: np.random.Generator,
    scene_entry_overrides: Optional[dict] = None,
) -> tuple[list[float], dict, dict]:
    """Sample start + gate from a converged CEM distribution.

    Returns ``(start_mocap, gate_perturbation, cem_provenance)``. The
    provenance block is recorded on every card so post-hoc you can
    trace which CEM run a given trial came from. The distribution file
    path is resolved relative to ``REPO_ROOT``.

    Per-scene override: if ``scene_entry_overrides`` carries a
    ``cem_distribution`` block (with its own ``path``), use it instead of
    the recipe-level distribution. This is how a multi-scene scenario
    YAML wires up one converged distribution *per scene* — necessary
    because each scene's CEM loop emits its own distribution file.
    """
    from falsify.cem.distribution import GaussianBoxDistribution
    cem_block = (
        (scene_entry_overrides or {}).get("cem_distribution")
        or recipe.get("cem_distribution")
    )
    if cem_block is None:
        raise SystemExit(
            "cem_distribution branch hit with no distribution path — set "
            "either recipe.cem_distribution.path or scenes[i].cem_distribution.path"
        )
    dist_path = _resolve(cem_block["path"])
    dist = GaussianBoxDistribution.load_json(dist_path)
    theta = dist.sample(rng, n=1)[0]
    start_mocap, gate_pert = _theta_to_card_fields(theta, scene_cfg)
    provenance = {
        "distribution_path": str(dist_path.relative_to(REPO_ROOT))
        if dist_path.is_relative_to(REPO_ROOT) else str(dist_path),
        "param_names":   list(dist.to_dict()["param_names"]),
        "theta":         theta.tolist(),
        "target_failure_type": dist.target_failure_type,
    }
    return start_mocap, gate_pert, provenance


def _mocap_start_to_ned(scene_yaml: Path, start_mocap: list[float]) -> list[float]:
    scene_cfg = load_yaml(scene_yaml)
    fg = build_frame_graph(scene_cfg, base_path=scene_yaml.parent)
    start_pt = Point.of(*start_mocap, fg.frame("mocap"))
    ned = fg.convert(start_pt, to="ned")
    return list(map(float, ned.xyz))


def _scene_key(scene_yaml: Path, scene_cfg: dict) -> str:
    return str(scene_cfg.get("scene_key") or scene_yaml.stem)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenario", required=True, type=Path,
                    help="Scenario YAML (e.g. configs/eval_suite/pure.yaml).")
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "runs" / "eval_bundles",
                    help="Root directory for bundle output. "
                         "Default: runs/eval_bundles/")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing trial cards (default: error out "
                         "if any card already exists, to avoid silent drift).")
    args = ap.parse_args(argv)

    scenario_path = _resolve(args.scenario)
    scenario = load_yaml(scenario_path)
    scenario_name = scenario["name"]
    n_trials = int(scenario["n_trials"])
    master_seed = int(scenario["master_seed"])
    recipe = scenario.get("recipe", {})
    start_enabled = (recipe.get("start_randomization") or {}).get("enabled", True)
    cem_mode = bool(recipe.get("cem_distribution"))
    if cem_mode and (
        (recipe.get("start_randomization") or {}).get("enabled", False)
        or (recipe.get("gate_perturbation") or {}).get("enabled", False)
    ):
        raise SystemExit(
            "recipe.cem_distribution is mutually exclusive with "
            "start_randomization / gate_perturbation — the CEM distribution "
            "owns both axes. Drop the uniform recipe blocks from the "
            "scenario YAML or unset cem_distribution."
        )
    scenes = _load_scenes(scenario)

    bundle_dir = args.out_root / scenario_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    print(f"[generate] scenario={scenario_name} n_trials={n_trials} "
          f"master_seed={master_seed} scenes={len(scenes)}")

    total_written = 0
    for entry in scenes:
        scene_cfg = load_yaml(entry.scene_yaml)
        scene_key = entry.scene_key_override or _scene_key(entry.scene_yaml, scene_cfg)
        scene_dir = bundle_dir / scene_key
        scene_dir.mkdir(parents=True, exist_ok=True)

        # Prompt registries: atomic_dataset_prompts (auto-generated from
        # data/atomic_datasets) plus an optional compositional_prompts
        # file for eval-only tasks with no training-data counterpart.
        # Compositional entries override atomic ones on key collision.
        prompts_yamls = [
            _resolve(Path("configs/prompts/atomic_dataset_prompts.yaml")),
            _resolve(Path("configs/prompts/compositional_prompts.yaml")),
        ]
        prompts: dict = {}
        for py in prompts_yamls:
            if py.is_file():
                prompts.update(load_yaml(py).get("prompts", {}))
        prompt_entry = prompts.get(entry.prompt_name)
        if prompt_entry is None:
            raise SystemExit(
                f"prompt_name {entry.prompt_name!r} not in registry "
                f"({[str(p) for p in prompts_yamls]})")
        prompt_text = prompt_entry["task"]

        entry_n_trials = entry.n_trials_override if entry.n_trials_override is not None else n_trials
        for i in range(entry_n_trials):
            seed = _seed_for(master_seed, scenario_name, scene_key, i)
            rng = np.random.default_rng(seed)
            cem_provenance: Optional[dict] = None
            scene_cem_mode = cem_mode or (entry.cem_distribution is not None)
            if scene_cem_mode:
                start_mocap, gate_pert, cem_provenance = (
                    _sample_from_cem_distribution(
                        recipe, scene_cfg, rng,
                        scene_entry_overrides={
                            "cem_distribution": entry.cem_distribution,
                        },
                    )
                )
            else:
                start_mocap = _sample_start_mocap(scene_cfg, rng, start_enabled)
                gate_pert = _sample_gate_perturbation(recipe, rng, scene_cfg=scene_cfg)
            start_ned = _mocap_start_to_ned(entry.scene_yaml, start_mocap)
            card = {
                "scenario": scenario_name,
                "scene": str(entry.scene_yaml.relative_to(REPO_ROOT)),
                "scene_key": scene_key,
                "safety": str(entry.safety_yaml.relative_to(REPO_ROOT)),
                "recovery": (
                    str(entry.recovery_yaml.relative_to(REPO_ROOT))
                    if entry.recovery_yaml is not None else None
                ),
                "prompt_name": entry.prompt_name,
                "prompt": prompt_text,
                "trial_index": i,
                "master_seed": master_seed,
                "trial_seed": int(seed),
                "start_position_mocap": start_mocap,
                "start_ned": start_ned,
                "gate_perturbation": gate_pert,    # None if not enabled
            }
            if cem_provenance is not None:
                card["cem_provenance"] = cem_provenance
            out_path = scene_dir / f"trial_{i:03d}.json"
            if out_path.exists() and not args.force:
                raise SystemExit(
                    f"refuse to overwrite existing {out_path}; pass --force "
                    f"to regenerate"
                )
            out_path.write_text(json.dumps(card, indent=2))
            total_written += 1
        print(f"  scene={scene_key}  wrote {entry_n_trials} trials to {scene_dir}")

    print(f"[generate] wrote {total_written} trial cards under {bundle_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
