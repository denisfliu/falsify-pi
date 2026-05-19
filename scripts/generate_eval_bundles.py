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


def _load_scenes(scenario: dict) -> list[SceneEntry]:
    out: list[SceneEntry] = []
    for s in scenario["scenes"]:
        out.append(SceneEntry(
            scene_yaml=_resolve(s["scene"]),
            prompt_name=s["prompt_name"],
            safety_yaml=_resolve(s["safety"]),
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


def _sample_gate_perturbation(recipe: dict, rng: np.random.Generator) -> Optional[dict]:
    gp = recipe.get("gate_perturbation") or {}
    if not gp.get("enabled", False):
        return None
    half_xyz = np.asarray(gp.get("offset_half_widths", [0.0, 0.0, 0.0]),
                          dtype=np.float64)
    half_yaw = float(gp.get("yaw_half_width_rad", 0.0))
    dxyz = rng.uniform(low=-half_xyz, high=+half_xyz, size=(3,))
    dyaw = float(rng.uniform(-half_yaw, +half_yaw))
    return {
        "name": "gate_rigid_perturbation",
        "delta_xyz": dxyz.tolist(),
        "delta_yaw_rad": dyaw,
    }


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
    scenes = _load_scenes(scenario)

    bundle_dir = args.out_root / scenario_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    print(f"[generate] scenario={scenario_name} n_trials={n_trials} "
          f"master_seed={master_seed} scenes={len(scenes)}")

    total_written = 0
    for entry in scenes:
        scene_cfg = load_yaml(entry.scene_yaml)
        scene_key = _scene_key(entry.scene_yaml, scene_cfg)
        scene_dir = bundle_dir / scene_key
        scene_dir.mkdir(parents=True, exist_ok=True)

        prompts_yaml = _resolve(Path("configs/prompts/atomic_dataset_prompts.yaml"))
        prompts = load_yaml(prompts_yaml).get("prompts", {})
        prompt_entry = prompts.get(entry.prompt_name)
        if prompt_entry is None:
            raise SystemExit(
                f"prompt_name {entry.prompt_name!r} not in registry "
                f"({prompts_yaml})")
        prompt_text = prompt_entry["task"]

        for i in range(n_trials):
            seed = _seed_for(master_seed, scenario_name, scene_key, i)
            rng = np.random.default_rng(seed)
            start_mocap = _sample_start_mocap(scene_cfg, rng, start_enabled)
            start_ned = _mocap_start_to_ned(entry.scene_yaml, start_mocap)
            gate_pert = _sample_gate_perturbation(recipe, rng)
            card = {
                "scenario": scenario_name,
                "scene": str(entry.scene_yaml.relative_to(REPO_ROOT)),
                "scene_key": scene_key,
                "safety": str(entry.safety_yaml.relative_to(REPO_ROOT)),
                "prompt_name": entry.prompt_name,
                "prompt": prompt_text,
                "trial_index": i,
                "master_seed": master_seed,
                "trial_seed": int(seed),
                "start_position_mocap": start_mocap,
                "start_ned": start_ned,
                "gate_perturbation": gate_pert,    # None if not enabled
            }
            out_path = scene_dir / f"trial_{i:03d}.json"
            if out_path.exists() and not args.force:
                raise SystemExit(
                    f"refuse to overwrite existing {out_path}; pass --force "
                    f"to regenerate"
                )
            out_path.write_text(json.dumps(card, indent=2))
            total_written += 1
        print(f"  scene={scene_key}  wrote {n_trials} trials to {scene_dir}")

    print(f"[generate] wrote {total_written} trial cards under {bundle_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
