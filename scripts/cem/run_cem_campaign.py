"""CEM optimization loop — converges a distribution that produces more
failures of a chosen type than uniform sampling does.

Each iteration:

1. Sample N θ vectors from the current distribution.
2. Write them as trial cards into ``<out>/<scene_key>/iter_NNN/bundle/``.
3. Invoke ``scripts/eval/run_eval_campaign.py`` as a subprocess against that
   bundle — produces per-trial ``episode_summary.json`` + ``rollout_states.npz``.
4. Score every trial via ``falsify.cem.scorer`` (continuous cost for the
   target failure type).
5. Pick top-K elites by cost and refit the distribution.

After ``n_iters`` iterations the final ``converged_distribution.json`` is
written under ``<out>/<scene_key>/converged_distribution.json``. The
eval-suite scenario YAML ``configs/eval_suite/cem_<...>_exploit.yaml``
loads this file and samples from it, producing trial cards that
``run_eval_campaign.py`` runs unchanged — that's the A/B test against the
uniform baseline.

The optimization is **per scene** — the loop runs once for each scene in
the CEM config, each producing its own ``converged_distribution.json``.

Usage::

    bash -c 'export PI_API_KEY=...; source tools/env.sh; \\
        source tools/pi_inference_env.sh; \\
        PYTHONPATH=src python scripts/cem/run_cem_campaign.py \\
            --cem-config configs/cem/collision_gate.yaml \\
            --policy-config configs/policies/pi_gateway/history_h6jtbq0w_20k.yaml \\
            --frame configs/frames/carl_dual.yaml \\
            --out runs/cem_campaigns/collision_gate_v0'
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve(p: str | Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (REPO_ROOT / pp).resolve()


@dataclass
class SceneEntry:
    scene_yaml: Path
    prompt_name: str
    safety_yaml: Path
    scene_key: str
    recovery_yaml: Optional[Path] = None


def _load_scenes(cem_cfg: dict) -> list[SceneEntry]:
    out: list[SceneEntry] = []
    for s in cem_cfg["scenes"]:
        scene_yaml = _resolve(s["scene"])
        # Determine scene_key — prefer override, fall back to the YAML's
        # `scene_key` field, then to the YAML stem.
        from falsify.io import load_yaml
        scene_cfg = load_yaml(scene_yaml)
        scene_key = (
            s.get("scene_key_override")
            or scene_cfg.get("scene_key")
            or scene_yaml.stem
        )
        out.append(SceneEntry(
            scene_yaml=scene_yaml,
            prompt_name=s["prompt_name"],
            safety_yaml=_resolve(s["safety"]),
            scene_key=str(scene_key),
            recovery_yaml=_resolve(s["recovery"]) if s.get("recovery") else None,
        ))
    return out


def _write_iter_scenario(
    *,
    out_path: Path,
    name: str,
    n_samples: int,
    distribution_path: Path,
    scene_entry: SceneEntry,
) -> None:
    """Author a stub scenario YAML so run_eval_campaign.py is happy.

    The bundle dir is passed explicitly via ``--bundle-dir`` so this YAML
    is consulted only for ``scenario["name"]`` and the per-scene
    references.
    """
    import yaml
    body = {
        "name": name,
        "description": f"CEM iter scenario (auto-generated)",
        "n_trials": int(n_samples),
        "master_seed": 0,
        "recipe": {
            "cem_distribution": {
                "path": str(distribution_path.relative_to(REPO_ROOT))
                if distribution_path.is_relative_to(REPO_ROOT)
                else str(distribution_path),
            },
        },
        "scenes": [{
            "scene":  str(scene_entry.scene_yaml.relative_to(REPO_ROOT))
                      if scene_entry.scene_yaml.is_relative_to(REPO_ROOT)
                      else str(scene_entry.scene_yaml),
            "prompt_name": scene_entry.prompt_name,
            "safety": str(scene_entry.safety_yaml.relative_to(REPO_ROOT))
                      if scene_entry.safety_yaml.is_relative_to(REPO_ROOT)
                      else str(scene_entry.safety_yaml),
            "scene_key_override": scene_entry.scene_key,
        }],
    }
    if scene_entry.recovery_yaml is not None:
        body["scenes"][0]["recovery"] = (
            str(scene_entry.recovery_yaml.relative_to(REPO_ROOT))
            if scene_entry.recovery_yaml.is_relative_to(REPO_ROOT)
            else str(scene_entry.recovery_yaml)
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(body, sort_keys=False))


def _run_cem_for_scene(
    *,
    cem_cfg: dict,
    scene_entry: SceneEntry,
    policy_config: Path,
    frame_yaml: Path,
    out_dir: Path,
    n_iters: int,
    n_samples: int,
    elite_frac: float,
    cov_shrink: float,
    base_seed: int,
    skip_flythrough: bool,
    no_rtc: bool,
) -> None:
    """Run the full per-scene CEM optimization loop."""
    from falsify.cem import GaussianBoxDistribution
    from falsify.cem.sampler import write_cem_trial_cards
    from falsify.cem.scorer import SceneContext, score_trial
    from falsify.io import load_yaml

    target_failure_type = cem_cfg["target_failure_type"]
    bounds_layout = cem_cfg["bounds"]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[cem] === scene={scene_entry.scene_key} ===")
    print(f"[cem]   target={target_failure_type}  out={out_dir}")
    print(f"[cem]   n_iters={n_iters}  n_samples={n_samples}  "
          f"elite_frac={elite_frac}  cov_shrink={cov_shrink}")

    # Prompt resolution (one-time).
    prompts = load_yaml(_resolve(Path("configs/prompts/atomic_dataset_prompts.yaml")))
    prompts_block = prompts.get("prompts", {})
    if scene_entry.prompt_name not in prompts_block:
        raise SystemExit(
            f"prompt_name {scene_entry.prompt_name!r} not in registry"
        )
    prompt_text = prompts_block[scene_entry.prompt_name]["task"]

    # Pre-build the scoring context (one PLY/YAML load per scene).
    ctx = SceneContext.from_yamls(scene_entry.scene_yaml, scene_entry.safety_yaml)

    # Iteration 0 distribution = uniform-equivalent prior.
    dist = GaussianBoxDistribution.uniform_prior(
        bounds_layout,
        target_failure_type=target_failure_type,
        provenance={
            "cem_config_name": cem_cfg.get("name"),
            "scene_key":       scene_entry.scene_key,
            "prompt_name":     scene_entry.prompt_name,
            "policy_config":   str(policy_config),
            "git_sha":         _git_sha(),
            "iter_index":      0,
            "n_samples":       n_samples,
            "elite_frac":      elite_frac,
            "cov_shrink":      cov_shrink,
        },
    )
    # Per-scene RNG — different scene_keys diverge after iter-0.
    rng = np.random.default_rng(
        abs(hash((int(base_seed), scene_entry.scene_key))) % (2 ** 32)
    )

    iter_summaries: list[dict] = []
    for it in range(n_iters):
        iter_dir = out_dir / f"iter_{it:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        dist_path = iter_dir / "distribution.json"
        dist.provenance["iter_index"] = int(it)
        dist.save_json(dist_path)

        bundle_dir_iter = iter_dir / "bundle"
        scene_bundle_dir = bundle_dir_iter / scene_entry.scene_key
        scenario_name = f"{cem_cfg.get('name', 'cem')}_iter_{it:03d}_{scene_entry.scene_key}"
        print(f"\n[cem] iter {it}/{n_iters - 1}  scene={scene_entry.scene_key}")
        print(f"[cem]   sampling {n_samples} cards → {scene_bundle_dir}")
        write_cem_trial_cards(
            distribution=dist,
            rng=rng,
            n_samples=n_samples,
            scenario_name=scenario_name,
            scene_yaml=scene_entry.scene_yaml,
            scene_key=scene_entry.scene_key,
            safety_yaml=scene_entry.safety_yaml,
            prompt_name=scene_entry.prompt_name,
            prompt_text=prompt_text,
            master_seed=int(base_seed),
            iter_index=int(it),
            distribution_path=dist_path,
            repo_root=REPO_ROOT,
            out_dir=scene_bundle_dir,
            recovery_yaml=scene_entry.recovery_yaml,
        )

        # Stub scenario YAML so the campaign runner can parse it.
        scenario_yaml_path = iter_dir / "iter_scenario.yaml"
        _write_iter_scenario(
            out_path=scenario_yaml_path,
            name=scenario_name,
            n_samples=n_samples,
            distribution_path=dist_path,
            scene_entry=scene_entry,
        )

        # Run the eval campaign for this iter.
        campaign_out = iter_dir / "campaign"
        cmd = [
            sys.executable, str(REPO_ROOT / "scripts" / "eval" / "run_eval_campaign.py"),
            "--scenario",      str(scenario_yaml_path),
            "--bundle-dir",    str(bundle_dir_iter),
            "--policy-config", str(policy_config),
            "--frame",         str(frame_yaml),
            "--out",           str(campaign_out),
        ]
        if skip_flythrough:
            cmd.append("--skip-flythrough")
        if no_rtc:
            cmd.append("--no-rtc")
        print(f"[cem]   → run_eval_campaign.py ({n_samples} trials)")
        t0 = time.time()
        subprocess.run(cmd, check=True)
        print(f"[cem]   ← campaign elapsed {time.time() - t0:.1f}s")

        # Score every trial.
        scored: list[dict] = []
        scene_trial_dir = campaign_out / scene_entry.scene_key
        for trial_card_path in sorted(scene_bundle_dir.glob("trial_*.json")):
            card = json.loads(trial_card_path.read_text())
            trial_index = int(card["trial_index"])
            theta = card["cem_provenance"]["theta"]
            gate_pert = card.get("gate_perturbation")
            trial_run_dir = scene_trial_dir / f"trial_{trial_index:03d}"
            rollout_npz = trial_run_dir / "rollout_states.npz"
            summary_path = trial_run_dir / "episode_summary.json"
            if not rollout_npz.is_file():
                # The eval campaign errored on this trial (error.txt
                # written instead of rollout_states.npz). Skip — these
                # trials don't contribute to the elite set.
                print(f"[cem]   trial {trial_index}: no rollout_states.npz "
                      f"(error?); skipping")
                continue
            result = score_trial(
                rollout_npz=rollout_npz,
                ctx=ctx,
                target_failure_type=target_failure_type,
                gate_deltas=gate_pert,
            )
            summary = (
                json.loads(summary_path.read_text())
                if summary_path.is_file() else {}
            )
            scored.append({
                "trial_index": trial_index,
                "theta":       list(theta),
                "cost":        float(result["cost"]),
                "actual_failure_type": result.get("actual_failure_type"),
                "target_matched":      result.get("target_matched"),
                "diagnostics":         {
                    k: v for k, v in result.items()
                    if k not in {"cost", "target_failure_type",
                                 "actual_failure_type", "target_matched"}
                },
                "episode_failure": summary.get("failure"),
            })

        if not scored:
            raise SystemExit(
                f"iter {it}: no successful trials to score — aborting"
            )

        scored.sort(key=lambda s: -s["cost"])   # high cost = more failure-like
        n_elites = max(1, int(round(elite_frac * len(scored))))
        elites = scored[:n_elites]
        elite_thetas = np.array([e["theta"] for e in elites], dtype=np.float64)

        (iter_dir / "scores.json").write_text(json.dumps(scored, indent=2))
        (iter_dir / "elites.json").write_text(json.dumps(elites, indent=2))

        # Iter summary line — easy to eyeball convergence.
        n_target = sum(1 for s in scored if s.get("target_matched"))
        costs = np.array([s["cost"] for s in scored])
        # Replace any non-finite cost with NaN for the diagnostic line.
        finite_costs = costs[np.isfinite(costs)]
        mean_cost = float(finite_costs.mean()) if finite_costs.size else float("nan")
        elite_costs = np.array([e["cost"] for e in elites])
        elite_mean = float(elite_costs[np.isfinite(elite_costs)].mean()) \
            if np.any(np.isfinite(elite_costs)) else float("nan")
        print(f"[cem]   scored: n={len(scored)}  target={target_failure_type}: "
              f"{n_target}/{len(scored)} hit; "
              f"cost mean={mean_cost:.4f}  elite mean={elite_mean:.4f}")
        iter_summaries.append({
            "iter":              it,
            "n_scored":          len(scored),
            "n_target_matched":  n_target,
            "mean_cost_all":     mean_cost,
            "mean_cost_elite":   elite_mean,
            "distribution_mean": dist.mean.tolist(),
        })

        # Refit for the next iteration.
        dist = dist.refit(elite_thetas, cov_shrink=cov_shrink)
        # The refit() returned a new distribution; update provenance for
        # the next iter's save.
        dist.provenance = dict(dist.provenance)
        dist.provenance["refit_from_iter"] = int(it)
        dist.provenance["n_elites"] = int(n_elites)

    # Final converged distribution.
    converged_path = out_dir / "converged_distribution.json"
    dist.provenance["final"] = True
    dist.provenance["n_iters"] = int(n_iters)
    dist.save_json(converged_path)
    (out_dir / "iter_summaries.json").write_text(json.dumps(iter_summaries, indent=2))
    print(f"\n[cem] scene={scene_entry.scene_key}: wrote {converged_path}")
    print(f"[cem] mean evolution:")
    for s in iter_summaries:
        print(f"  iter {s['iter']}: n_target={s['n_target_matched']}/{s['n_scored']}  "
              f"mean_cost={s['mean_cost_all']:.4f}  elite={s['mean_cost_elite']:.4f}  "
              f"μ={[round(x, 3) for x in s['distribution_mean']]}")


def _git_sha() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        )
        return r.stdout.strip()
    except Exception:
        return "unknown"


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cem-config", required=True, type=Path)
    ap.add_argument("--policy-config", required=True, type=Path)
    ap.add_argument("--frame", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-iters", type=int, default=None,
                    help="Override cem.n_iters from the YAML.")
    ap.add_argument("--n-samples", type=int, default=None,
                    help="Override cem.n_samples_per_iter from the YAML.")
    ap.add_argument("--scenes", nargs="+", default=None,
                    help="Optional scene_key filter to run a subset.")
    ap.add_argument("--skip-flythrough", action="store_true", default=True,
                    help="Don't render flythrough mp4s — saves ~30 s/trial "
                         "and we don't need them for scoring.")
    ap.add_argument("--no-rtc", action="store_true", default=True,
                    help="Pass --no-rtc to run_eval_campaign.py — ~22× faster "
                         "by querying the VLA once per chunk. CEM operates on "
                         "the chunk-execution policy by default; flip this off "
                         "if you want byte-identical rollouts to a "
                         "sample_actions_fixed_noise checkpoint.")
    args = ap.parse_args(argv)

    cem_cfg_path = _resolve(args.cem_config)
    import yaml
    cem_cfg = yaml.safe_load(cem_cfg_path.read_text())
    cem_meta = cem_cfg.get("cem", {})
    n_iters = int(args.n_iters if args.n_iters is not None
                  else cem_meta.get("n_iters", 8))
    n_samples = int(args.n_samples if args.n_samples is not None
                    else cem_meta.get("n_samples_per_iter", 30))
    elite_frac = float(cem_meta.get("elite_frac", 0.25))
    cov_shrink = float(cem_meta.get("cov_shrink", 0.1))
    base_seed = int(cem_meta.get("seed", 0))

    args.out.mkdir(parents=True, exist_ok=True)
    # Freeze the input config for traceability.
    (args.out / "cem_config.yaml").write_text(cem_cfg_path.read_text())

    scenes = _load_scenes(cem_cfg)
    if args.scenes:
        scenes = [s for s in scenes if s.scene_key in set(args.scenes)]
        if not scenes:
            raise SystemExit(f"no scenes matched filter {args.scenes}")

    policy_config = _resolve(args.policy_config)
    frame_yaml = _resolve(args.frame)

    t0 = time.time()
    for scene in scenes:
        _run_cem_for_scene(
            cem_cfg=cem_cfg,
            scene_entry=scene,
            policy_config=policy_config,
            frame_yaml=frame_yaml,
            out_dir=args.out / scene.scene_key,
            n_iters=n_iters,
            n_samples=n_samples,
            elite_frac=elite_frac,
            cov_shrink=cov_shrink,
            base_seed=base_seed,
            skip_flythrough=args.skip_flythrough,
            no_rtc=args.no_rtc,
        )
    print(f"\n[cem] all scenes done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
