---
name: falsify-collect-recoveries
description: Stream-sample gate perturbations and harvest MPC recovery trajectories for one (policy, scene) pair until N NPZs are saved. Each recovery is a canonical Trajectory NPZ (NED), ready to hand straight to `falsify-export-parquet` or `falsify-orchestrate-batch`. Use when building corrective-maneuver training data from a deployed VLA: failures of the policy under perturbation become labeled "I'm off; recover" demonstrations.
---

# falsify-collect-recoveries

A streaming generator for **corrective-maneuver training data**. We
roll a VLA against perturbed scenes; whenever the policy fails, the
`CoursedMpcPlanner` produces a recovery trajectory from the
last-safe-state through the gate to the goal — and that's the
labeled "I'm off; recover" demonstration the next training run will
consume.

Sibling of [`falsify-trajectory-from-vla`](../falsify-trajectory-from-vla/SKILL.md)
(one rollout → one trajectory) and
[`falsify-perturb-course`](../falsify-perturb-course/SKILL.md)
(perturb a *waypoint* in a Course YAML to fabricate recoveries). This
skill is for **policy-driven, on-the-fly** generation: the perturbation
sampling is closed-loop with the policy's actual behaviour, so the
distribution of recoveries reflects the policy's real failure modes
rather than a hand-authored set.

## When to use

- The policy is mature enough to succeed most of the time but fails
  in known ways under small perturbations, and we want corrective
  demos for the next finetune.
- We have a recovery YAML wired (`configs/recovery/<scene>_mpc.yaml`)
  with a Course YAML that threads the gate.
- We don't want to manually author N hand-picked failure conditions —
  let the sampler explore the perturbation envelope.

Not for:

- Eval (use [`falsify-eval-campaign`](../falsify-eval-campaign/SKILL.md) —
  same machinery, fixed-cards-then-stop instead of stream-until-N).
- Recoveries from a hand-authored course (use
  [`falsify-perturb-course`](../falsify-perturb-course/SKILL.md) +
  [`falsify-trajectory-from-waypoints`](../falsify-trajectory-from-waypoints/SKILL.md)).

## Inputs

| Flag | Meaning |
|---|---|
| `--policy-config` | `configs/policies/pi_gateway/<x>.yaml` — the VLA we're falsifying. Bridge handshake fires once on first rollout. |
| `--scene` / `--safety` / `--recovery` | The scene + safety stack + recovery YAML (must be MPC-backed and have non-empty `trigger_failure_types`). |
| `--frame` | Drone-frame YAML — `configs/frames/carl_dual.yaml` for the standard dual-cam build. |
| `--perturbation-recipe` | Scenario YAML whose `recipe.gate_perturbation` block (must be `enabled: true`) defines the Δxyz/Δyaw bounds. The scenario's scenes/safety/recovery fields are ignored; only the recipe is consumed. Common choice: `configs/eval_suite/gate_perturbed_small.yaml` (±3 cm / ±3°). |
| `--prompt-name` | Same registry key the bundles use — resolved against `configs/prompts/atomic_dataset_prompts.yaml` + `configs/prompts/compositional_prompts.yaml`. |
| `--n-recoveries` | Target count (default 50). Loop exits when this many NPZs are saved. |
| `--max-trials` | Safety cap (default 500). Loop exits early if we don't hit the target. |
| `--collection-seed` | Master seed for the per-trial seed_for hash. Default 100 000 — well clear of eval bundles' `master_seed=0`, so cards drawn here are statistically independent of any eval card. |

## Output layout — this is the contract

```
runs/recovery_collection/<policy_id>/<scene_key>/run-NNN-<YYYYMMDD_HHMMSS>/
├── collection_manifest.json    # CLI argv, recipe sha, seed, target, final stats (n_sampled, n_recovered, by_outcome)
├── policy_manifest.json        # mirrors eval campaigns: YAML sha, bridge id, traceability
├── collection.log              # teed stdout/stderr from the loop
├── recoveries/                 # the deliverable
│   ├── recovery_000.npz        # canonical Trajectory NPZ — ready for export_training_data --trajectory
│   ├── recovery_001.npz
│   └── ...                     # numbered by collection order
├── <scene_key>/                # per-trial detail (mirrors eval-campaign layout so the viz emitter Just Works)
│   └── trial_NNN/
│       ├── trial_card.json
│       ├── episode_summary.json
│       ├── rollout_states.npz
│       └── recovery_trajectory.npz   # copy of recoveries/recovery_KKK.npz (only on failed trials)
└── viz/
    └── trajectories.html       # rollouts + recovery polylines overlaid on the scene
```

`run-NNN` auto-increments per `(policy, scene)` pair via the same scan
the eval pipeline uses (`max(existing run-*) + 1`).

## How to run

### Step 1 — single scene

```bash
bash -c 'export PI_API_KEY=...; source tools/env.sh; \
    PYTHONPATH=src python scripts/recovery/collect_recovery_trajectories.py \
        --policy-config configs/policies/pi_gateway/nonhistory_real_synth_31ohxgxv_5000.yaml \
        --scene         configs/scenes/left_gate.yaml \
        --safety        configs/safety/left_gate.yaml \
        --recovery      configs/recovery/left_gate_mpc.yaml \
        --frame         configs/frames/carl_dual.yaml \
        --perturbation-recipe configs/eval_suite/gate_perturbed_small.yaml \
        --prompt-name   left_gate \
        --n-recoveries  50'
```

### Step 2 — multi-scene driver

`tools/collect_recoveries_real_synth_small.sh` loops over both gate
scenes, sharing the bridge session:

```bash
export PI_API_KEY="pi-jt-moraband-dev-001"
source tools/env.sh
bash tools/collect_recoveries_real_synth_small.sh
```

Tunables via env: `N_RECOVERIES`, `MAX_TRIALS`, `COLLECTION_SEED`.
Edit the file directly to change the policy or the scene list.

### Step 3 — render parquets from the harvested NPZs

Hand the `recoveries/` dir straight to the parquet exporter:

```bash
PYTHONPATH=src python -m falsify.cli.export_training_data \
    --trajectories-dir runs/recovery_collection/<policy>/<scene>/run-NNN-<ts>/recoveries \
    --scene configs/scenes/<scene>.yaml \
    --frame configs/frames/carl_dual.yaml \
    --embodiment configs/embodiments/carl_dual_mocap.yaml \
    --out runs/recovery_parquets/<scene>
```

Or use [`falsify-orchestrate-batch`](../falsify-orchestrate-batch/SKILL.md)
to bulk-render across multiple scenes while reusing one renderer per
scene.

## How it works (under the hood)

The loop reuses every piece of the eval pipeline; only the driver
shape changes. Per iteration:

1. Sample one card with `falsify.eval.sampling.sample_start_mocap` +
   `sample_gate_perturbation`. Seed = `seed_for(collection_seed,
   "recovery_collection", scene_key, trial_idx)`.
2. Construct a fresh `PiGatewayPolicy` (so the bridge sees a clean
   reset), wire the `CoursedMpcPlanner` with the trial's `gate_deltas`
   so the MPC plans through the *perturbed* gate, and run via
   `falsify.orchestrator.run_episode`.
3. The shared `GSplatRenderer` + detector are reused across all trials
   — only built once per scene, so the 30 s gsplat load is paid once
   per script invocation, not per trial.
4. If `episode.recovery_trajectory is not None`, the trajectory is
   saved with `save_trajectory(…, source="recovery")` — both to the
   harvest `recoveries/recovery_NNN.npz` (numbered by collection order)
   and as a copy inside the trial dir.
5. The trial's `episode_summary.json` is written in the same schema
   the eval campaigns use (`posthoc_outcome`, `gate_aabb_mocap`,
   directional crossing fields when applicable) so the existing viz
   emitter handles the result without modification.

## Hard rules

1. **`source="recovery"` on every NPZ.** The parquet exporter keys
   off this field for chunking decisions; don't override.
2. **One (policy, scene) per invocation.** Two reasons: lets the
   renderer cache, and lets a left_gate failure not abort right_gate.
   The shell driver loops.
3. **`recoveries/` is the handoff dir.** Don't add anything else
   there — `--trajectories-dir <recoveries>` is the contract with
   downstream skills, and they discover `*.npz` blind.
4. **`collection_seed` lives outside the eval-bundle namespace.**
   Default 100 000; eval bundles use 0. Sharing the namespace would
   yield colliding seeds and break determinism for both.

## Reference docs

- `scripts/recovery/collect_recovery_trajectories.py` — the driver.
- `src/falsify/eval/sampling.py` — the shared per-trial sampling
  helpers (`sample_gate_perturbation`, `sample_start_mocap`,
  `seed_for`).
- `src/falsify/recovery/CLAUDE.md` — `CoursedMpcPlanner` contract +
  replanning-seed sampling.
- `configs/recovery/<scene>_mpc.yaml` — the per-scene recovery
  triggers + course path.
- [`falsify-eval-campaign`](../falsify-eval-campaign/SKILL.md) —
  fixed-bundle eval driver that shares the per-trial wiring this
  collector inverts.
- [`falsify-export-parquet`](../falsify-export-parquet/SKILL.md) —
  what to do with the harvested NPZs.
