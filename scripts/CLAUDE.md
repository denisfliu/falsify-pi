# scripts/ — CLI utilities, grouped by purpose

Scripts here are CLIs that compose existing `falsify` library code, not
re-implementations of it. Many of the most common workflows are wrapped
as **skills** under `.claude/skills/` — check `.claude/skills/README.md`
first; if a skill already covers what you want, use the skill.

The categories below are physical (subfolders) so paths are stable and
greppable. Inside each subfolder, scripts are listed in rough order of
how often they're used.

## Path convention

All scripts assume invocation **from the repo root** with `PYTHONPATH=src`:

```bash
PYTHONPATH=src python scripts/<category>/<name>.py …
```

Each script computes `REPO_ROOT = Path(__file__).resolve().parent.parent.parent`
so any `--scene`, `--config`, etc. relative paths resolve against the repo
root regardless of the shell's CWD.

## Layout

```
scripts/
├── eval/                campaign generation, execution, posthoc, plotting
├── recovery/            MPC recovery-trajectory collection + viz
├── cem/                 cross-entropy-method falsification driver
├── figures/             paper-quality scene + trajectory visualizations
├── dataset/             dataset utilities (registry, schema migration, viz)
├── action_prediction/   offline action-prediction eval (no rollout)
└── debug/               renderer + replay diagnostics
```

---

## `eval/` — evaluation campaigns

Reproducible eval pipeline over pre-sampled trial cards. Cards are
authored once per `(scenario YAML + master_seed)` so multiple policies
see byte-identical conditions. Wrapped by the **`falsify-eval-campaign`**
skill.

### Pipeline

| Script | Purpose |
|---|---|
| `generate_eval_bundles.py` | Sample N absolute trial cards from a scenario YAML into `runs/eval_bundles/<scenario>/`. Idempotent given the same YAML + master_seed. |
| `run_eval_campaign.py` | Iterate a bundle, run one episode per card under a policy, write per-trial outputs + `campaign_summary.json` under `runs/eval_campaigns/<policy_id>/run-NNN-<scenario>-<ts>/`. |
| `summarize_eval_campaign.py` | Print per-scenario / cross-policy outcome breakdowns from one or more campaign dirs. |
| `reclassify_campaign.py` | Re-run `classify_trajectory_posthoc` on captured campaigns when the classification logic changes (preserves originals as `.bak`). |
| `plot_eval_run.py` | Backfill `viz/trajectories.html` + `viz/outcome_charts.html` on a historical campaign without re-rolling — wraps `falsify.visualization.eval_report`. |

### Comparison & per-campaign plots

| Script | Purpose |
|---|---|
| `plot_campaign_compare.py` | A/B compare two campaigns in a single 3-D plot (dash encodes campaign, color encodes outcome). |
| `plot_campaign_grid.py` | N-up subplot grid for parameter sweeps (e.g. chunk size); imports the rendering helpers from `plot_campaign_compare.py` as a sibling module. |
| `plot_failures_with_recoveries.py` | Overlay failed rollouts and their `recovery_trajectory.npz` polylines per scene. |
| `plot_miss_skip_trajectories.py` | Filtered view: only MISS_GATE + SKIPPED_GATE rollouts across scenes. |
| `plot_rollout_trajectories.py` | Quick viewer for one or more `runs/vla_*/` directories (reads `vla_io/query_*/data.json`). |
| `compare_training_vs_eval.py` | Single HTML comparing one training-dataset episode against one eval-pipeline trial (images, state, action chunk, full trajectory). |

---

## `recovery/` — MPC recovery collection

Stream-sample gate perturbations, run a policy, harvest the MPC recovery
trajectories that fire when the policy fails. Output is corrective-
maneuver training data. Wrapped by the **`falsify-collect-recoveries`**
skill.

| Script | Purpose |
|---|---|
| `collect_recovery_trajectories.py` | The driver. Loops until N recovery NPZs are saved or `--max-trials` is exhausted. Writes `runs/recovery_collection/<policy_id>/<scene_key>/run-NNN-<ts>/`. |
| `render_recoveries_to_dataset.py` | Render a recovery-collection run into per-episode parquets, re-applying each trial's `GateRigidPerturbation` before render. Required — naive `export_training_data --trajectories-dir` would render against the nominal gate and silently mislabel frames. |
| `live_recovery_dashboard.py` | Self-refreshing HTML dashboard (3-D rollouts + progress counters) for in-flight collection runs. Polls the newest `run-*` under each watched `(policy, scene)` pair. |
| `viewer_with_recoveries.py` | Launch nerfstudio's ns-viewer with rollout + recovery polylines overlaid as viser scene primitives — useful for finding a camera angle for a screenshot. |

---

## `cem/` — cross-entropy method falsification

| Script | Purpose |
|---|---|
| `run_cem_campaign.py` | Per-scene CEM loop: sample → write cards → invoke `eval/run_eval_campaign.py` as a subprocess → score via `falsify.cem.scorer` → refit. Emits `converged_distribution.json` that an exploit scenario YAML can load via `recipe.cem_distribution.path`. |

---

## `figures/` — paper visualizations

Scene + trajectory rendering for figures. The point-cloud path is
CPU-only (sidesteps the gsplat CUDA JIT); the `_splat` variants require
a working `tools/env.sh` for photorealistic backdrops.

### Scene rendering

| Script | Purpose |
|---|---|
| `render_scene_pointclouds.py` | Per-scene colored point-cloud PNG (gaussian means + DC SH → RGB, scene_edits applied, optional `--exclude-points-dir` cleanup). CPU-only. |
| `render_scene_overview.py` | Overhead + start-position-front photoreal renders through `GSplatRenderer` — visually confirm `scene_edits` (incl. DuplicateAABB) hit the real gsplat. |

### Composed figures

| Script | Purpose |
|---|---|
| `figure_failure_recovery.py` | "VLA fails, recovery MPC replans" panel with a faint point-cloud backdrop. Inputs: trial dir / run dir / scene_key dir + scene point-cloud cache. |
| `figure_failure_recovery_splat.py` | Same panel, but the backdrop is a photoreal gsplat render projected to 2-D via the OpenCV-pinhole pose used for the render. |

### Exclude-mask authoring (used by `render_scene_pointclouds.py`)

| Script | Purpose |
|---|---|
| `paint_exclude_aabbs.py` | Dash app — interactively paint exclude AABBs against the MOCAP-frame gaussian means cloud. Writes a `boxes + exclude_points_mocap + source_gate` JSON. |
| `transform_exclude_points.py` | Transport painted points from one gate's frame to all other scenes' gate frames (rotate about source anchor by source→target normal angle, then translate). Point-level avoids the AABB re-bracketing inflation that rotating boxes would cause. Output is consumed by `render_scene_pointclouds.py --exclude-points-dir`. |

---

## `dataset/` — dataset utilities

| Script | Purpose |
|---|---|
| `build_prompt_registry.py` | Walk every `data/atomic_datasets/<name>/meta/tasks.jsonl`, dedupe identical task strings across datasets, emit `configs/prompts/atomic_dataset_prompts.yaml`. `run_vla_episode` and other CLIs reject prompts not in this registry, so refresh after adding a dataset. |
| `convert_no_3pov_to_v3.py` | Schema migration — convert a LeRobot v2.1 dataset to the v3.0 layout SousVide's `build_recovery_dataset.py` consumes (column renames, type widens, meta restructure). |
| `strip_3pov.py` | Drop the `3pov_1` column from a v2.1 dataset (and its `info.json` / `episodes_stats.jsonl` references). Composes with `convert_no_3pov_to_v3.py`. |
| `synth_episode_to_npz_and_viz.py` | Pick a random episode from a synth atomic dataset, emit a **MOCAP-frame** NPZ + plotly viz overlaid on the scene's point cloud. The NPZ is intentionally `positions_mocap` / `yaws_mocap` so it cannot be silently fed to `falsify-export-parquet` (which is NED-only by contract). |

---

## `action_prediction/` — offline action-prediction eval

No rollout — feed dataset frames through the policy and compare its first
predicted action against the dataset's ground truth. "MPC style" (one
infer per frame).

| Script | Purpose |
|---|---|
| `eval_action_prediction.py` | Iterate a v3 parquet dataset frame-by-frame, query the policy gateway, save `per_step.npz` + `summary.json` under `runs/action_prediction/<name>/`. Bypasses `PiGatewayPolicy`'s NED↔MOCAP boundary because the dataset state is already in MOCAP. |
| `plot_action_prediction.py` | Render the result NPZ as one HTML — per-dim pred-vs-gt scatter, error trace, error histogram. |

---

## `debug/` — renderer + replay diagnostics

| Script | Purpose |
|---|---|
| `debug_render_at_pose.py` | Render the forward camera at the drone's initial pose, dump `Tw2g`, the NED → NS pose chain, and the NS-frame gate AABB so you can tell whether a gray render is a pose-chain bug or a model bug. Wrapped by `falsify-debug-render`. |
| `replay_renders.py` | Re-render a previous run's flown trajectory (forward + downward cameras) into PNGs + flythrough MP4s. Mirrors the simulator's chunked-rollout indexing so the viewpoints match what the rollout actually visited. |

---

## When to add vs. when to skill-ify

If you find yourself reaching for a one-off script, ask: **is this
chained from other things often enough to deserve a skill?** If yes,
write the script in the appropriate subfolder *and* add a SKILL.md under
`.claude/skills/` that documents how to chain it.

## One-off scripts must be clearly demarcated

A "one-off" here means a script with hard-coded paths, hard-coded
episode/dataset names, a hard-coded recipe, or a single-use schema
migration — anything that won't reasonably be re-run as-is. The default
should be **don't add one** — do it inline in a notebook and delete
after. We pruned 8 such scripts out of this folder once already
(`build_combined_datasets.py`, `relabel_combined_tasks.py`,
`verify_check_parquets.py`, etc.) and they had all silently rotted.

If you must commit a one-off:

1. **Prefix the filename with `oneoff_`** — e.g. `oneoff_relabel_v3_tasks.py`.
2. **Top of the docstring**: a `ONE-OFF (YYYY-MM-DD):` line stating
   what it ran against and why it isn't reusable. Example:

   ```python
   """ONE-OFF (2026-05-25): re-label task_index across the 4 derived
   gate_scenes_* datasets. Hard-coded to the alphabetical-merge order
   combine_lerobot used at that snapshot; not parameterized."""
   ```

3. **Add it to the appropriate subfolder's section below** under a
   `### One-offs (kept for provenance)` subheading, with the date and
   the one-line reason it still exists.

If a script is genuinely reusable, do not prefix it. The `oneoff_`
prefix is the marker that lets the next cleanup pass delete confidently
without re-litigating each file.
