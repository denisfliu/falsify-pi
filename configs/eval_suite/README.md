# `configs/eval_suite/` — evaluation scenarios

Declarative scenario definitions consumed by the evaluation framework
(scripts/eval/generate_eval_bundles.py + scripts/eval/run_eval_campaign.py +
scripts/eval/summarize_eval_campaign.py).

## Pipeline

```
scenario YAML    →    trial cards (JSON)        →    per-trial outputs    →    campaign summary + viz
configs/eval_suite/  runs/eval_bundles/<name>/        runs/eval_campaigns/<policy_id>/<group>/run-NNN-<scenario>-<ts>/
                                                          <scene>/trial_NNN/                  (per-trial)
                                                          campaign_summary.json
                                                          policy_manifest.json
                                                          run_manifest.json
                                                          campaign.log
                                                          viz/trajectories.html
                                                          viz/outcome_charts.html
```

Trial cards are **absolute** descriptions of each trial (start position
in mocap+ned, gate-perturbation deltas in mocap). Two policies hitting
the same card see byte-identical conditions, regardless of any later
refactor to the random samplers.

Campaign output dirs are **rooted per policy**: ``<policy_id>`` is the
stem of the ``--policy-config`` YAML (e.g.
``nonhistory_ccvhs1do_20k``). Inside each policy folder, runs are
grouped under either:

- ``sweep-NNN-<YYYYMMDD_HHMMSS>[-<tag>]/`` — produced by
  ``tools/run_eval_sweep.sh``. ``NNN`` is shared across all policies in
  the same launch so cohort-paired comparisons line up by folder name.
  Carries a ``sweep_manifest.json``.
- ``adhoc/`` — produced by a one-off ``scripts/eval/run_eval_campaign.py``
  invocation without ``--out``.

Inside the grouping folder each campaign is ``run-NNN-<scenario>-<ts>/``
where ``NNN`` auto-increments within that grouping folder.

## Scenarios shipped

| File | Scenes | Perturbations |
|---|---|---|
| `pure.yaml` | left_gate, right_gate, center_gate split into `center_gate_from_left` + `center_gate_from_right` (10 trials each, same scene YAML / two prompts) | start jitter only |
| `gate_perturbed_small.yaml` | left_gate, right_gate | start jitter + GateRigidPerturbation (±3 cm, ±3°) |
| `gate_perturbed_large.yaml` | left_gate, right_gate | start jitter + GateRigidPerturbation (±10 cm, ±10°) |
| `compositional.yaml` (Phase 3) | left_and_center, right_and_center | two gates, combined prompt |

## How to run

```bash
# 1. Generate trial bundles (idempotent; commit them to lock the run).
PYTHONPATH=src python scripts/eval/generate_eval_bundles.py \
    --scenario configs/eval_suite/pure.yaml

# 2a. Run a single ad-hoc campaign against a policy (Pi-gateway only).
#     --out is optional; when omitted the script writes to
#     runs/eval_campaigns/<policy_id>/adhoc/run-NNN-<scenario>-<ts>/ and
#     emits viz/trajectories.html + viz/outcome_charts.html inside it.
#     Add --no-viz to skip the HTML reports.
bash -c 'export PI_API_KEY=...; source tools/env.sh; \
    source tools/pi_inference_env.sh; \
    PYTHONPATH=src python scripts/eval/run_eval_campaign.py \
        --scenario configs/eval_suite/pure.yaml \
        --policy-config configs/policies/pi_gateway/history_h6jtbq0w_20k.yaml \
        --frame configs/frames/carl_dual.yaml'

# 2b. Run a full sweep (policies × scenarios). All runs in one launch
#     land under a shared sweep-NNN-<ts>/ folder per policy so cohort-
#     paired comparisons are by-folder. Edit the POLICIES + SCENARIOS
#     arrays at the top of tools/run_eval_sweep.sh to change the cohort.
bash tools/run_eval_sweep.sh --tag goal-fix   # --tag is optional

# 3. Summarise (one or more campaigns side-by-side).
PYTHONPATH=src python scripts/eval/summarize_eval_campaign.py \
    runs/eval_campaigns/history_h6jtbq0w_20k/sweep-001-*/run-001-pure-* \
    runs/eval_campaigns/nonhistory_ccvhs1do_20k/sweep-001-*/run-001-pure-*

# 4. Backfill / re-render viz on an existing campaign dir (no GPU,
#    no rollout — pure consumer of the on-disk artifacts).
PYTHONPATH=src python scripts/eval/plot_eval_run.py \
    runs/eval_campaigns/<policy_id>/run-NNN-<scenario>-<ts>
```

## Recovery toggle

Per-trial recovery-trajectory planning is opt-in per scenario YAML (via
each scene entry's `recovery: configs/recovery/<name>.yaml`). The
campaign runner also exposes CLI overrides so you don't have to
regenerate bundles when you want a fast eval-only sweep:

```bash
# (default) — recovery fires when cards have a recovery YAML, skips otherwise
PYTHONPATH=src python scripts/eval/run_eval_campaign.py --scenario ... --out ...

# Disable recovery globally, even for cards that carry a recovery YAML.
# Failed trials still record the failure but skip the MPC plan + NPZ save.
# Per-trial log line shows `recovery=off`; per-trial summary records
# `recovery.fired: false, reason: "disabled by --no-recovery"`.
PYTHONPATH=src python scripts/eval/run_eval_campaign.py --scenario ... --out ... \
    --no-recovery

# Force recovery for every failed trial, using a fallback YAML when a
# card's `recovery:` field is null (e.g. compositional cards).
PYTHONPATH=src python scripts/eval/run_eval_campaign.py --scenario ... --out ... \
    --force-recovery \
    --recovery-yaml-default configs/recovery/left_gate_mpc.yaml
```

`--no-recovery` and `--force-recovery` are mutually exclusive. At
campaign launch the script logs the active mode plus how many of the
loaded cards declare a recovery YAML — e.g.
`recovery mode='off'  cards-with-recovery=60/60`.

## Scenario YAML schema

```yaml
name: <slug>               # used as bundle dir name + campaign tag
description: <text>
n_trials: 20               # per scene
master_seed: 0             # hashed with (scenario, scene, trial_index)
recipe:
  start_randomization:
    enabled: true          # uses scene's start_randomization.half_widths_mocap
  gate_perturbation:
    enabled: false         # set true to add a GateRigidPerturbation per trial
    offset_half_widths: [hx, hy, hz]   # required if enabled (mocap meters)
    yaw_half_width_rad: theta          # required if enabled (radians)
scenes:
  - scene:       configs/scenes/<scene>.yaml
    prompt_name: <key from configs/prompts/atomic_dataset_prompts.yaml>
    safety:      configs/safety/<scene>.yaml
    # Optional: recovery YAML. When set, run_eval_campaign.py wires a
    # CoursedMpcPlanner; trials that fire a triggering failure save a
    # `recovery_trajectory.npz` under <trial_dir>/ that
    # `export_training_data --trajectory` can render into a parquet.
    # Omit to run eval-only (failures recorded, no replan).
    recovery:    configs/recovery/<scene>_mpc.yaml
    # Optional per-entry overrides — let one scene YAML serve multiple
    # evaluation entries (e.g. center_gate evaluated under two prompts).
    scene_key_override: <unique bundle-dir name>   # default: scene.scene_key
    n_trials_override:  <int>                       # default: top-level n_trials
```

## Falsification → parquet chain

When `recovery:` is set on a scene entry, every trial whose failure type
matches the recovery YAML's `trigger_failure_types` produces:

```
runs/eval_campaigns/<campaign>/<scene_key>/trial_NNN/
  episode_summary.json        # with a top-level "recovery" block
  recovery_trajectory.npz     # frame-tagged NED Trajectory, prompt + source="recovery"
  trial_card.json
  vla_io/                     # PiGateway debug bundles
```

`campaign_summary.json` records `n_recovery_fired` and a flat list of all
saved NPZ paths under `recovery_npzs:` for downstream consumption.

Batch-render the NPZs into LeRobot parquets with:

```bash
# Per-scene batch (one renderer load per scene_key):
PYTHONPATH=src python -m falsify.cli.export_training_data \
    --trajectories-dir runs/eval_campaigns/<campaign>/<scene_key>/recovery_npzs \
    --scene configs/scenes/<scene>.yaml \
    --frame configs/frames/carl_dual.yaml \
    --embodiment configs/embodiments/carl_dual_mocap.yaml \
    --out runs/recovery_parquets/<scene>
```

(`--trajectories-dir` discovers `*.npz` directly under that dir; collect
the per-trial recovery NPZs into one dir per scene first, e.g. via a
symlink fan-out, since the campaign output is one NPZ per
`trial_NNN/`.)

When `scene_key_override` is set, the per-trial seed hashes
`(master_seed, scenario_name, scene_key_override, trial_index)` — so two
entries against the same scene file but different `scene_key_override`s
draw **independent** start jitter, not identical jitter under two prompts.

## Current scoring rules (cross-cutting — apply to every scenario)

1. **Goal-tolerance region**: `safety.miss_gate.goal_tolerance_half_extents`
   (axis-aligned box in MOCAP) takes precedence over `goal_tolerance_m`
   (sphere). The gate-scenes safety YAMLs ship a `0.6 × 0.6 × 1.0 m`
   box around the goal; the sphere is left as a legacy fallback.
2. **Directional gate transit**: scene_keys ending in `_from_left` /
   `_from_right` enforce a signed crossing of the gate's mid-y plane
   inside the aperture. A SUCCESS posthoc requires (gate-AABB transit)
   ∧ (within goal box) ∧ (≥1 correct-direction aperture crossing) ∧
   (zero wrong-direction aperture crossings). Other scene_keys skip
   the directional check.
3. **VLA image preprocess (v7 finetunes only)**: every
   `configs/policies/pi_gateway/*.yaml` for the v7 gate-scenes
   finetunes must set `image_size: 256` and `channel_order: "BGR"` —
   these mirror the training-data preprocess in
   `src/falsify/training/exporter.py`. Without them the policy sees
   color-swapped pixels at a different aspect ratio than at training
   time.

To re-apply rules (1) and (2) to a campaign whose rollouts were
captured under older rules, run `scripts/eval/reclassify_campaign.py
--campaign runs/eval_campaigns/<name>`. It updates per-trial
`episode_summary.json` and `campaign_summary.json` in place (with
`*.json.bak` backups on first run). Rule (3) cannot be applied
post-hoc — it changes what the policy sees, so it requires
re-rolling.

## Determinism contract

- The trial-card seed is `hash((master_seed, scenario_name, scene_key, trial_index)) mod 2**32`.
  Adding a new scenario or scene cannot perturb existing trials.
- Trial cards capture **absolute** values (start position in NED + mocap,
  gate Δxyz + Δyaw in mocap). The campaign runner uses these directly —
  no resampling.
- Output of `generate_eval_bundles.py` should be committed to the repo so
  the campaign's source-of-truth is in git.
