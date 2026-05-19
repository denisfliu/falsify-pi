# `configs/eval_suite/` — evaluation scenarios

Declarative scenario definitions consumed by the evaluation framework
(scripts/generate_eval_bundles.py + scripts/run_eval_campaign.py +
scripts/summarize_eval_campaign.py).

## Pipeline

```
scenario YAML    →    trial cards (JSON)        →    per-trial outputs    →    campaign summary
configs/eval_suite/  runs/eval_bundles/<name>/        runs/eval_campaigns/<name>/<scene>/trial_NNN/
```

Trial cards are **absolute** descriptions of each trial (start position
in mocap+ned, gate-perturbation deltas in mocap). Two policies hitting
the same card see byte-identical conditions, regardless of any later
refactor to the random samplers.

## Scenarios shipped

| File | Scenes | Perturbations |
|---|---|---|
| `pure.yaml` | left_gate, right_gate, center_gate | start jitter only |
| `gate_perturbed_small.yaml` | left_gate, right_gate | start jitter + GateRigidPerturbation (±3 cm, ±3°) |
| `gate_perturbed_large.yaml` | left_gate, right_gate | start jitter + GateRigidPerturbation (±10 cm, ±10°) |
| `compositional.yaml` (Phase 3) | left_and_center, right_and_center | two gates, combined prompt |

## How to run

```bash
# 1. Generate trial bundles (idempotent; commit them to lock the run).
PYTHONPATH=src python scripts/generate_eval_bundles.py \
    --scenario configs/eval_suite/pure.yaml

# 2. Run a campaign against a policy (Pi-gateway only at the moment).
bash -c 'export PI_API_KEY=...; source tools/env.sh; \
    source tools/pi_inference_env.sh; \
    PYTHONPATH=src python scripts/run_eval_campaign.py \
        --scenario configs/eval_suite/pure.yaml \
        --policy-config configs/policies/pi_gateway/history_h6jtbq0w_20k.yaml \
        --frame configs/frames/carl_dual.yaml \
        --out runs/eval_campaigns/<campaign_name>'

# 3. Summarise (one or more campaigns side-by-side).
PYTHONPATH=src python scripts/summarize_eval_campaign.py \
    runs/eval_campaigns/pi07_history_pure \
    runs/eval_campaigns/pi07_nonhistory_pure
```

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
```

## Determinism contract

- The trial-card seed is `hash((master_seed, scenario_name, scene_key, trial_index)) mod 2**32`.
  Adding a new scenario or scene cannot perturb existing trials.
- Trial cards capture **absolute** values (start position in NED + mocap,
  gate Δxyz + Δyaw in mocap). The campaign runner uses these directly —
  no resampling.
- Output of `generate_eval_bundles.py` should be committed to the repo so
  the campaign's source-of-truth is in git.
