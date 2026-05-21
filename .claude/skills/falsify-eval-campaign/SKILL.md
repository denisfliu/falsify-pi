---
name: falsify-eval-campaign
description: Run a reproducible evaluation campaign against a Pi gateway policy and auto-emit per-campaign visualization. Bundles trial-card generation, the rollout loop, and the two HTML reports (3-D trajectories overlay + per-scene stacked outcome bars) under a policy-rooted output tree at `runs/eval_campaigns/<policy_id>/run-NNN-<scenario>-<ts>/`. Also covers re-rendering viz on historical campaigns without re-running rollouts.
---

# falsify-eval-campaign

Reproducible per-policy evaluation. The pipeline takes a scenario YAML
(start jitter / gate-perturbation recipe, scenes, prompts), turns it
into byte-identical trial cards, rolls each card against a Pi gateway
policy, and writes one self-contained run bundle per invocation.

## When to use

- You want to evaluate a new dronevla v7 finetune against the standard
  scenario set (`pure`, `gate_perturbed_small`, `gate_perturbed_large`,
  `compositional`).
- You're A/B-comparing two checkpoints — they must hit byte-identical
  trial cards, which means generating one bundle then running it under
  each policy.
- You're iterating on the viz code and want to backfill the two HTML
  reports onto a historical campaign without paying for another rollout.

Not for: one-off VLA rollouts (use
[`falsify-trajectory-from-vla`](../falsify-trajectory-from-vla/SKILL.md));
authoring new courses (`falsify-author-waypoints`).

## Output layout — this is the contract

```
runs/eval_campaigns/
└── <policy_id>/                            # = policy-YAML stem
    └── run-NNN-<scenario>-<YYYYMMDD_HHMMSS>/
        ├── campaign_summary.json           # aggregated histogram + trials list
        ├── policy_manifest.json            # policy YAML sha, bridge id, traceability
        ├── run_manifest.json               # CLI argv, scenario sha, git rev, started/finished
        ├── campaign.log                    # captured stdout/stderr
        ├── viz/
        │   ├── trajectories.html           # 3-D rollouts, scene-edits-aware, toggleable per scene×outcome
        │   └── outcome_charts.html         # per-scene stacked bars
        └── <scene_key>/trial_NNN/
            ├── episode_summary.json
            ├── rollout_states.npz
            ├── trial_card.json
            ├── vla_io/                     # PiGatewayPolicy debug bundles
            ├── recovery_trajectory.npz     # only if a recovery fired
            └── flythrough_forward.gif      # only with --gif-trials-per-scene > 0
```

- `<policy_id>` is the **stem of the `--policy-config` YAML**, e.g.
  `nonhistory_ccvhs1do_20k`. One folder per deployed checkpoint config.
- `NNN` auto-increments per policy from existing `run-*` siblings (zero-
  padded). Two campaigns against the same policy in the same second land
  in `run-001-…` and `run-002-…`.
- Legacy campaigns (pre-2026-05-20) live under
  `runs/eval_campaigns/legacy/` and are excluded from auto-numbering.

## Step 1 — Generate bundles (once per scenario)

```bash
PYTHONPATH=src python scripts/generate_eval_bundles.py \
    --scenario configs/eval_suite/pure.yaml
```

Writes idempotent `runs/eval_bundles/<scenario>/<scene>/trial_NNN.json`.
Same scenario YAML + `master_seed` → byte-identical cards across hosts
and runs. **Commit the bundle dir** to lock the eval source-of-truth in
git.

## Step 2 — Run the campaign

```bash
bash -c 'export PI_API_KEY=...; source tools/env.sh; \
    source tools/pi_inference_env.sh; \
    PYTHONPATH=src python scripts/run_eval_campaign.py \
        --scenario   configs/eval_suite/pure.yaml \
        --policy-config configs/policies/pi_gateway/nonhistory_ccvhs1do_20k.yaml \
        --frame      configs/frames/carl_dual.yaml'
```

`--out` is optional — when omitted, the runner derives
`runs/eval_campaigns/nonhistory_ccvhs1do_20k/run-NNN-pure-<ts>/` from
the policy stem and scenario name. At end of run it auto-emits
`viz/trajectories.html` and `viz/outcome_charts.html`. Add `--no-viz`
to skip them (the artifacts are still on disk; you can rebuild later
with step 4).

Useful flags:

| Flag | Purpose |
|---|---|
| `--scenes left_gate right_gate` | Restrict to one or more scene_keys for fast smoke. |
| `--trials 0 1 2` | Restrict to specific trial indices (still uses the bundle cards). |
| `--no-rtc` | Disable async RTC; query once per chunk (`execute_chunk_size`). ~22× faster, **not byte-identical** to a deployed RTC checkpoint. |
| `--execute-chunk-size 1` | MPC-style receding horizon (only takes effect with `--no-rtc`). |
| `--no-recovery` / `--force-recovery` | Override the per-card recovery YAML (see `configs/eval_suite/README.md`). |
| `--resume` | Skip trials whose `episode_summary.json` already exists. |
| `--gif-trials-per-scene N` | Render forward-cam GIF for the first N trials of each scene. |

The runner tees stdout/stderr into `<out>/campaign.log` so the on-disk
bundle is self-describing — no `<dir>.log` siblings.

## Step 3 — Summarise (and compare policies)

```bash
PYTHONPATH=src python scripts/summarize_eval_campaign.py \
    runs/eval_campaigns/nonhistory_ccvhs1do_20k/run-001-pure-* \
    runs/eval_campaigns/history_h6jtbq0w_20k/run-001-pure-*
```

Prints per-scenario success/failure breakdowns side-by-side. Use after
running the same bundles under each policy.

## Step 4 — Backfill / re-render viz on an existing campaign

```bash
PYTHONPATH=src python scripts/plot_eval_run.py \
    runs/eval_campaigns/<policy_id>/run-NNN-<scenario>-<ts>
```

Pure consumer of the on-disk artifacts (no GPU, no rollout). Useful
when:
- A campaign was run with `--no-viz`.
- The renderer code in `src/falsify/visualization/eval_report.py` was
  updated and you want the new HTMLs.
- You want viz on a legacy campaign (works against
  `runs/eval_campaigns/legacy/<old-name>/` too).

Flags: `--trajectories-only`, `--charts-only`, `--max-cloud-points N`.

## What the two HTMLs show

**`viz/trajectories.html`** — one 3-D Plotly figure per campaign:

- Per `scene_key`: scene-edits-aware scene PLY (so `center_gate_*` shows
  the gate at its moved position), nominal gate AABB, goal marker, and
  goal-tolerance box / sphere from the matching `configs/safety/*.yaml`.
  All in legendgroup `context_<scene_key>` so the whole scene can be
  toggled off.
- Per trial: rollout polyline colored by `posthoc_outcome`
  (green=SUCCESS, red=COLLISION_GATE, orange=MISS_GATE, purple=
  COLLISION_OTHER, gray=OUT_OF_BOUNDS). Start dot + end ×. Legendgroup
  `<scene_key>/<outcome>` — one legend entry per outcome per scene,
  click to hide that whole group.
- Recovery overlay in cyan dashed when `recovery_trajectory.npz` exists,
  legendgroup `<scene_key>/recovery`.
- Per-trial **perturbed gate AABB** in the trial's outcome color
  whenever the trial card carries a `gate_perturbation` (the actual
  AABB is read from `episode_summary.json.gate_aabb_mocap` written by
  `run_eval_campaign.py`). Legendgroup `<scene_key>/perturbed_gates`.

**`viz/outcome_charts.html`** — per-`scene_key` stacked-bar histogram
of post-hoc outcomes, using the same palette as the trajectories plot.
Each segment is labeled `n/total (pct%)`. Title includes the policy
traceability (variant + W&B run + step) pulled from
`policy_manifest.json`.

## Determinism contract

- Trial-card seed = `hash((master_seed, scenario_name, scene_key, trial_index)) mod 2**32`.
- Trial cards encode **absolute** values (start in NED + mocap,
  gate Δxyz + Δyaw in mocap). The campaign runner uses them directly;
  it never reseeds.
- Two policies hitting the same bundle dir see byte-identical
  conditions.

## Hard rules

1. **Don't write to `runs/eval_campaigns/` top-level by hand.** Only
   the runner should create policy folders + run dirs there. Manual
   artifacts go under `runs/eval_campaigns/legacy/` or your own
   working dir.
2. **`--policy-config` is the source of truth.** The output path is
   derived from its stem; the bridge handshake (`bridge_admin_url` +
   `bridge_policy_id`) flips moraband to the matching checkpoint
   before any inference. Don't rename the YAML mid-experiment.
3. **Bundle cards are git-tracked.** Regenerating with a different
   `master_seed` invalidates every past campaign's reproducibility
   claim. If you must change the recipe, bump `name:` instead.
4. **Viz code is a pure consumer.** `src/falsify/visualization/
   eval_report.py` reads only `campaign_summary.json` +
   per-trial `episode_summary.json` + `rollout_states.npz` (+
   `recovery_trajectory.npz` when present). If you find yourself
   wanting to add a new piece of data to the plot, write it into one
   of those files at run time and update the emitter — don't reach
   into the orchestrator from inside the viz layer.

## Reference docs

- `configs/eval_suite/README.md` — scenario YAML schema + post-hoc
  scoring rules.
- `src/falsify/visualization/eval_report.py` — the two emit functions.
- `scripts/plot_eval_run.py` — backfill / re-render CLI.
- `scripts/run_eval_campaign.py` — the campaign loop itself.
- [`falsify-infer-from-checkpoint`](../falsify-infer-from-checkpoint/SKILL.md)
  — companion skill for the `PiGatewayPolicy` config that
  `--policy-config` points at.
- [`falsify-host-checkpoint`](../falsify-host-checkpoint/SKILL.md) —
  server-side setup for the bridge.
