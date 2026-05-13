---
name: falsify-falsify-trajectory
description: (STUB) Apply perturbations / inject failure conditions on a nominal Trajectory NPZ to produce falsified variants used for adversarial training data.
---

# falsify-falsify-trajectory

**Status: stub.** The component perturbations (`falsify.perturbations`)
exist and operate at policy-output time during a rollout. This skill
will package them as a *post-hoc* trajectory operator: load a nominal
NPZ, apply perturbations, save the falsified NPZ.

## Intended inputs

- nominal Trajectory NPZ (any source)
- perturbation suite YAML (`configs/perturbations/*.yaml`)
- (optional) failure-injection spec (e.g., "drop altitude after 3s")

## Intended pipeline

1. Load nominal Trajectory.
2. Construct `PerturbationSuite` from YAML.
3. Apply `apply_action(traj, frame_graph=fg)` to every chunk window (or
   to the whole trajectory at once).
4. Optionally splice in a recovery trajectory after a chosen failure
   point (uses `falsify-trajectory-from-splatnav` for the recovery
   segment).
5. Save as a new NPZ with `source="falsified"` plus a sibling JSON
   describing the perturbation chain.

## Intended CLI shape

```bash
.venv/bin/python -m falsify.cli.falsify_trajectory \
    --trajectory runs/vla_*/trajectory.npz \
    --perturbations configs/perturbations/light_action.yaml \
    --seed 0 \
    --out runs/falsified_$(date +%Y%m%d_%H%M%S)/trajectory.npz
```

## Hands off to

- **`falsify-export-parquet`** — render the falsified trajectory to
  training data. Tag the dataset's `manifest.json` with the
  perturbation manifest for reproducibility.

## TODO

- Lift `PerturbationSuite.apply_action` from the per-step orchestrator
  loop into a "whole trajectory" call.
- Splice semantics for recovery handoff (where exactly does the
  nominal end and recovery begin?).
- Write `cli/falsify_trajectory.py`.
