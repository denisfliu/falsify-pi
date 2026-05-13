---
name: falsify-trajectory-from-splatnav
description: (STUB — recovery wired, full pipeline pending) Plan a collision-free trajectory through a Gaussian splat scene via SplatNav (A* + spline), emit as a canonical Trajectory NPZ.
---

# falsify-trajectory-from-splatnav

**Status: partially implemented.** `falsify.recovery.SplatNavPlanner`
already exists for the failure-recovery path; this skill packages
SplatNav as a *primary* trajectory generator.

## Intended inputs

- scene YAML
- start position in NED (e.g., drone's current pose)
- goal position in NED
- (optional) collision bbox in NED

## Intended pipeline

1. Build the `FrameGraph` and `SplatNavPlanner(cfg, frame_graph, …)`.
2. Call `planner.plan(start_ned, goal_ned)` → `RecoveryResult` with
   `Trajectory["ned"]` field.
3. Resample (`falsify.training.resample`) to the embodiment's fps.
4. Save as a Trajectory NPZ.

## Intended CLI shape

```bash
.venv/bin/python -m falsify.cli.plan_splatnav_trajectory \
    --scene configs/scenes/left_gate.yaml \
    --start "[-0.5, -0.7, -1.5]" \
    --goal  "[1.5, -0.7, -1.5]" \
    --hz 10 \
    --out runs/splatnav_$(date +%Y%m%d_%H%M%S)/trajectory.npz
```

## Hands off to

- **`falsify-export-parquet`** — turn the planned trajectory into training data.
- **`falsify-falsify-trajectory`** — inject perturbations on top.

## TODO

- A small `cli/plan_splatnav_trajectory.py` wrapping
  `falsify.recovery.SplatNavPlanner` (the planner itself is done).
- Decide whether quaternion authoring is "look at next waypoint" or
  identity-with-yaw-rate, and document under `recovery/CLAUDE.md`.
