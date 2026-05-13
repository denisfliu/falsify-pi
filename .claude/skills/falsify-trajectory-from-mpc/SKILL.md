---
name: falsify-trajectory-from-mpc
description: (STUB — not yet implemented) Plan a feasible trajectory via FiGS' VehicleRateMPC over a user-supplied course of waypoints; emit as a canonical Trajectory NPZ.
---

# falsify-trajectory-from-mpc

**Status: not yet implemented.** This skill describes the intended
interface; implement when the FiGS-MPC integrator replaces the
trajectory-replay `_step_replay` (see `src/falsify/sim/CLAUDE.md`).

## Intended inputs

- scene YAML (defines `FrameGraph` + gsplat + drone start/goal in mocap)
- drone-frame YAML (mass, inertia, MPC weights)
- a *course* — start/end keyframes + optional intermediate waypoints
  (the FiGS `course` dict format: `keyframes`, `waypoints.Nco`, etc.)
- task prompt

## Intended pipeline

1. Build a `Simulator` with the FiGS-MPC integrator (replacing the
   `_step_replay` trajectory-replay one).
2. Construct `VehicleRateMPC` from the drone-frame YAML.
3. Set its reference trajectory (`tXUd`) from the course.
4. Roll out closed-loop until the course's time budget elapses,
   collecting per-step `DroneState`s in NED.
5. Wrap into a `Trajectory` via `from_episode_trace(ep)`.

## Intended CLI shape

```bash
.venv/bin/python -m falsify.cli.plan_mpc_trajectory \
    --scene configs/scenes/left_gate.yaml \
    --frame configs/frames/carl_dual.yaml \
    --course configs/courses/through_gate.yaml \
    --out runs/mpc_$(date +%Y%m%d_%H%M%S)/trajectory.npz
```

## Hands off to

- **`falsify-export-parquet`** — turn the planned trajectory into training data.
- **`falsify-falsify-trajectory`** — inject perturbations / failures on top
  of the nominal MPC plan.

## TODO before this skill becomes real

- Replace `Simulator._step_replay` with an MPC-driven integrator (acados).
- Author a `course` YAML schema (start/end/intermediate waypoints with
  feasibility constraints).
- Implement `cli/plan_mpc_trajectory.py` analogous to `run_vla_episode.py`.
