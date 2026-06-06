---
name: falsify-trajectory-from-mpc
description: Plan a dynamically-feasible trajectory via FiGS' VehicleRateMPC over a Course YAML. Returns a canonical Trajectory NPZ — same downstream hand-off as the spline planner.
---

# falsify-trajectory-from-mpc

**Status: done.** `plan_mpc` is shipped, the `--planner mpc` CLI flag is
wired, and `falsify.recovery.CoursedMpcPlanner` consumes it as the
default recovery backend.

## Quick start

```bash
.venv/bin/python -m falsify.cli.plan_trajectory \
    --course configs/courses/through_left_gate.yaml \
    --scene configs/scenes/left_gate.yaml \
    --planner mpc \
    --out runs/courses/through_left_gate/trajectory.npz
```

Drop `--planner mpc` to fall back to the geometric cubic-spline planner
(the default; faster, no dynamics).

## Where the pieces live

- **CLI entry**: `src/falsify/cli/plan_trajectory.py` (`--planner mpc`).
- **Planner**: `src/falsify/planning/mpc.py::plan_mpc(course, frame_graph, *,
  prompt="", start_state_ned=None, total_time_s=None, hz=None,
  policy_cfg=None, frame_cfg=None, use_rti=True)`.
- **Reference generator**: `figs.tsplines.min_time_snap` over the course
  waypoints; tracked closed-loop by `figs.control.vehicle_rate_mpc.VehicleRateMPC`
  (acados-generated IRK integrator for the quadcopter rate-input model).
- **Drone params YAML**: defaults to `configs/frames/figs/carl.json`
  (mass, inertia, motor coefficients). Override with `--frame-cfg`.

## Recovery hook

`plan_mpc(..., start_state_ned=...)` re-plans from any 10-vector NED
state. `falsify.recovery.CoursedMpcPlanner` uses this seam to resume from
`last_safe_state` mid-rollout. See `src/falsify/recovery/CLAUDE.md`.

## Performance knobs

- `use_rti=True` (default) — SQP-RTI, one SQP iteration per tick. ~3×
  faster than full SQP, byte-identical to 1e-7 m on the gate courses.
- acados compiles into a fresh `tempfile.TemporaryDirectory` so
  concurrent planners (recovery MPC + future VLA-side MPC) don't fight
  over `./c_generated_code/`.

## Inputs (preflight)

- Working acados install — the symlinked `.venv` inherits SousVide's.
  Sanity check: `from figs.control.vehicle_rate_mpc import VehicleRateMPC`
  must succeed.
- A drone-frame YAML (`configs/frames/figs/carl.json`) with `mass`,
  inertia, `kt`, `Nrtr` populated. Hover-thrust sanity:
  `-(m*g) / (Nrtr*kt)` should be in the motor's normal operating range.

## Hands off to

- **`falsify-export-parquet`** — render the NPZ against a scene → parquet.
- **`falsify-orchestrate-batch`** — bulk-plan + export many courses.
