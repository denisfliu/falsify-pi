---
name: falsify-trajectory-from-mpc
description: (STUB — half of the infrastructure is already done) Plan a dynamically-feasible trajectory via FiGS' VehicleRateMPC over a Course YAML. The Course schema and visualization are shared with the spline planner; only the planner backend itself is pending.
---

# falsify-trajectory-from-mpc

**Status: stub. The Course YAML schema, the waypoint visualization, the
`plan_trajectory` CLI scaffold, and the downstream export pipeline are all
done — only the MPC backend itself is pending.**

When this is implemented, the user-facing workflow is *identical* to the
spline planner — only `--planner mpc` changes:

```bash
.venv/bin/python -m falsify.cli.plan_trajectory \
    --course configs/courses/through_left_gate.yaml \
    --scene configs/scenes/left_gate.yaml \
    --planner mpc \
    --out runs/courses/through_left_gate/trajectory.npz
```

## What's already done (do not redo)

- **Course YAML schema** (`configs/courses/*.yaml`, loader in
  `src/falsify/planning/waypoints.py`). MPC will honour the same
  `waypoints`, `total_time_s`, `yaw_mode`, and `velocity_constraints`
  fields — the latter is already parsed and currently informational.
- **Waypoint authoring + visualization** via
  `falsify-author-waypoints` and `cli/visualize_waypoints.py`.
- **`plan_trajectory` CLI** (`src/falsify/cli/plan_trajectory.py`)
  already has a `--planner` argument; adding `mpc` is one extra
  conditional branch.
- **Downstream export** to LeRobot-style parquet works regardless of
  the planner.

## What you (the implementer) actually need to do

1. Add `src/falsify/planning/mpc.py` exposing
   `plan_mpc(course, frame_graph, *, prompt="") -> Trajectory`.

2. Inside it:
   - Convert the course's waypoints to a FiGS-style `course` dict
     (start/end keyframes, intermediate waypoints, `Nco`,
     `forces`). See SousVide's
     `vla_falsification/control/vla_policy.py:build_mpc` for a working
     example of constructing the dict.
   - Build a `VehicleRateMPC` from `figs.control.vehicle_rate_mpc`.
   - Set the reference trajectory (`tXUd`) covering the course time
     budget at the course's `fps`.
   - Simulate the closed loop forward using `figs.dynamics`'s
     `quadcopter_rate_model` (open-loop integrator) — same pattern
     as SousVide's orchestrator but without the VLA in the loop.
   - Collect per-step `DroneState`s and pack into a Trajectory NPZ
     via `falsify.training.from_episode_trace` (works on any
     iterable of `DroneState`s).

3. Register `plan_mpc` in `src/falsify/planning/__init__.py`.

4. Add a `mpc` branch in `cli/plan_trajectory.py`'s `--planner`
   argument and a `MPC_PARAMS` YAML if MPC weights/horizon need
   tuning per course.

5. Update this SKILL.md: drop the "stub" status, add the actual
   gotchas you hit during implementation.

## Inputs at implementation time

You'll need:
- A working acados install (SousVide's `.venv` has it; the symlinked
  `.venv` in this repo inherits it).
- The drone-frame YAML for vehicle parameters (mass, inertia, motor
  coefficients — same one the rest of the pipeline uses).
- A scene YAML — the planner doesn't query the gsplat, but does use
  the FrameGraph to convert course waypoints from mocap to NED.

## Honest preflight

Before coding, confirm:
- `from figs.control.vehicle_rate_mpc import VehicleRateMPC` succeeds
  under our env. If acados or its python bindings are missing, see
  `external/FiGS/acados/` for the build steps.
- The drone-frame YAML's `mass` / `kt` / `Nrtr` values produce a
  sensible hover thrust under your MPC settings. SousVide uses
  `hover_thrust = -(m*g) / (Nrtr*kt)`.

## Hands off to

- **`falsify-export-parquet`** — same path the spline planner uses.
- **`falsify-orchestrate-batch`** — bulk-plan + export many courses.
