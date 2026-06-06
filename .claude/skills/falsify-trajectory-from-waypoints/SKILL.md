---
name: falsify-trajectory-from-waypoints
description: Turn a Course YAML (waypoints) into a canonical Trajectory NPZ via the spline planner. Hand it directly to falsify-export-parquet to produce training data without ever running a VLA.
---

# falsify-trajectory-from-waypoints

The "I authored some waypoints, now give me a Trajectory NPZ" step. Default
planner is cubic spline; pass `--planner mpc` to use FiGS' VehicleRateMPC
instead (same Course YAML, dynamically-feasible output) — see
`falsify-trajectory-from-mpc`.

## Inputs

- `configs/courses/<course>.yaml` — authored via
  **`falsify-author-waypoints`**.
- `configs/scenes/<scene>.yaml` — the scene the waypoints reference.

## Procedure

### Single course

```bash
PYTHONPATH=src .venv/bin/python -m falsify.cli.plan_trajectory \
    --course configs/courses/through_left_gate.yaml \
    --scene configs/scenes/left_gate.yaml \
    --out runs/courses/through_left_gate/trajectory.npz \
    --prompt "go through the gate and hover over the stuffed animal"
```

Output: a single Trajectory NPZ. Stdout reports frame count and duration.

### Many courses, one scene

If you've authored a directory of `<scene>__<variation>.yaml` files,
loop over them with a tiny shell wrapper:

```bash
for course in configs/courses/left_gate_*.yaml; do
  name=$(basename "$course" .yaml)
  PYTHONPATH=src .venv/bin/python -m falsify.cli.plan_trajectory \
      --course "$course" --scene configs/scenes/left_gate.yaml \
      --out "runs/courses/${name}/trajectory.npz" \
      --prompt "go through the gate and hover over the stuffed animal"
done
```

## What the spline planner does

1. Resolves per-waypoint `t` and `yaw` from the course (gaps filled per
   `yaw_mode` and path-length parameterisation; see
   `src/falsify/planning/waypoints.py`).
2. Converts waypoint positions from `course.frame` (mocap by default) to
   NED via the scene's FrameGraph.
3. Fits a cubic spline (not-a-knot bc) through the NED positions.
4. Linearly interpolates yaws (with shortest-arc unwrap) and converts to
   NED yaw — `yaw_ned = z_sign * yaw_<src>` per the FrameGraph rotation
   sign at the relevant edge.
5. Samples at `fps`, builds quaternions in NED directly.

The result is a NED-frame Trajectory NPZ identical in shape to one
produced by `falsify-trajectory-from-vla`.

## Hands off to

- **`falsify-export-parquet`** — render and emit the parquet.
- **`falsify-orchestrate-batch`** — bulk-export many courses across scenes.

## Other planners (same Course YAML, swap the backend)

- **`falsify-trajectory-from-mpc`** (done) — wrap the same Course YAML
  with `--planner mpc` and let `VehicleRateMPC` produce a dynamically-
  feasible trajectory. Used by `falsify.recovery.CoursedMpcPlanner` as
  the default recovery backend.
- **`falsify-trajectory-from-splatnav`** (stub — primary planner
  pending; recovery wired) — replace the spline segment with SplatNav's
  collision-free A* + spline through the Gaussian-splat scene.

## Gotchas

- The spline is **purely geometric** — it does not enforce a max speed
  or thrust limit. If the resulting trajectory is too aggressive,
  increase `total_time_s` in the course YAML and replan. (When MPC
  arrives, `velocity_constraints` in the YAML will be honoured.)
- Spline samples land at `fps` ticks; if the parquet target wants a
  different rate, pass `--hz` to `falsify.cli.export_training_data`
  (the exporter resamples).
- The CLI is fast (~milliseconds — no rendering). Iterating on courses
  is cheap; do the visualization step first to catch geometry issues
  before any rendering.
