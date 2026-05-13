---
name: falsify-author-waypoints
description: Author a Course YAML (waypoints in a scene), visualize them against the scene's gate/table point clouds, and iterate until they look right. The only step a human has to do thoughtfully — everything downstream is automatic.
---

# falsify-author-waypoints

The one task that genuinely needs human judgment: choosing good
waypoints. Everything after (planning, rendering, exporting parquets)
is automatic.

## When to use

You want to produce training data for a specific maneuver — fly through
a gate, hover over an object, follow a corridor — and need to author
the waypoint sequence.

## Inputs

- a scene (`configs/scenes/<scene>.yaml`)
- a rough idea of what the drone should do, expressed as 3–10 mocap
  positions

## Procedure

### 1. Start from the template course

```bash
cp configs/courses/through_left_gate.yaml configs/courses/<my_course>.yaml
```

Open the YAML and edit:

- `name`: short slug; appears in the `Trajectory.source` and dataset manifest.
- `scene`: the scene this course is intended for (informational).
- `total_time_s`: how long the drone has to traverse the course.
- `yaw_mode`:
  - `tangent` (default) — yaw follows the path
  - `interp` — linearly interpolate between waypoints that specify `yaw`
  - `fixed` — all unset yaws default to 0
- `waypoints`: 3D positions in MOCAP (z-up, origin at the ArUco tag).
  Optional `yaw` and `t` per waypoint.

### 2. Visualize against the scene

```bash
PYTHONPATH=src .venv/bin/python -m falsify.cli.visualize_waypoints \
    --course configs/courses/<my_course>.yaml \
    --scene configs/scenes/<scene>.yaml \
    --out runs/wpviz/<my_course> \
    --plan
```

This writes, for each of `mocap`, `ned`, `ns`:

```
combined_<frame>.ply   — waypoints (colored markers) + planned spline
                         (yellow line) + scene objects (gate, table)
waypoints_<frame>.ply  — just the markers
spline_<frame>.ply     — just the planned spline (requires --plan)
```

**Open `combined_mocap.ply` in MeshLab / open3d / blender** and check:
- Waypoints are *outside* solid objects (gate posts, table).
- Endpoint colors are right: start = green, goal = yellow, interior = blue.
- The yellow spline curves *between* the gates rather than through a post.
- The whole spline stays inside the gsplat's training extent (no waypoints
  in the corners of the room — there are no gaussians there).

### 3. Iterate

Edit the YAML, re-run the visualizer, repeat. A good course usually
needs 3–6 iterations before the spline traces a clean path.

### 4. Quick numerical sanity (optional)

The visualizer's stdout already prints each waypoint's resolved `t` and
yaw. Cross-check that:
- `t` values are monotonically increasing and within `[0, total_time_s]`.
- For `yaw_mode: tangent`, the yaw rotates smoothly along the path (no
  +π/-π jumps unless the trajectory genuinely doubles back).

## Hands off to

- **`falsify-trajectory-from-waypoints`** — once the YAML looks right,
  plan an NPZ and (optionally) export training data.

## Gotchas

- **Waypoints are MOCAP**, not NED. If you typed NED-style values
  (z-down, negative for "up"), the spline will be upside down. The
  visualizer's `combined_<frame>.ply` makes this immediately obvious in
  MeshLab.
- **Gates have non-trivial thickness.** Inspect `objects_summary.json`
  for AABBs; a "go through the gate" waypoint at the gate's *center*
  (x ≈ 0.86 for left_gate) often clips a post. Bias toward the open
  side.
- **The gsplat extent is finite.** Gaussians thin out beyond the
  trained workspace (roughly mocap-y ∈ [-1.5, 1.2] for the gate
  scenes). Waypoints outside that envelope render to gray.
- **Yaw is specified in the course's frame** (mocap by default). The
  planner converts to NED. Don't pre-flip the sign.

## What's NOT your job here

- Frame conversions — handled by the FrameGraph and the planner.
- Trajectory smoothness — handled by the cubic spline.
- Rendering — `falsify-export-parquet` does it later.
- Reaching the OpenPI server — not used in this workflow at all.
