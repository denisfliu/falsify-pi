# `falsify.planning/` — waypoint-driven trajectory planning

**Status:** spline planner done; MPC / SplatNav planners stubbed in
sibling skill docs.

## Workflow

```
   YAML (configs/courses/*.yaml)
            │
            ▼  load_course
       Course   (waypoints in MOCAP, total_time, yaw_mode, ...)
            │
            ▼  plan_spline (default)  /  plan_mpc (future)  /  plan_splatnav (future)
   Trajectory NPZ (NED, fps-sampled)
            │
            ▼  TrainingDataExporter
   episode_*.parquet
```

The `Course` dataclass is the single shared input across all planner
backends. Adding a new planner = a single new function returning a
canonical `Trajectory`; the rest of the pipeline doesn't change.

## Course YAML

Documented inline in `waypoints.py`. The essential bits:

```yaml
name: <slug>
scene: configs/scenes/<scene>.yaml
frame: mocap                 # frame the waypoint positions live in
fps: 10
total_time_s: 8.0
yaw_mode: tangent            # fixed | interp | tangent
waypoints:
  - { name: start, p: [...], yaw: 0.0, t: 0.0 }
  - { name: ...,   p: [...] }
  - ...
velocity_constraints:        # optional; honoured by future MPC planner
  max_speed_mps: 1.5
```

Per-waypoint `yaw` and `t` are optional. Missing `t`s fill in by
chord-length between set values; missing `yaw`s resolve per `yaw_mode`.

## Public API

- `load_course(path) -> Course`
- `plan_spline(course, frame_graph, *, prompt="") -> Trajectory`
  (returns ``falsify.training.Trajectory``)

## Yaw frame handling

The course author writes yaws in the source frame (typically mocap,
where +z is up). NED's yaw rotates the other way (NED z is down). The
planner reads the scene's FrameGraph and applies a sign flip per the
rotation matrix's z-axis — so authors *never* hand-flip yaw signs.

## Hands-off integration with other modules

- `falsify.training` consumes the planner's output Trajectory NPZ
  directly. No schema translation; the planner emits the canonical type.
- `falsify.cli.visualize_waypoints` and `falsify.cli.plan_trajectory`
  wrap the planning module for shell access.
- The `falsify-author-waypoints` and `falsify-trajectory-from-waypoints`
  skills document the human-driven steps end-to-end.

## Adding a new planner

1. Add `planning/<name>.py` exposing `plan_<name>(course, frame_graph,
   *, prompt="") -> Trajectory`.
2. Add it to `planning/__init__.py`'s exports.
3. Add a `--planner <name>` branch in `cli/plan_trajectory.py`.
4. Write a new skill `falsify-trajectory-from-<name>` documenting the
   inputs and any extra configuration (e.g. an MPC tuning YAML).
5. The Course YAML schema should not change unless the new planner
   genuinely needs a new field — if it does, make it optional and
   default-fallback so existing courses keep working.
