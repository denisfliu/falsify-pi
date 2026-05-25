# `falsify.planning/` — waypoint-driven trajectory planning

**Status:** `plan_spline` and `plan_mpc` shipped. `plan_mpc` is the
default recovery planner used by `falsify.recovery.CoursedMpcPlanner`
and is what every `configs/recovery/*_mpc.yaml` resolves to. A
collision-aware `plan_splatnav` entry point is planned but does not
yet exist in this module — the closest in-tree analogue is
`falsify.recovery.SplatNavPlanner` (A*+spline over the gsplat).

## Workflow

```
   YAML (configs/courses/*.yaml)
            │
            ▼  load_course
       Course   (waypoints in MOCAP, total_time, yaw_mode, ...)
            │
            ▼  plan_spline (default for offline data)
            ▼  plan_mpc    (default for recovery; dynamically feasible)
            ▼  plan_splatnav (future; collision-aware; not yet in planning/)
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
velocity_constraints:        # optional; informational for spline,
  max_speed_mps: 1.5         # honoured implicitly by plan_mpc via the
                             # min-time-snap reference + rate bounds
```

Per-waypoint `yaw` and `t` are optional. Missing `t`s fill in by
chord-length between set values; missing `yaw`s resolve per `yaw_mode`.

## Public API

- `load_course(path) -> Course`, `save_course(course, path) -> Path`
- `plan_spline(course, frame_graph, *, prompt="") -> Trajectory`
  (returns ``falsify.training.Trajectory``). Geometric cubic spline
  through the waypoint positions; fast (~ms); no dynamics.
- `plan_mpc(course, frame_graph, *, prompt="", start_state_ned=None,
  total_time_s=None, hz=None, policy_cfg=None, frame_cfg=None,
  use_rti=True) -> Trajectory`. Builds a `figs.tsplines.min_time_snap`
  reference through the waypoints and tracks it closed-loop with
  `figs.control.vehicle_rate_mpc.VehicleRateMPC` (acados-generated IRK
  integrator for the quadcopter rate-input model). `start_state_ned`
  is the recovery hook — pass `last_safe_state` as a 10-vector to
  re-plan from mid-rollout. `frame_cfg` defaults to
  `configs/frames/figs/carl.json` (drone physical parameters).
  acados compiles into a fresh `tempfile.TemporaryDirectory` so
  concurrent planners (e.g. recovery MPC + a future VLA-side MPC)
  don't fight over `./c_generated_code/`. `use_rti=True` is SQP-RTI
  (one SQP iteration per tick, ~3× faster than full SQP, byte-identical
  to 1e-7 m on the gate courses).
- `perturb_waypoint(course, name, direction, magnitude_m) -> Course`
  — single-shot nudge. `direction ∈ {center, up, down, left, right}`;
  left/right are body-relative (perpendicular to local heading in xy).
- `sample_variants(course, waypoint_name, *, modes, magnitude_range_m,
  n_per_mode, seed) -> list[CourseVariant]` — reproducible batch
  generator. Used by the `falsify-perturb-course` skill to author
  corrective-maneuver datasets.

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
