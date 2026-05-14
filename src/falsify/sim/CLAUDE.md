# `falsify.sim/` — simulator wrapper

**Status:** done (v0). Trajectory-replay integrator only — FiGS MPC + ACADOS
integration is deferred behind `_step_replay` until we want closed-loop
dynamics. The public API is stable enough that swapping integrators won't
churn callers.

## Public surface

- `DroneState` — frame-tagged drone state: `pos: Point["ned"]`, `vel`, `quat_xyzw`, `t`. Encapsulates the FiGS 10-vector layout (`from_vector` / `to_vector`).
- `SimulatorConfig(hz, horizon_s, policy_hz, chunk_steps)` — `chunk_steps` re-queries the policy after that many simulator steps (or sooner if the active chunk's waypoints run out), used by VLA-style policies that emit ~50-step chunks. `policy_hz` is the legacy fixed-cadence fallback when `chunk_steps` is unset.
- `Simulator(cfg, frame_graph)` — wraps the integrator. Methods: `reset(initial_state)`, `state` property, `rollout_with_policy(policy, sensor_rig, max_steps=None, detector=None, perturbations=None)`.
- `EpisodeTrace` — `states`, `policy_outputs`, `chunk_starts`, `failure`. `.trajectory()` is the only frame-tagged accessor.
- `GSplatRenderer(gsplat_config_yml_path, world_frame="ned")` — lazy-imports FiGS. `render(camera_pose_world: Pose, intrinsics: dict) -> (rgb_uint8, depth_or_None)`. Cameras themselves live in `falsify.sensors/` so policies can opt in.
- `body_to_world_se3(state)` and `camera_to_world_pose(state, body_from_camera)` — runtime body↔world hinge in `poses.py`. The `Simulator` and the sensor factory take this as a callable so the static/runtime split stays explicit.

## Replay-integrator semantics

`_step_replay(state, chunk, offset, dt)` advances one timestep by indexing
the chunk at `offset`:

- Position from `chunk.positions[idx]`, velocity from `chunk.velocities[idx]` (zero if absent), **orientation from `chunk.quaternions[idx]` when present** (so the VLA's yaw-deltas drive the camera between queries) else the state's previous quat is held. `idx` clamps to the last waypoint.
- Re-query is triggered when `chunk_offset >= chunk_steps` (or the chunk is empty / exhausted), whichever happens first.

## Frame contract

Simulator state, integration, and the FiGS control interface all live in
`"ned"`. `GSplatRenderer` is a thin wrapper around `figs.render.gsplat.GSplat`
that inherits FiGS' boundary: **takes NED in**, FiGS internally applies
``Tw2g = Tdp_scaled @ diag(1,-1,-1,1)`` (perm5 NED→MOCAP, then the scene's
training-time dataparser MOCAP→NS), and hands NS-frame camera-to-world
poses to the nerfstudio pipeline that actually renders the gsplat.

The gsplat itself **lives in NS** — that's where the gaussians are stored.
We just don't expose that frame at the wrapper boundary; it stays inside
FiGS so callers can keep speaking NED. The correctness of this hinges on
the scene YAML's ``ned↔mocap`` matching FiGS' baked-in perm5
(``diag(1, -1, -1)``); if they disagreed, the renders would silently
disagree with the policy's view of where the drone is.

Runtime body→world composition (drone state ⊗ static body→camera) happens
in `poses.py`, not in the `FrameGraph` — see `geometry/CLAUDE.md` for the
static/runtime split.

## Scene edits (`scene_edits.py`)

A `RigidTransformAABB` edit moves a subset of the loaded Gaussians by
selecting them with axis-aligned and yawed boxes, then applying a rigid
transform authored in some frame (typically MOCAP). The selection mask
is::

    broad   = target_aabb \ (exclude_aabbs ∪ oriented_exclude_aabbs)
    precise = include_aabbs ∪ oriented_include_aabbs
    move    = broad ∪ precise

**Precise inclusions override excludes.** `target_aabb_*` is meant to be
a loose-fit bracket — anything inside it that *isn't* in an exclusion
moves. `include_aabbs` / `oriented_include_aabbs` are hand-curated ground
truth: every Gaussian inside one of them moves, *even if it also falls
inside an exclusion*. The intent is that when a human paints a region in
`paint_gaussian_mask` and asserts "this is gate", the table-exclude
shouldn't be allowed to bite into it.

Containment is evaluated in the edit's `target_aabb_frame` directly
(via a cached NS→authored-frame transform of `means`), so the user's
authored box bounds are respected exactly — no corner-rebracketing
approximation. This matters when MOCAP→NS has a non-axis-aligned
rotation: the old NS-frame AABB test was an over-approximation.

Two parallel appliers keep visualisation and rendering in sync:

- `apply_edits_to_arrays(means_ns, quats_wxyz, edits, fg)` — pure-numpy,
  used by the inspector / PLY exporter / classifier tools.
- `apply_edits_to_pipeline(pipeline, edits, fg)` — mutates a loaded
  nerfstudio pipeline in place; used by `preview_scene_nsviewer` and
  the runtime `GSplatRenderer`. Returns the same mask count as the array
  applier on the same inputs (verified in tests).

Tests live in `tests/test_scene_edits.py`. When changing the mask
semantics, update **both** appliers, the dash classifier
(`cli/paint_gaussian_mask.py::_load_means_mocap`), and the SKILL doc
under `.claude/skills/falsify-author-gaussian-mask/`.
