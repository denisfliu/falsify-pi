# `falsify.sim/` — simulator wrapper

**Status:** Phase 2 in progress. `dynamics_state.DroneState` is in place.
`simulator.py` and `renderer.py` are pending and will be filled out once we
exercise FiGS imports.

## Public surface

- `DroneState` — frame-tagged drone state: `pos: Point["ned"]`, `vel`, `quat_xyzw`, `t`. Encapsulates the FiGS 10-vector layout (`from_vector` / `to_vector`).
- `Simulator(scene_cfg, frame_cfg, forces_cfg, frame_graph)` — thin wrapper around `figs.simulator.Simulator`. Methods: `reset`, `step`, `rollout_with_policy`.
- `Renderer(gsplat, frame_graph)` — GSplat-backed renderer. Public API: `render(camera_pose_world, intrinsics) -> (rgb, depth)`. Used by `falsify.sensors.CameraSensor` as the `renderer` callable.

Cameras themselves live in `falsify.sensors/` so policies can opt in. This
module only exposes the renderer; the sensor system orchestrates *which*
cameras get rendered when.

## Frame contract

Simulator state, integration, and the FiGS control interface all live in
`"ned"`. The renderer translates world-frame poses to the GSPlat's expected
frame via the active `FrameGraph`. Runtime body→world composition (drone
state ⊗ static body→camera) happens here, not in the `FrameGraph` — see
`geometry/CLAUDE.md` for the static/runtime split.
