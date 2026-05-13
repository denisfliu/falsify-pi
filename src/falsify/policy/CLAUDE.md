# `falsify.policy/` — policy interface

**Status:** mocks + VLA done. The MPC-backed control path is deferred until
the FiGS integrator replaces the trajectory-replay integrator (Phase 7+).

## Contract

```python
class Policy(ABC):
    required_modalities: frozenset[str] = frozenset()   # e.g. {"images.forward", "images.downward"}
    @abstractmethod
    def observe(self, obs: Observation) -> Trajectory: ...   # frame == "ned"
    def reset(self): ...
```

`Observation` carries `state: DroneState` + dotted-key `data: dict` +
`prompt`. Policies read modality values with `obs.require("images.forward")`
(raises on missing) or `obs.get(...)`. The orchestrator wires a `SensorRig`
to cover `required_modalities` exactly — a mock policy with empty
requirements triggers no camera renders.

## From rollout to training data

`falsify.training.from_episode_trace(ep)` packs a live `EpisodeTrace`
into a canonical `Trajectory` NPZ, which `TrainingDataExporter` then
renders into a LeRobot-style parquet. See `src/falsify/training/CLAUDE.md`
for the full pipeline.

## Implementations

| Class | required_modalities | Notes |
|-------|---------------------|-------|
| `MockStraightLine` | `frozenset()` | Straight-line to goal at constant speed; ignores images. |
| `MockNoisy`        | `frozenset()` | Straight-line + Gaussian noise on waypoints. |
| `VLAPolicy`        | `{"images.forward", "images.downward"}` | OpenPI websocket client; converts NED ↔ MOCAP at the boundary. |

## `VLAPolicy` frame contract

The OpenPI VLA on moraband was trained in MOCAP-Z-up with SousVide perm5
(`R_mocap_from_ned = diag(1, -1, -1)`). `VLAPolicy.observe`:

1. Reads `obs.state.pos` (must be NED) and `obs.state.quat_xyzw` (xyzw).
2. Converts pos NED → MOCAP via the active `FrameGraph`. Computes NED yaw and **negates** it for the server (NED z-down vs. MOCAP z-up induce opposite yaw senses).
3. Resizes both camera RGBs to a square (`image_size`, default 256).
4. Sends the SousVide-style payload over the OpenPI websocket:
   - `observation/image`         — forward, uint8 (256, 256, 3)
   - `observation/wrist_image`   — downward, uint8 (256, 256, 3)
   - `observation/3pov_1`        — static third-person, zeros by default
   - `observation/state`         — float32 shape **(7,)** = `[px, py, pz, -yaw, 0, 0, 0]`
   - `prompt`                    — str
   The `front_1`/`down_1` keys that exist in some SousVide payloads are fake duplicates and intentionally skipped.
5. Receives `actions` (N×≥3 MOCAP-frame position deltas; optional yaw delta at column 3). Integrates cumulatively, prepends the current pose so the chunk starts where the drone is, builds a yaw-only quaternion per waypoint (NED yaw, sign-flipped each step relative to the MOCAP yaw delta).
6. Converts the whole `Trajectory` back to NED and returns it.

`openpi_client` is imported lazily; for testing inject a `pol._client`
attribute with `.infer(payload)` returning `{"actions": np.ndarray}`.

`VLAPolicyConfig`:
- `host`, `port`, `prompt` — server + task.
- `hz`, `actions_per_chunk` — chunk shape; orchestrator's `chunk_steps` should match `actions_per_chunk`.
- `image_size`, `forward_camera`, `downward_camera` — input shape.
- `server_frame` (default `"mocap"`).
- `third_person_image_path` — optional static 3pov channel (defaults to zeros).
- `record_dir` — when set, each query saves a debug bundle under
  `record_dir/query_<NNNN>_step_<KKKKK>/`: native + post-resize renders for
  every channel, raw actions, integrated NED waypoints, and a `data.txt` with
  the state vector actually sent.
