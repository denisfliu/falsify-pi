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
| `PiGatewayPolicy`  | `{"images.<cam>" for cam in cfg.camera_map}` | `pi-inference-client.PolicyClient` against a Pi gateway URL (WSS + `Api-Key`). Successor for the dronevla v7 gate-scenes finetunes — single base + single wrist cam, 30 Hz, 7-D state/action, server-driven image size. Same NED↔MOCAP boundary as `VLAPolicy`; modality→server-key mapping, control rate, state/action dims, and action column layout are YAML-configurable. Variant configs live under `configs/policies/pi_gateway/`. |

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

## `PiGatewayPolicy` frame contract

Identical NED↔MOCAP boundary to `VLAPolicy` (perm5, negated yaw on the way
out). Differences are in transport and configurability:

- **Transport**: WSS gateway + `Api-Key` header via
  `pi_inference_client.PolicyClient`. `pi_inference_client` is imported
  lazily — falsify loads without the wheel installed.
- **Image resolution**: server-driven. The client reads the server's
  `image_preprocess` contract on connect and resizes natively
  (v7 gate-scenes = 448²). Do not pre-resize on the falsify side.
- **Cameras**: a `camera_map: {falsify_name: server_key}` block in YAML
  decides modality→payload-key mapping; the v7 finetunes use
  `forward → observation/image`, `downward → observation/wrist_image`
  (no 3pov channel). On connect the policy cross-checks
  `server_config.camera_names` and fails fast on mismatch.
- **Control rate / dims**: `hz`, `state_dim`, `action_dim`,
  `action_pos_slice`, `action_yaw_index` are YAML knobs (defaults
  30 / 7 / 7 / `[0:3]` / `3`).
- **API key**: `${env:VAR}` indirection in YAML; source
  `tools/pi_inference_env.sh` to populate `$PI_API_KEY`.
- **Traceability**: each variant YAML carries a `traceability:` block
  (W&B run, step, GCS checkpoint URI, processor name) that the debug
  recorder writes alongside each query bundle.

Variant configs live under `configs/policies/pi_gateway/`.

### Where the gateway runs

`PiGatewayPolicy` speaks the **wire protocol** of Pi's commercial gateway,
but it does not care **who** runs that gateway. Two supported topologies:

1. **Self-hosted** (default for now): a `pi_local_bridge` server on
   moraband (or any GPU host) that loaded the dronevla v7 checkpoint via
   `pi_inference_client.local` and re-exposes it on `ws://host:port/path`.
   This is how we run inference on our own hardware while keeping the
   client side unchanged. Bridge code + deploy notes live in
   `pi_local_bridge/` at the repo root; it has its own `pyproject.toml`
   so the JAX-heavy `[local]` extras stay out of the falsify venv.

2. **Pi-hosted**: a `wss://api.pi-fleet.com/v1/models/<id>` URL Pi
   provisions for a deployed checkpoint. Same client, different `gateway_url`.

Switching topologies is purely a YAML edit — no code change.

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
