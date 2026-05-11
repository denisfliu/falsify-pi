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

## Implementations

| Class | required_modalities | Notes |
|-------|---------------------|-------|
| `MockStraightLine` | `frozenset()` | Straight-line to goal at constant speed; ignores images. |
| `MockNoisy`        | `frozenset()` | Straight-line + Gaussian noise on waypoints. |
| `VLAPolicy`        | `{"images.forward", "images.downward"}` | OpenPI websocket client; converts NED ↔ MOCAP at the boundary. |

## `VLAPolicy` frame contract

The VLA was trained in MOCAP-Z-up:
1. `observe(obs)` reads `obs.state.pos` (must be NED).
2. Converts to MOCAP via the active `FrameGraph`.
3. Sends `(pos_mocap, yaw, images, prompt)` to the OpenPI server.
4. Receives `actions` (N×3 position deltas in MOCAP).
5. Integrates deltas into MOCAP waypoints.
6. Converts the whole trajectory back to NED.
7. Returns `Trajectory["ned"]`.

`openpi_client` is imported lazily; for testing inject a `pol._client`
attribute with `.infer(payload)` that returns `{"actions": np.ndarray}`.

`server_frame` is configurable on `VLAPolicyConfig`. If a future VLA trains
in a different frame, edit the config — no code change.
