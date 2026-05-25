# `falsify.sensors/` — pluggable observation pipeline

**Status:** done. `Sensor` ABC + `SensorRig` + `StateSensor` + `PromptSensor`
+ `CameraSensor` + `build_sensor_rig` factory all in place. The factory
wires sensors to a policy's `required_modalities` automatically.

## Contract

```python
class Sensor(ABC):
    keys_provided: frozenset[str]              # what this sensor writes
    def sense(state, builder) -> None: ...     # populate exactly those keys
    def reset(): ...                            # called once per episode
```

```python
class SensorRig:
    def __init__(self, sensors: Sequence[Sensor]): ...
    def assert_covers(self, required: Iterable[str]) -> None: ...
    def build(self, state: DroneState) -> Observation: ...
```

Invariants enforced by `SensorRig`:
- **Single-writer per key.** Two sensors providing overlapping keys raises at construction.
- **No spillover.** A sensor that writes a key not in its declared `keys_provided` raises at runtime.
- **Coverage.** `assert_covers(policy.required_modalities)` is the orchestrator's smoke check — fails fast if a policy expects a modality nobody produces.

## Why this exists

Previously each policy implicitly assumed a specific camera setup (forward +
downward + maybe a third-person view). New sensor needs (depth, lidar,
event-cam, IMU) would have meant churning the `Observation` type and every
intermediate function. Now:
- The mock straight-line policy declares `required_modalities = frozenset()` and runs with just `StateSensor` — no camera renders, fast smoke tests.
- A VLA wanting `images.forward + images.downward` declares those keys; the orchestrator wires two `CameraSensor`s and nothing else.
- A future policy wanting lidar declares `lidar.points`; we add a `LidarSensor` class and register it. No existing code changes.

## Naming convention

Keys are dotted: ``namespace.specifier``. Reserved namespaces:
- `state.*` — drone state passthrough (always provided by `StateSensor`).
- `images.<camera_name>` — RGB from one named camera.
- `depth.<camera_name>` — depth from one named camera.

New namespaces (`lidar.`, `imu.`, `event.`, …) are introduced by adding a
sensor class; document the schema for each new namespace here.

### Always-on sensors

- `StateSensor` is **always** instantiated by `build_sensor_rig`,
  regardless of `required_modalities`. Every policy gets `state.*` for
  free; declaring it is unnecessary.
- `PromptSensor` writes to the dedicated `Observation.prompt` field
  (a `str`), **not** a dotted data key. Its `keys_provided` is
  `frozenset()`, so it doesn't participate in the single-writer /
  coverage checks. Wire it via the factory when the policy needs a
  task prompt; it's harmless to include even for prompt-agnostic
  policies.

## Adding a sensor

1. Subclass `Sensor`. Declare `keys_provided`. Implement `sense()`.
2. If the sensor maps to a modality namespace the policy will request, add a branch to `build_sensor_rig` in `factory.py` that constructs it from the relevant config slice.
3. Update the namespace table above.
