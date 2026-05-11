# `falsify.sensors/` — pluggable observation pipeline

**Status:** Phase 2/3 scaffold in place (`Sensor` ABC, `SensorRig`,
`StateSensor`, `PromptSensor`, `CameraSensor`). The actual gsplat-backed
renderer that `CameraSensor` calls is wired in Phase 2 alongside the
simulator wrapper.

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

## Adding a sensor

1. Subclass `Sensor`. Declare `keys_provided`. Implement `sense()`.
2. Register a factory in `factory.py` (Phase 3) or instantiate directly in the orchestrator.
3. Update the namespace table above.
