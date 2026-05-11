# `falsify.perturbations/` — three perturbation surfaces

**Status:** done. Environment-perturbation impls are intentionally stubbed
until the new gsplat asset arrives (the user noted Splat-MOVER edits will
land then).

## Surfaces

| Stage | ABC | When it runs | Mutates |
|-------|-----|--------------|---------|
| observation | `ObservationPerturbation` | after `SensorRig.build()`, before `policy.observe()` | the `Observation` |
| action      | `ActionPerturbation`      | after `policy.observe()`, before the integrator step    | the `Trajectory` (still in NED) |
| environment | `EnvironmentPerturbation` | at episode reset (and optionally per-step)              | the gsplat (`Splat-MOVER`) |

```python
class PerturbationSuite:
    def __init__(observation=(), action=(), environment=(), seed=None): ...
    def reset(): ...                                  # re-seed RNG + per-pert reset
    def apply_observation(obs) -> Observation: ...
    def apply_action(traj, frame_graph=None) -> Trajectory: ...
    def apply_environment(gsplat) -> None: ...
    def manifest() -> dict: ...                       # JSON-serializable run record
```

## Frame contract

Every perturbation preserves the frame tag on every value it touches:
- `ActionPerturbation`s operate on a `Trajectory` and return a `Trajectory`
  whose `frame` is the same. Position-space perturbations apply noise in the
  trajectory's own frame.
- `ObservationPerturbation`s modify keyed modalities inside `Observation.data`.
  When they touch a `Point` (e.g. `StateNoise` on `state.pos`), they
  construct a new `Point` in the original frame.
- `EnvironmentPerturbation`s mutate the gsplat in its own internal frame
  (NS); the FrameGraph is available for any coordinate conversions.

## Built-in perturbations

| Name | Surface | What it does |
|------|---------|--------------|
| `PositionNoise(std)` | action | Adds Gaussian noise to every waypoint position |
| `PositionBias(bias_xyz)` | action | Constant 3-vector bias on every waypoint |
| `VelocityScale(scale)` | action | Multiplies trajectory velocities |
| `ImageGaussianNoise(camera, std)` | observation | Adds Gaussian noise to one camera's image |
| `ImageBlur(camera, kernel)` | observation | Box-blurs one camera's image |
| `StateNoise(std)` | observation | Frame-preserving noise on `state.pos` |
| `StubEnvironmentPerturbation` | environment | Placeholder; raises until concrete impls land |

## Reproducibility

The suite owns a single numpy `Generator` seeded at construction. `reset()`
re-seeds before each episode, so given the same manifest + seed, the same
sequence of perturbations is replayed. The manifest captures the seed +
every perturbation's parameters; persist it with the episode and you can
rerun deterministically.

## Adding a perturbation

1. Subclass the right ABC (`ObservationPerturbation` / `ActionPerturbation` / `EnvironmentPerturbation`).
2. Implement `apply(...)`. Construct any new geometry values frame-tagged.
3. Override `manifest()` to return a JSON-serializable dict of parameters.
4. Register the class in `cli/smoke_test.py`'s factory dispatch (or the
   campaign loader once Phase 7 lands).
