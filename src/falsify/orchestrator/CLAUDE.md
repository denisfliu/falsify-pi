# `falsify.orchestrator/` — episode runner

**Status:** episode runner done; campaign sweep pending.

## Public API

```python
class EpisodeConfig:
    scene_cfg: dict
    frame_cfg: dict
    episode_cfg: dict
    scene_cfg_dir: Path

    @classmethod
    def from_yaml(scene_path, frame_path, episode_path) -> EpisodeConfig: ...
```

```python
def run_episode(
    cfg: EpisodeConfig,
    *,
    policy_factory: Callable[[Point["ned"], dict], Policy],
    renderer: Optional[Any] = None,
    detector_factory: Optional[Callable[[FrameGraph, dict], Any]] = None,
    recovery_factory: Optional[Callable[[FrameGraph, dict], Any]] = None,
    recovery_triggers: Optional[Any] = None,
    perturbations_factory: Optional[Callable[[FrameGraph, dict], Any]] = None,
    rng: "np.random.Generator | None" = None,
    initial_state_override: Optional[DroneState] = None,
    perturbation_overrides: Optional[dict] = None,
) -> FalsificationEpisode: ...
```

## What happens

1. Build the `FrameGraph` from the scene YAML.
2. Resolve the initial `DroneState`. If `initial_state_override` is set (trial-card / replay path), it is used verbatim (must be NED); otherwise the start position is read from `episode_cfg`, converted MOCAP→NED, and given zero velocity.
3. Convert the goal from MOCAP to NED — passed into the policy factory.
4. Construct the policy. Construct the `SensorRig` to cover its `required_modalities`.
5. Construct the `FailureDetector`, recovery planner, and `PerturbationSuite` from their respective factories (each optional). `perturbation_overrides` (when set) lets a trial card supply exact pre-sampled deltas instead of letting the suite resample.
6. Build `SimulatorConfig` from `episode_cfg["hz" | "horizon_s" | "policy_hz" | "chunk_steps"]`. For VLA-style chunked rollout set `chunk_steps = actions_per_chunk` so the simulator re-queries only after a chunk is consumed.
7. Reset the simulator. Roll out under the detector. On failure, the detector returns a `FailureRecord` with both `last_safe_state` and the full `safe_history`.
8. If recovery is wired and the failure type is in `recovery_triggers`, the orchestrator picks a seed via `sample_recovery_seed(safe_history, failure_type, rng, ...)` and the planner produces a `Trajectory[ned]` from `seed → goal`. The picked-seed metadata is persisted under `FalsificationEpisode.metadata["recovery_seed"]`.
9. Bundle everything into a `FalsificationEpisode`.
10. **Always** (success or exception) call `policy.close()` if the policy
    defines it — the policy object never escapes `run_episode`, and
    `PiGatewayPolicy`'s gateway WS / RTC threads are non-daemon: leaking
    them keeps the interpreter alive after a campaign script's `main()`
    returns (the "script never exits" hang fixed 2026-06-12).

## Why factories instead of objects

Each `*_factory` is a callable so the orchestrator can supply the
`FrameGraph` and the relevant config slice at construction time. This keeps
all dependency injection on the call site, avoids importing heavy modules
(torch, FiGS) when they're not needed, and makes the orchestrator easy to
unit-test with stub factories.
