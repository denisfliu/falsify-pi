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
    renderer: Optional[Callable] = None,
    detector_factory: Optional[Callable[[FrameGraph, dict], FailureDetector]] = None,
    recovery_factory: Optional[Callable[[FrameGraph, dict], SplatNavPlanner]] = None,
) -> FalsificationEpisode: ...
```

## What happens

1. Build the `FrameGraph` from the scene YAML.
2. Convert the start position from MOCAP to NED. Build the initial `DroneState`.
3. Convert the goal from MOCAP to NED — passed into the policy factory.
4. Construct the policy. Construct the `SensorRig` to cover its `required_modalities`.
5. Construct the `FailureDetector` (optional) and the `SplatNavPlanner` (optional).
6. Reset the simulator. Roll out under the detector. On failure, the detector returns a `FailureRecord` with the `last_safe_state`.
7. If recovery is wired and failure fired, the planner produces a `Trajectory[ned]` from `last_safe → goal`.
8. Bundle everything into a `FalsificationEpisode`.

## Why factories instead of objects

Each `*_factory` is a callable so the orchestrator can supply the
`FrameGraph` and the relevant config slice at construction time. This keeps
all dependency injection on the call site, avoids importing heavy modules
(torch, FiGS) when they're not needed, and makes the orchestrator easy to
unit-test with stub factories.
