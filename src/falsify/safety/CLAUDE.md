# `falsify.safety/` — failure detection

**Status:** done.

## Contract

```python
class SafetyCriterion(ABC):
    operates_in_frame: str = "ned"
    name: str = ""
    def check(state_in_frame: DroneState) -> Optional[Violation]: ...
    def check_with_graph(state, frame_graph) -> Optional[Violation]: ...   # converts then checks
```

```python
class FailureDetector:
    def __init__(criteria, frame_graph): ...
    def update(state, step) -> Optional[FailureRecord]: ...   # voting + last-safe tracking
```

## How a criterion handles frames

Each criterion declares ``operates_in_frame`` (default ``"ned"``). The
detector calls ``check_with_graph(state, frame_graph)`` which converts the
incoming `DroneState.pos` into the criterion's frame before invoking
`check`. Velocity and orientation aren't auto-rotated by the default
helper — criteria that need cross-frame velocity should override
`check_with_graph` (no such criterion exists yet; the bounds/velocity/tilt
criteria all live in NED).

## Last-safe state

`FailureDetector` records the most recent safe state. On first failure, the
emitted `FailureRecord` includes:

- `failure_state` — the offending `DroneState`
- `last_safe_state` — the previous safe `DroneState`
- `last_safe_step` — its index in the rollout
- `description`, `value`, `threshold` — diagnostics

`last_safe_state` is what the recovery planner consumes.

## Adding a criterion

Subclass `SafetyCriterion`, set `name`/`operates_in_frame`, implement
`check(state_in_frame) -> Optional[Violation]`. Register it in the
orchestrator's detector factory (or pass it directly when constructing
`FailureDetector`). If the criterion is collision-aware (proximity to
Gaussians, etc.), override `check_with_graph` to consult the gsplat or
other side-info that needs the unconverted state.
