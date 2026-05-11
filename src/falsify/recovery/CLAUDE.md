# `falsify.recovery/` — SplatNav recovery planner

**Status:** done.

## Public API

```python
class SplatNavPlanner:
    def __init__(cfg: RecoveryConfig, frame_graph, *, backend=None, gsplat_config_path=None, horizon_s, hz): ...
    def plan(start: Point["ned"], goal: Point["ned"]) -> RecoveryResult: ...   # Trajectory["ned"]
```

`RecoveryResult.trajectory.frame.name == "ned"` always — the wrapper is the
**single** translation site that bridges public-API NED and SplatPlan's
NS-internal frame.

## Frame contract

- **In:** NED `Point`s for start and goal; NED bounds in `RecoveryConfig`.
- **Inside:** the wrapper converts to NS via the active `FrameGraph` once,
  calls `backend.generate_path(x0_ns, xf_ns)`, and converts the returned
  waypoint sequence back to NED.
- **Out:** a frame-tagged `Trajectory` ready for the simulator to follow.

Bounds in `RecoveryConfig` are declared in NED. The wrapper composes them
through the `FrameGraph` and takes element-wise min/max so that frames
which flip signs (e.g. perm5 NED→MOCAP) still yield a valid axis-aligned
box in NS.

## Backend protocol

```python
class PlannerBackend(Protocol):
    def generate_path(x0_ns: np.ndarray, xf_ns: np.ndarray) -> np.ndarray: ...
```

The real backend wraps `splatplan.splatplan.SplatPlan` and is lazy-loaded
in `_make_splatnav_backend` — torch/splatnav imports never run on machines
that don't need them. Tests pass a stub backend (a straight line of N
waypoints) so the recovery contract is testable without CUDA.

The smoke-test CLI also supports `--stub-recovery` for the same reason.

## Adding a new backend

Implement the `PlannerBackend` protocol and pass it to the wrapper as
``backend=``. The wrapper is otherwise unchanged. This is how future
planners (RRT, MPC-based collision avoidance, etc.) can be slotted in
without touching the orchestrator.
