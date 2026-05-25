# `falsify.recovery/` — recovery planners

**Status:** done.

Three planner backends ship in this subpackage. All produce a
`RecoveryResult.trajectory` tagged `"ned"` regardless of what frame the
underlying solver lives in:

| Planner | Where | When to use |
|---------|-------|-------------|
| `CoursedMpcPlanner` | `coursed_mpc.py` | **Default in falsification runs.** Re-runs `planning.plan_mpc` over the same Course YAML the rollout was trying to fly, starting from the chosen `recovery_seed` state. Dynamically feasible. |
| `SplatNavPlanner`   | `planner.py` | A*+spline (collision-aware) over the gsplat. NED in, NED out; lazily imports `splatplan`. |
| `SplatNavMpcPlanner` | `splatnav_mpc.py` | A* (from `splatplan`) → waypoint list → `plan_mpc` tracker. Shares its loaded gsplat pipeline with the rest of the run. Used by `scripts/recovery/collect_recovery_trajectories.py`. |

`CoursedMpcPlanner` is what every `configs/recovery/*_mpc.yaml` resolves to;
`SplatNavPlanner` is the canonical fallback when no course YAML is in scope.

## Public API (SplatNavPlanner)

```python
class SplatNavPlanner:
    def __init__(cfg: RecoveryConfig, frame_graph, *, backend=None, gsplat_config_path=None, horizon_s, hz): ...
    def plan(start: Point["ned"], goal: Point["ned"]) -> RecoveryResult: ...   # Trajectory["ned"]
```

The wrapper is the **single** translation site that bridges public-API NED
and SplatPlan's NS-internal frame.

## Public API (CoursedMpcPlanner)

```python
class CoursedMpcPlanner:
    def __init__(cfg: CoursedMpcConfig, frame_graph): ...
    def plan(start: DroneState, goal: Optional[Point["ned"]] = None) -> RecoveryResult: ...
```

`cfg.course_yaml_path` is the course the original rollout was tracking;
the planner re-solves the MPC with `start_state_ned` set from the recovery
seed's `DroneState`. Returns a `Trajectory[ned]` ready for the simulator
or training-data exporter.

## Public API (SplatNavMpcPlanner)

Hybrid that runs splatplan A* in NS, lifts the path back to NED, then
hands the waypoint list to `plan_mpc` for a dynamically-feasible refit.
Constructed in scripts that already hold a loaded gsplat pipeline so the
30 s load isn't paid twice. See `scripts/recovery/collect_recovery_trajectories.py`.

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

## Replanning seed sampling (`seed_sampling.py`)

**TL;DR for future Claudes.** When a falsification episode fails, the
recovery planner needs *one* `DroneState` to plan **from**. We don't
always want `last_safe_state` — it's a poor seed for failure modes
where the whole approach drifted (miss-gate, collision with the table,
out-of-bounds), and equally a poor seed for `GOAL_NOT_REACHED` where
the drone already crossed the gate and replanning from before the gate
would just re-fly what worked. `sample_recovery_seed` is the single
function that picks the right seed for each failure mode.

### Public API

```python
sample_recovery_seed(
    safe_history: list[tuple[int, DroneState]],
    failure_type: Optional[FailureType],
    rng: np.random.Generator,
    *,
    transit_time: Optional[float] = None,           # back-compat only — no longer alters sampling
    gate_1_transit_time: Optional[float] = None,    # back-compat only
    failure_phase: Optional[str] = None,            # back-compat only
) -> tuple[int, DroneState]
```

**Phase scoping is now the caller's responsibility.** The legacy
`transit_time` / `gate_1_transit_time` / `failure_phase` kwargs are
accepted for back-compat but no longer alter the draw — callers should
pre-filter `safe_history` to the desired phase window before invoking.

`safe_history` is the ordered list of `(step, DroneState)` pairs the
`FailureDetector` saw before firing — it's published on
`FailureRecord.safe_history`. The function returns `(step, state)` so
callers can log which step they replanned from.

### Bias table

The function only chooses a Beta shape; the **scope** of `safe_history`
is the caller's responsibility (see "Phase scoping" below).

| Failure type      | Beta(α, β) | Mean idx | Why                                                          |
|-------------------|------------|----------|--------------------------------------------------------------|
| `COLLISION_GATE`  | (3, 1)     | 0.75·n   | Gate clip ⇒ approach was almost right ⇒ restart **near gate** |
| anything else     | (1, 3)     | 0.25·n   | Conservative default — wholesale re-plan with **runway**     |

### Phase scoping is the caller's job

Earlier versions of `sample_recovery_seed` consulted `transit_time` /
`gate_1_transit_time` / `failure_phase` to scope `safe_history` to a
window (e.g. "post-gate-1 only"). That responsibility has moved to the
caller — by the time you call the sampler, `safe_history` should already
be filtered to the phase you want to draw from. The legacy kwargs are
still accepted for back-compat but no longer alter sampling.

`trim_tail: int = 0` is the one in-sampler scoping knob: drops the last
`trim_tail` entries before drawing. Used by the recovery collector — the
runtime collision criterion checks against the drone OBB only, so the
last few "safe" entries can still be spatially right next to the
obstacle. SplatPlan inflates Gaussians by the drone clearance radius
when voxelising, which makes those tail entries unreachable starts.
Trimming the tail buys the spatial margin back.

**Fallbacks** (all degrade silently rather than raise):

- `safe_history` empty ⇒ orchestrator skips the sampler entirely and
  uses `failure.last_safe_state` directly (failure on step 0).
- `safe_history` singleton ⇒ that one state is returned.

### Data flow at a glance

```
FailureDetector              Orchestrator                          sample_recovery_seed
───────────────              ────────────                          ────────────────────
on failure firing:           after rollout, if failure:            picks (step, state)
  FailureRecord(               filter safe_history to the desired    by Beta(α, β) over
    ...,                       phase window (e.g. post-gate-1 only)  the caller-supplied
    safe_history=...,          seed_step, seed_state                 candidates
    last_safe_state=...,         = sample_recovery_seed(
  )                                filtered_history,
                                   failure_type,
                                   rng,
                                   trim_tail=...,
                                 )
```

### Persisted metadata

The orchestrator writes the sampled-seed details to
`FalsificationEpisode.metadata["recovery_seed"]`, which the CLIs
serialise into `episode_summary.json`:

```json
"recovery_seed": {
  "step": 47,                  // step index of the chosen safe state
  "bias": "early" | "late",    // string label from `bias_for(failure_type)`
  "n_safe": 80,                // total safe states the detector saw
  "transit_time": 5.0,         // null unless GOAL_NOT_REACHED
  "n_post_transit": 33         // null unless transit_time is set
}
```

These five fields are the contract — keep them in sync if you add new
fields to `sample_recovery_seed` or change its return shape.

### Adding a new failure-type bias

1. Define the new `FailureType` in `falsify.safety.records`.
2. Decide its bias intent. Add it to `_BIAS_LATE` in
   `seed_sampling.py` if you want Beta(3, 1); otherwise it defaults
   to Beta(1, 3) (bias-early).
3. If the new failure type needs *scoped* sampling (analogous to
   `GOAL_NOT_REACHED`'s post-transit window), add a branch inside
   `sample_recovery_seed` that filters `safe_history` before the Beta
   draw. Pass the scoping parameter as a new kwarg, default `None`,
   so legacy callers don't break.
4. If the scoping parameter is derived from a criterion's internal
   state, stamp it into `Violation.extra` from the criterion and read
   it via `failure.extra` in the orchestrator. **Do not** plumb new
   detector → orchestrator channels — `Violation.extra → FailureRecord.extra`
   is the established passthrough.
5. Update the bias / scope table above. Add a row to
   `tests/test_recovery_seed_sampling.py`.

### Don't confuse `last_safe_state` vs sampled seed

- `failure.last_safe_state` — the **single** most recent safe
  `DroneState` (the one immediately before the failure step). Always
  populated. Used by the orchestrator as a hard fallback when
  `safe_history` is empty.
- `sample_recovery_seed(...)` — picks **one of many** safe states
  with a failure-type-aware bias / scope. Preferred for replanning.

Older callers (and the simpler stub recovery path) still use
`last_safe_state`; the orchestrator's normal flow uses the sampler.

## MPC speed

`plan_mpc` defaults to `use_rti=True` (SQP-RTI: one SQP iteration per
tick). Measured against the previous full-SQP path on a left_gate
recovery: 7.07 s → 2.23 s (~3x), trajectory bit-identical to within
1e-7 m. Disable (`use_rti=False`) only if a future trajectory needs
multi-iteration convergence per tick.
