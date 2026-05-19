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
    transit_time: Optional[float] = None,
) -> tuple[int, DroneState]
```

`safe_history` is the ordered list of `(step, DroneState)` pairs the
`FailureDetector` saw before firing — it's published on
`FailureRecord.safe_history`. The function returns `(step, state)` so
callers can log which step they replanned from.

### Bias / scope table

| Failure type                             | Scope of draw                          | Beta(α, β) | Mean idx | Why                                                          |
|------------------------------------------|----------------------------------------|------------|----------|--------------------------------------------------------------|
| `COLLISION_GATE`                         | full `safe_history`                    | (3, 1)     | 0.75·n   | Gate clip ⇒ approach was almost right ⇒ restart **near gate** |
| `MISS_GATE`                              | full `safe_history`                    | (1, 3)     | 0.25·n   | Approach drifted ⇒ restart with **runway**                    |
| `COLLISION_OTHER`                        | full `safe_history`                    | (1, 3)     | 0.25·n   | Hit something off-path ⇒ wholesale re-plan                    |
| `OUT_OF_BOUNDS`                          | full `safe_history`                    | (1, 3)     | 0.25·n   | Diverged ⇒ wholesale re-plan                                  |
| `GOAL_NOT_REACHED`                       | states with `state.t ≥ transit_time`   | (1, 3)     | post-transit start + 0.25·m | Drone *did* transit; replan **after the gate**, biased toward the earliest post-transit state |
| anything else / `None`                   | full `safe_history`                    | (1, 3)     | 0.25·n   | Conservative default — only push *toward* failure when sure   |

### `GOAL_NOT_REACHED` post-transit scoping — the long form

`MissGateCriterion` records `state.t` the moment `_transited` flips
`True` (gate-plane crossing falls inside the aperture rectangle). It
stamps that value into `Violation.extra["transit_time"]` when the
post-transit no-progress check fires. `FailureDetector` merges
`Violation.extra` into `FailureRecord.extra`. The orchestrator pulls
`transit_time = trace.failure.extra.get("transit_time")` and forwards
it to `sample_recovery_seed(transit_time=…)`.

The sampler then filters `safe_history` to states with
`state.t >= transit_time` and applies Beta(1, 3) **inside that
window**. The user's design intent: *closer emphasis on the earlier
post-transit states* — i.e. the seed should sit just after the gate
crossing, not deep into the post-transit hover where things had
already gone wrong. Beta(1, 3) inside the window gives mean idx 0.25·m
(m = post-transit count), so the expected seed is one quarter into
the post-transit run, biased early.

**Fallbacks** (all degrade silently rather than raise):

- `safe_history` empty ⇒ orchestrator skips the sampler entirely and
  uses `failure.last_safe_state` directly (failure on step 0).
- `safe_history` singleton ⇒ that one state is returned.
- `transit_time` not provided / failure isn't `GOAL_NOT_REACHED` ⇒
  scoping is skipped; full history is used with the type-driven bias.
- No safe state has `state.t >= transit_time` (failure fired
  immediately after transit before a single safe step was recorded) ⇒
  falls back to the full history with bias-early.

### Data flow at a glance

```
MissGateCriterion       FailureDetector              Orchestrator                       sample_recovery_seed
─────────────────       ───────────────              ────────────                       ────────────────────
on transit:             on failure firing:           after rollout, if failure:         picks (step, state)
  self._transit_t         FailureRecord(             transit_time = failure.extra        from safe_history /
    = state.t              ...,                        .get("transit_time")               post-transit window
                           safe_history=...,         seed_step, seed_state               with bias from table
on stuck-after-transit:    extra={                     = sample_recovery_seed(            above
  Violation(                 ...mode...,                  history,
    failure_type=              transit_time:             failure_type,
      GOAL_NOT_REACHED,        <copied from              rng,
    extra={                    Violation.extra>          transit_time=<…>,
      transit_time:          },                        )
        self._transit_t,    )
      ...,
    },
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
