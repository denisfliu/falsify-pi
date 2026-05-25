# `falsify.safety/` — failure detection

**Status:** done.

## Contract

```python
class SafetyCriterion(ABC):
    operates_in_frame: str = "ned"
    name: str = ""
    def check(state_in_frame: DroneState) -> Optional[Violation]: ...
    def check_with_graph(state, frame_graph) -> Optional[Violation]: ...   # converts then checks
    def reset() -> None: ...   # default no-op; override if you carry per-episode state
```

```python
@dataclass
class Violation:
    description: str
    value: float
    threshold: float
    failure_type: Optional[FailureType] = None    # override the name-based lookup
    extra: dict = field(default_factory=dict)     # merged into FailureRecord.extra
```

```python
class FailureDetector:
    def __init__(criteria, frame_graph): ...
    def update(state, step) -> Optional[FailureRecord]: ...   # voting + last-safe tracking
    def reset() -> None: ...   # also resets per-criterion state
```

## How a criterion handles frames

Each criterion declares ``operates_in_frame`` (default ``"ned"``). The
detector calls ``check_with_graph(state, frame_graph)`` which converts the
incoming `DroneState.pos` into the criterion's frame before invoking
`check`. Velocity and orientation aren't auto-rotated by the default
helper — criteria that need cross-frame velocity should override
`check_with_graph` (no such criterion exists yet; the bounds/velocity/tilt
criteria all live in NED).

## Last-safe state + safe-state history

`FailureDetector` records both the most recent safe state and the full
ordered history of safe `(step, DroneState)` pairs for the episode. On
first failure, the emitted `FailureRecord` includes:

- `failure_state` — the offending `DroneState`
- `last_safe_state` / `last_safe_step` — the previous safe state and its index
- `safe_history: list[(step, DroneState)]` — every safe step the detector
  saw before the failure, in chronological order
- `description`, `value`, `threshold` — diagnostics

`last_safe_state` is a natural single-state seed for recovery. The full
`safe_history` is consumed by
`falsify.recovery.seed_sampling.sample_recovery_seed` — see
`src/falsify/recovery/CLAUDE.md§ Replanning seed sampling` for the
full contract (failure-type bias, post-transit scoping for
`GOAL_NOT_REACHED`, persisted metadata). Quick summary:

| Failure type                                            | Sampling scope          | Bias        |
|--------------------------------------------------------|-------------------------|-------------|
| `COLLISION_GATE`                                       | full `safe_history`     | Beta(3, 1)  |
| `MISS_GATE`, `COLLISION_OTHER`, `OUT_OF_BOUNDS`        | full `safe_history`     | Beta(1, 3)  |
| `GOAL_NOT_REACHED`                                     | states with `state.t ≥ transit_time` | Beta(1, 3) |

For `GOAL_NOT_REACHED`, `MissGateCriterion` records the time of
successful transit in `_transit_t` and stamps it into
`Violation.extra["transit_time"]`; the detector merges it into
`FailureRecord.extra`; the orchestrator forwards it as the
`transit_time=` kwarg to the sampler.

## Failure taxonomy

`FailureType` enumerates the modes the detector can emit:

| Type | Emitted by | Meaning |
|------|------------|---------|
| `OUT_OF_BOUNDS` | `BoundsCriterion` | drone outside the declared box |
| `EXCESSIVE_VELOCITY` | `VelocityCriterion` | ‖v‖ exceeds threshold |
| `EXCESSIVE_TILT` | `TiltCriterion` | body-z deviation from world-z exceeds threshold |
| `COLLISION_GATE` | `PointCloudCollisionCriterion` | drone OBB intersects a `"gate"`-labelled point |
| `COLLISION_OTHER` | `PointCloudCollisionCriterion` | drone OBB intersects a non-gate-labelled point |
| `MISS_GATE` | `MissGateCriterion` | drone failed to navigate the gate. **Legacy mode** (default): three sub-modes — (a) plane-crossing outside aperture (BUGGY for center_gate — disabled in eval_stop_mode), (b) reached goal proximity without transit, (c) no progress toward aperture. **eval_stop_mode**: only emitted as a no-progress stop signal; final MISS_GATE classification is decided post-hoc. |
| `GOAL_NOT_REACHED` | `MissGateCriterion` | drone *did* transit the aperture but then failed to reach `goal_position` — no progress toward goal in `min_progress_window_s` seconds. The "post-gate hover failed" case. |
| `GOAL_REACHED` | `MissGateCriterion` (eval_stop_mode only) | drone is inside the goal-tolerance region (box `goal ± goal_tolerance_half_extents` when set, sphere `goal_tolerance_m` otherwise). This is a **success stop signal**, not a failure; the post-hoc classifier resolves SUCCESS vs SKIPPED_GATE vs MISS_GATE (the last via the directional check). Never in `recovery_triggers`. |
| `PROXIMITY_COLLISION` | (reserved for future scalar-distance criterion) | |
| `CUSTOM` | anything without a name-mapping | fallback |

Gate-collision takes priority over other-collision: if the drone OBB
contains both gate- and table-labelled points in the same step, the
violation is reported as `COLLISION_GATE`.

## Collision: drone-as-OBB

`DroneBody(half_extents, center_offset_body=0)` declares a rectangular
prism in body **FRD** frame (x forward, y right, z down). The OBB
centre in world is ``state.pos + R(state.quat_xyzw) · center_offset``;
its half-extents along the body axes are read directly from the config.

`PointCloudCollisionCriterion(drone_body, labeled_clouds)` accepts a
dict of label → ``(N, 3)`` point clouds, all in the criterion's
``operates_in_frame`` (typically NED). Per step it sphere-culls the
points to those within ``drone_body.bounding_radius`` of the body
origin, then runs the exact axis-aligned containment test in body
frame. The smoke-test factory loads these from the scene YAML's
``scene_objects:`` block (extracted per-object PLYs) and converts each
to NED once at factory build time.

The classification (gate vs other) lives in the `Violation.failure_type`
field — the detector reads that directly, bypassing the
`criterion.name → FailureType` lookup so one criterion can emit multiple
failure types.

## Miss-gate: aperture-crossing

`MissGateCriterion(corners, frame_name, margin_m=0.0)` takes 4 corners
ordered to trace the aperture rectangle (so ``corners[1]-corners[0]``
and ``corners[3]-corners[0]`` are orthogonal adjacent edges; the
constructor enforces both orthogonality and coplanarity). From these
it derives the plane normal, in-plane axes ``(u, v)``, half-widths
``(hu, hv)``, and centre.

The criterion overrides `check_with_graph` because it needs the
previous state alongside the current one — the segment ``prev → current``
is intersected with the plane (signed distance changes sign); if a
crossing exists, the intersection's ``(u, v)`` coordinates relative to
centre are tested against the half-widths. Outside ⇒ `MISS_GATE`.
Inside ⇒ mark `_transited` so re-crossings (e.g. recovery looping back
through) don't re-fire. ``margin_m`` shrinks the aperture inwards for a
stricter "well-centred passage" check.

### `eval_stop_mode` (post-hoc classification path)

For evaluation campaigns the in-flight plane-cross-outside-aperture
check was firing false-positives (center_gate's natural approach arcs
clipped the strict rectangle). Pass ``eval_stop_mode=True`` (set in
``configs/safety/*.yaml::miss_gate.eval_stop_mode``) to switch the
criterion into a **stop-signal-only** mode:

- Mode (a) plane-crossing-outside-aperture is **skipped entirely**. No
  MISS_GATE emitted at runtime for that reason.
- On goal proximity, fire `FailureType.GOAL_REACHED` regardless of
  whether the drone's trajectory crossed the plane inside the
  rectangle. This is a **success stop signal**, not a failure. The
  "in proximity" test is either the box `|xyz - goal| ≤
  goal_tolerance_half_extents` (when configured — current default for
  the gate scenes) or the legacy Euclidean sphere `||xyz - goal|| ≤
  goal_tolerance_m`. Box wins when set; sphere is still honoured by
  the no-progress mode (b)/(d) so the legacy field's meaning doesn't
  silently change.
- Stuck check still fires (`MISS_GATE` pre-transit / `GOAL_NOT_REACHED`
  post-transit), but it's just a stop signal — the post-hoc classifier
  reclassifies based on the gate's MOCAP AABB.

Final SUCCESS / MISS_GATE / GOAL_NOT_REACHED / SKIPPED_GATE classification
happens in `falsify.safety.posthoc.classify_trajectory_posthoc`, which
walks the trajectory's MOCAP positions against
``scene_cfg.gate_region.aabb_min/max`` (with any
``GateRigidPerturbation`` Δ applied via the same MOCAP rigid transform).
The runtime `failure.failure_type` is *only* used to forward
COLLISION_GATE / COLLISION_OTHER / OUT_OF_BOUNDS / EXCESSIVE_* verbatim;
everything else is reclassified.

## Post-hoc classification (`posthoc.py`)

`classify_trajectory_posthoc(positions_mocap, scene_cfg, ...)` returns a
`{outcome, transited, first_inside_step, last_inside_step,
n_states_inside, aabb_mocap}` dict. Outcome is one of:

| Outcome              | Condition                                               |
|----------------------|--------------------------------------------------------|
| `SUCCESS`            | runtime fired `GOAL_REACHED` + trajectory entered AABB + (when configured) at least one correct-direction aperture crossing and zero wrong-direction crossings |
| `MISS_GATE`          | rollout stopped (stuck / timeout / OOB) + never inside; **also** the demotion target when directional check fails (wrong-direction or no correct crossing) |
| `GOAL_NOT_REACHED`   | rollout stopped + trajectory was inside the AABB        |
| `COLLISION_GATE`     | runtime fired collision, gate-labeled                   |
| `COLLISION_OTHER`    | runtime fired collision, other-labeled                  |
| `OUT_OF_BOUNDS`      | runtime fired bounds                                    |
| `EXCESSIVE_VELOCITY` | runtime fired velocity                                  |
| `EXCESSIVE_TILT`     | runtime fired tilt                                      |
| `ERROR`              | orchestrator raised                                     |

`SKIPPED_GATE` is retained as a back-compat alias of `MISS_GATE` and
no longer appears as a distinct outcome — both mean "drone never went
through the gate" under the post-hoc rule.

The campaign runner (`scripts/eval/run_eval_campaign.py`) writes
`posthoc_outcome` + `transited` + transit-step indices into each trial's
`episode_summary.json` and uses `posthoc_outcome == "SUCCESS"` as the
single source of truth for `n_succeeded` in `campaign_summary.json`.
`by_outcome` is the authoritative histogram; `by_failure_type` (the
runtime stop-signal histogram) is kept as a diagnostic alongside.

The gate AABB consumed by the classifier lives on each scene YAML under
`gate_region.aabb_frame` + `aabb_min` + `aabb_max` (MOCAP only today).
The same block is used by `GateRigidPerturbation` for Gaussian selection.

### Directional gate-transit check (`expected_dy_sign`)

When the trial card's `scene_key` ends in `_from_left` or `_from_right`,
`run_eval_campaign.py` derives `expected_dy_sign` and passes it to
`classify_trajectory_posthoc`:

| Suffix      | `expected_dy_sign` | Interpretation              |
|-------------|--------------------|-----------------------------|
| `_from_left`  | `-1` | drone must cross the gate plane in -y (mocap) |
| `_from_right` | `+1` | drone must cross the gate plane in +y (mocap) |

`check_directional_transit` walks the trajectory segment-by-segment,
detects crossings of the gate's mid-y plane, checks the interpolated
(x, z) at each crossing against the AABB extents, and records the sign
of `dy` for crossings inside the aperture. Crossings outside the
aperture (e.g. the drone dipping below the gate plane upstream of the
gate, then curving around to enter the AABB from behind) are ignored.

The post-hoc classifier then:

- demotes a runtime `GOAL_REACHED` to `MISS_GATE` if there are zero
  correct-direction crossings or any wrong-direction crossing;
- on no-progress / timeout, classifies `MISS_GATE` if no correct
  crossing, `GOAL_NOT_REACHED` otherwise.

Scenes without the `_from_*` suffix (e.g. `left_gate`, `right_gate`)
skip the directional check and use the legacy "any AABB touch counts
as transit" rule. To re-apply this check to already-captured campaigns
without re-rolling, use `scripts/eval/reclassify_campaign.py` — it walks the
trial dirs, re-classifies, and rewrites the per-trial summaries +
`campaign_summary.json` (with `*.json.bak` backups on first run).

## Adding a criterion

Subclass `SafetyCriterion`, set `name`/`operates_in_frame`, implement
`check(state_in_frame) -> Optional[Violation]`. Register it in the
orchestrator's detector factory (or pass it directly when constructing
`FailureDetector`). If the criterion is collision-aware (proximity to
Gaussians, etc.), override `check_with_graph` to consult the gsplat or
other side-info that needs the unconverted state. If a single criterion
needs to emit different `FailureType`s by case, set
`Violation.failure_type` and the detector will honour it. If the
criterion accumulates per-episode state (previous samples, transition
counts, etc.), override `reset()` so the detector can clear it between
episodes.
