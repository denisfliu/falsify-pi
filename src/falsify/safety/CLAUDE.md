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

## Last-safe state

`FailureDetector` records the most recent safe state. On first failure, the
emitted `FailureRecord` includes:

- `failure_state` — the offending `DroneState`
- `last_safe_state` — the previous safe `DroneState`
- `last_safe_step` — its index in the rollout
- `description`, `value`, `threshold` — diagnostics

`last_safe_state` is what the recovery planner consumes.

## Failure taxonomy

`FailureType` enumerates the modes the detector can emit:

| Type | Emitted by | Meaning |
|------|------------|---------|
| `OUT_OF_BOUNDS` | `BoundsCriterion` | drone outside the declared box |
| `EXCESSIVE_VELOCITY` | `VelocityCriterion` | ‖v‖ exceeds threshold |
| `EXCESSIVE_TILT` | `TiltCriterion` | body-z deviation from world-z exceeds threshold |
| `COLLISION_GATE` | `PointCloudCollisionCriterion` | drone OBB intersects a `"gate"`-labelled point |
| `COLLISION_OTHER` | `PointCloudCollisionCriterion` | drone OBB intersects a non-gate-labelled point |
| `MISS_GATE` | `MissGateCriterion` | drone crossed the gate plane outside the aperture rectangle |
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
