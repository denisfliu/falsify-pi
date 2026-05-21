"""Post-hoc trajectory classification.

The runtime safety stack (criteria.py + detector.py) is responsible only
for STOPPING the rollout — collision, goal-proximity, velocity/tilt/bounds,
no-progress. Final per-trial outcome classification (`SUCCESS` /
`MISS_GATE` / `GOAL_NOT_REACHED` / `SKIPPED_GATE` / `COLLISION_*` / etc.)
is decided here by walking the trial's trajectory states against the
scene's published gate AABB (`scene_cfg.gate_region`).

Why post-hoc?
-------------
The previous design used `MissGateCriterion`'s in-flight plane-crossing
check (mode (a)) as the MISS_GATE classifier. That fired false positives
in center_gate when the policy's natural arc clipped the aperture plane
outside the strict rectangle. The fix: drop the in-flight rectangle check
(it lives behind `eval_stop_mode=False` for legacy callers), let the
rollout run to one of its real stop conditions (collision / goal-proximity
/ stuck / timeout), then decide MISS_GATE here using the gate's MOCAP
bounding box.

Outcome taxonomy
----------------
- ``SUCCESS``         — runtime fired `GOAL_REACHED`. Under the current
                        `MissGateCriterion` wiring, GOAL_REACHED is gated
                        on the runtime AABB-transit latch — it cannot
                        fire until the drone has been inside the gate's
                        MOCAP AABB at least once. So GOAL_REACHED here
                        implies transit, and post-hoc rubber-stamps it.
- ``MISS_GATE``       — rollout stopped (stuck / timeout / OOB / sim
                        instability) without any trajectory state inside
                        the gate's MOCAP AABB. Subsumes the previous
                        SKIPPED_GATE category — both mean "drone never
                        went through the gate"; under the user's policy
                        intent that's one bucket.
- ``GOAL_NOT_REACHED``— rollout stopped (stuck / timeout) but at least
                        one trajectory state WAS inside the gate AABB
                        (drone transited, then failed to reach goal).
- ``COLLISION_GATE``  — runtime fired collision against a `gate`-labeled
                        point cloud. Verbatim.
- ``COLLISION_OTHER`` — runtime fired collision against any other cloud.
                        Verbatim.
- ``OUT_OF_BOUNDS``   — runtime fired bounds. Verbatim.
- ``EXCESSIVE_VELOCITY`` / ``EXCESSIVE_TILT`` — sim instabilities.
                        Verbatim. Should not be falsification targets.
- ``ERROR``           — orchestrator raised. Verbatim.

The classifier returns one of these strings (not a `FailureType` enum,
since the success cases don't fit cleanly in that enum). Callers that
want the FailureType for downstream filtering can re-look up via
``OUTCOME_TO_FAILURE_TYPE``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .records import FailureType


# Outcome strings. Public — the per-trial summary records one of these
# under `posthoc_outcome` and the campaign aggregate uses them as
# histogram keys.
SUCCESS            = "SUCCESS"
MISS_GATE          = "MISS_GATE"
# Back-compat alias — retained so old campaign summaries that recorded
# SKIPPED_GATE still resolve; callers should treat both as MISS_GATE.
SKIPPED_GATE       = MISS_GATE
GOAL_NOT_REACHED   = "GOAL_NOT_REACHED"
COLLISION_GATE     = "COLLISION_GATE"
COLLISION_OTHER    = "COLLISION_OTHER"
OUT_OF_BOUNDS      = "OUT_OF_BOUNDS"
EXCESSIVE_VELOCITY = "EXCESSIVE_VELOCITY"
EXCESSIVE_TILT     = "EXCESSIVE_TILT"
ERROR              = "ERROR"


_VERBATIM_FROM_FAILURE_TYPE = {
    FailureType.COLLISION_GATE:     COLLISION_GATE,
    FailureType.COLLISION_OTHER:    COLLISION_OTHER,
    FailureType.OUT_OF_BOUNDS:      OUT_OF_BOUNDS,
    FailureType.EXCESSIVE_VELOCITY: EXCESSIVE_VELOCITY,
    FailureType.EXCESSIVE_TILT:     EXCESSIVE_TILT,
}


@dataclass
class TransitResult:
    """Output of the gate-AABB containment scan."""

    transited: bool
    n_states_inside_aabb: int
    first_inside_step: Optional[int]
    last_inside_step: Optional[int]
    aabb_min_mocap: np.ndarray
    aabb_max_mocap: np.ndarray


@dataclass
class DirectionalTransitResult:
    """Output of the signed gate-plane crossing scan.

    A "crossing" is a step where consecutive trajectory states sit on
    opposite sides of the gate's mid-y plane AND the interpolated (x, z)
    of the intersection lies inside the gate AABB's x/z extents — i.e.
    the drone actually passed through the gate aperture region, not
    around it.
    """

    correct_crossings: int            # crossings whose dy sign matches expected
    wrong_crossings: int              # crossings whose dy sign opposes expected
    first_correct_step: Optional[int]
    first_wrong_step: Optional[int]
    expected_dy_sign: int             # +1 or -1
    gate_plane_y_mocap: float


def _gate_aabb_mocap(
    scene_cfg: dict,
    gate_deltas_mocap: Optional[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve the gate's MOCAP-frame AABB, applying any gate perturbation.

    `gate_deltas_mocap` follows the shape stamped on
    ``FalsificationEpisode.metadata['perturbations']`` by
    `_extract_gate_deltas` (orchestrator) — i.e.
    ``{anchor_mocap, delta_xyz_mocap, delta_yaw_rad}``. The gate Gaussians
    are rigidly translated + yawed about that anchor; we lift the AABB's
    corners through the same transform and rebracket.
    """
    region = scene_cfg.get("gate_region")
    if not region:
        raise ValueError(
            "scene_cfg.gate_region is required for post-hoc transit checks; "
            "see configs/scenes/left_gate.yaml for the canonical layout"
        )
    if region.get("aabb_frame", "mocap") != "mocap":
        raise NotImplementedError(
            "post-hoc transit check assumes gate_region.aabb_frame == mocap"
        )
    aabb_min = np.asarray(region["aabb_min"], dtype=np.float64)
    aabb_max = np.asarray(region["aabb_max"], dtype=np.float64)

    if gate_deltas_mocap is None:
        return aabb_min, aabb_max

    anchor = np.asarray(gate_deltas_mocap["anchor_mocap"], dtype=np.float64)
    dxyz   = np.asarray(gate_deltas_mocap["delta_xyz_mocap"], dtype=np.float64)
    dyaw   = float(gate_deltas_mocap["delta_yaw_rad"])
    c, s = np.cos(dyaw), np.sin(dyaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    # Apply the rigid edit to the 8 corners, then rebracket.
    corners = np.array([
        [aabb_min[0], aabb_min[1], aabb_min[2]],
        [aabb_max[0], aabb_min[1], aabb_min[2]],
        [aabb_min[0], aabb_max[1], aabb_min[2]],
        [aabb_max[0], aabb_max[1], aabb_min[2]],
        [aabb_min[0], aabb_min[1], aabb_max[2]],
        [aabb_max[0], aabb_min[1], aabb_max[2]],
        [aabb_min[0], aabb_max[1], aabb_max[2]],
        [aabb_max[0], aabb_max[1], aabb_max[2]],
    ])
    moved = (corners - anchor) @ R.T + anchor + dxyz
    return moved.min(axis=0), moved.max(axis=0)


def _compositional_phase_from_history(
    positions_mocap: np.ndarray,
    scene_cfg: dict,
) -> Optional[str]:
    """Derive the compositional phase the trajectory ended in by
    replaying ordered AABB-transit latches against the trajectory.

    Returns ``"pre_gate_1"`` / ``"between_gates"`` / ``"post_gate_2"`` or
    ``None`` if the scene doesn't declare per-gate ``gate_regions``.
    Mirrors ``OrderedMissGateCriterion``'s runtime latch logic so the
    post-hoc-derived phase for a COLLISION_GATE trial agrees with the
    runtime-stamped phase for a MISS_GATE trial.
    """
    regions = scene_cfg.get("gate_regions")
    if not regions or len(regions) != 2:
        return None
    aabb_1_min = np.asarray(regions[0]["aabb_min"], dtype=np.float64)
    aabb_1_max = np.asarray(regions[0]["aabb_max"], dtype=np.float64)
    aabb_2_min = np.asarray(regions[1]["aabb_min"], dtype=np.float64)
    aabb_2_max = np.asarray(regions[1]["aabb_max"], dtype=np.float64)
    latched_1 = False
    latched_2 = False
    for p in positions_mocap:
        if not latched_1 and ((p >= aabb_1_min).all() and (p <= aabb_1_max).all()):
            latched_1 = True
        elif latched_1 and not latched_2 and (
            (p >= aabb_2_min).all() and (p <= aabb_2_max).all()
        ):
            latched_2 = True
    if not latched_1:
        return "pre_gate_1"
    if not latched_2:
        return "between_gates"
    return "post_gate_2"


def check_directional_transit(
    positions_mocap: np.ndarray,
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
    *,
    expected_dy_sign: int,
) -> DirectionalTransitResult:
    """Count signed crossings of the gate's mid-y plane inside the AABB.

    The gate plane is at ``(aabb_min.y + aabb_max.y) / 2`` (mocap). A
    "crossing" is a step ``i → i+1`` where the y-component changes sign
    relative to that plane and where the linearly-interpolated (x, z) at
    the crossing lands inside ``[aabb_min.xz, aabb_max.xz]``. The sign
    of the crossing is ``sign(positions[i+1].y - positions[i].y)``.

    ``expected_dy_sign`` must be ``+1`` or ``-1`` — the dy sign that a
    *correct* transit produces for this prompt direction. Crossings that
    match are counted as ``correct_crossings``; opposite-sign crossings
    are counted as ``wrong_crossings``.
    """
    if expected_dy_sign not in (-1, 1):
        raise ValueError(
            f"expected_dy_sign must be -1 or 1, got {expected_dy_sign}"
        )
    if positions_mocap.ndim != 2 or positions_mocap.shape[1] != 3:
        raise ValueError(
            f"positions_mocap must be (N, 3); got {positions_mocap.shape}"
        )

    y_plane = 0.5 * (float(aabb_min[1]) + float(aabb_max[1]))
    x_lo, x_hi = float(aabb_min[0]), float(aabb_max[0])
    z_lo, z_hi = float(aabb_min[2]), float(aabb_max[2])

    correct = 0
    wrong = 0
    first_correct: Optional[int] = None
    first_wrong: Optional[int] = None

    dy_signs = positions_mocap[1:, 1] - positions_mocap[:-1, 1]
    side_prev = positions_mocap[:-1, 1] - y_plane
    side_next = positions_mocap[1:, 1] - y_plane

    # Segment crosses the plane iff signs differ (and neither is exactly
    # zero straddling the plane).
    crosses = (side_prev * side_next) < 0

    for i in np.where(crosses)[0]:
        # Interpolate (x, z) at the crossing.
        sp = side_prev[i]
        sn = side_next[i]
        # Fraction along the segment where y == y_plane.
        # |sp| / (|sp| + |sn|) is the standard linear interp; the sign
        # cancels because sp and sn have opposite sign.
        t = float(sp / (sp - sn))
        x_cross = float(positions_mocap[i, 0] + t * (positions_mocap[i + 1, 0]
                                                     - positions_mocap[i, 0]))
        z_cross = float(positions_mocap[i, 2] + t * (positions_mocap[i + 1, 2]
                                                     - positions_mocap[i, 2]))
        if not (x_lo <= x_cross <= x_hi and z_lo <= z_cross <= z_hi):
            continue  # plane crossing outside the aperture — ignore
        dy = float(dy_signs[i])
        dy_sign = 1 if dy > 0 else (-1 if dy < 0 else 0)
        if dy_sign == expected_dy_sign:
            correct += 1
            if first_correct is None:
                first_correct = int(i)
        elif dy_sign == -expected_dy_sign:
            wrong += 1
            if first_wrong is None:
                first_wrong = int(i)

    return DirectionalTransitResult(
        correct_crossings=correct,
        wrong_crossings=wrong,
        first_correct_step=first_correct,
        first_wrong_step=first_wrong,
        expected_dy_sign=int(expected_dy_sign),
        gate_plane_y_mocap=y_plane,
    )


def check_transit(
    positions_mocap: np.ndarray,
    scene_cfg: dict,
    *,
    gate_deltas_mocap: Optional[dict] = None,
) -> TransitResult:
    """Walk the trajectory and check whether any state's MOCAP position
    fell inside the gate's AABB."""
    if positions_mocap.ndim != 2 or positions_mocap.shape[1] != 3:
        raise ValueError(
            f"positions_mocap must be (N, 3); got {positions_mocap.shape}"
        )
    aabb_min, aabb_max = _gate_aabb_mocap(scene_cfg, gate_deltas_mocap)
    inside = ((positions_mocap >= aabb_min) & (positions_mocap <= aabb_max)).all(axis=1)
    if not inside.any():
        return TransitResult(
            transited=False,
            n_states_inside_aabb=0,
            first_inside_step=None,
            last_inside_step=None,
            aabb_min_mocap=aabb_min,
            aabb_max_mocap=aabb_max,
        )
    where = np.where(inside)[0]
    return TransitResult(
        transited=True,
        n_states_inside_aabb=int(inside.sum()),
        first_inside_step=int(where[0]),
        last_inside_step=int(where[-1]),
        aabb_min_mocap=aabb_min,
        aabb_max_mocap=aabb_max,
    )


def classify_trajectory_posthoc(
    *,
    positions_mocap: np.ndarray,
    scene_cfg: dict,
    runtime_failure_type: Optional[FailureType],
    horizon_steps: int,
    n_states: int,
    gate_deltas_mocap: Optional[dict] = None,
    runtime_error: bool = False,
    expected_dy_sign: Optional[int] = None,
) -> dict:
    """Decide the final outcome of a trial.

    Parameters
    ----------
    positions_mocap
        ``(N, 3)`` array of the rolled-out drone positions in MOCAP. The
        caller is expected to convert from NED via the active FrameGraph
        before handing them in.
    scene_cfg
        Parsed scene YAML — must declare ``gate_region``.
    runtime_failure_type
        The ``FailureType`` recorded by the runtime detector (if any).
        ``None`` means the rollout ran to horizon without a violation.
    horizon_steps
        Total simulator-step budget for the trial (used to decide whether
        we hit the timeout boundary).
    n_states
        Number of states the rollout actually produced. ``n_states >=
        horizon_steps`` indicates timeout.
    gate_deltas_mocap
        Optional gate-perturbation deltas (same shape as orchestrator
        emits on ``episode.metadata['perturbations']``). Used to move
        the AABB so the post-hoc check stays aligned with the actual
        perturbed gate Gaussians.
    runtime_error
        True if the orchestrator raised (e.g. policy timed out). Returns
        ``ERROR`` verbatim.

    Returns
    -------
    A dict with keys:

      - ``outcome``         : one of the string constants above
      - ``transited``       : bool — any state inside the gate AABB
      - ``first_inside_step``: optional int
      - ``last_inside_step`` : optional int
      - ``n_states_inside`` : int
      - ``aabb_mocap``      : [min, max] pair — the AABB we tested against
    """
    if runtime_error:
        return {
            "outcome": ERROR,
            "transited": False,
            "first_inside_step": None,
            "last_inside_step": None,
            "n_states_inside": 0,
            "aabb_mocap": None,
        }

    # Compositional / multi-gate scenes don't publish a single `gate_region`
    # — `OrderedMissGateCriterion` does its own ordered-transit accounting
    # at runtime. For these scenes we trust the runtime's final word
    # verbatim, skip the single-AABB post-hoc transit check, and infer
    # the failure phase from per-gate AABB latches if the scene declares
    # `gate_regions:` (plural — list of per-gate AABBs).
    if not scene_cfg.get("gate_region"):
        if runtime_failure_type is None:
            outcome = SUCCESS
        elif runtime_failure_type in _VERBATIM_FROM_FAILURE_TYPE:
            outcome = _VERBATIM_FROM_FAILURE_TYPE[runtime_failure_type]
        elif runtime_failure_type == FailureType.MISS_GATE:
            outcome = MISS_GATE
        elif runtime_failure_type == FailureType.GOAL_NOT_REACHED:
            outcome = GOAL_NOT_REACHED
        elif runtime_failure_type == FailureType.GOAL_REACHED:
            outcome = SUCCESS
        else:
            outcome = MISS_GATE
        phase = _compositional_phase_from_history(
            positions_mocap, scene_cfg,
        )
        return {
            "outcome": outcome,
            "transited": None,       # not checked
            "first_inside_step": None,
            "last_inside_step": None,
            "n_states_inside": None,
            "aabb_mocap": None,
            "phase": phase,
        }

    # Collisions / OOB / sim instabilities map straight through — they are
    # the rollout's authoritative final word.
    if runtime_failure_type in _VERBATIM_FROM_FAILURE_TYPE:
        # We still record transit info for context.
        tr = check_transit(positions_mocap, scene_cfg,
                           gate_deltas_mocap=gate_deltas_mocap)
        return {
            "outcome": _VERBATIM_FROM_FAILURE_TYPE[runtime_failure_type],
            "transited": tr.transited,
            "first_inside_step": tr.first_inside_step,
            "last_inside_step": tr.last_inside_step,
            "n_states_inside": tr.n_states_inside_aabb,
            "aabb_mocap": [tr.aabb_min_mocap.tolist(), tr.aabb_max_mocap.tolist()],
        }

    # Everything else (GOAL_REACHED / MISS_GATE / GOAL_NOT_REACHED / None
    # = timeout) depends on whether any state was inside the gate AABB.
    tr = check_transit(positions_mocap, scene_cfg,
                       gate_deltas_mocap=gate_deltas_mocap)
    base = {
        "transited": tr.transited,
        "first_inside_step": tr.first_inside_step,
        "last_inside_step": tr.last_inside_step,
        "n_states_inside": tr.n_states_inside_aabb,
        "aabb_mocap": [tr.aabb_min_mocap.tolist(), tr.aabb_max_mocap.tolist()],
    }

    # ---- directional gate-transit check ---------------------------------
    # When the prompt is direction-sensitive (e.g. center_gate_from_left
    # requires crossing the gate plane in -y; from_right in +y), demand
    # at least one correct-direction crossing inside the aperture and zero
    # wrong-direction crossings. Otherwise the trial is a MISS_GATE
    # regardless of what the runtime stopped on.
    directional_result: Optional[DirectionalTransitResult] = None
    if expected_dy_sign is not None:
        directional_result = check_directional_transit(
            positions_mocap,
            tr.aabb_min_mocap,
            tr.aabb_max_mocap,
            expected_dy_sign=expected_dy_sign,
        )
        base["expected_dy_sign"]  = int(expected_dy_sign)
        base["gate_plane_y_mocap"] = directional_result.gate_plane_y_mocap
        base["correct_crossings"] = directional_result.correct_crossings
        base["wrong_crossings"]   = directional_result.wrong_crossings
        base["first_correct_crossing_step"] = directional_result.first_correct_step
        base["first_wrong_crossing_step"]   = directional_result.first_wrong_step

    if runtime_failure_type == FailureType.GOAL_REACHED:
        # With the runtime AABB-transit latch (MissGateCriterion's
        # `transit_aabb_*` kwargs), GOAL_REACHED cannot fire unless the
        # drone has been inside the gate AABB at least once — so SUCCESS
        # would normally hold here. With directional-transit enforcement,
        # we additionally require a correct-direction aperture crossing
        # and zero wrong-direction crossings; otherwise the goal-prox stop
        # was reached via the wrong side and the trial is a MISS_GATE.
        if directional_result is not None and (
            directional_result.wrong_crossings > 0
            or directional_result.correct_crossings == 0
        ):
            base["outcome"] = MISS_GATE
        else:
            base["outcome"] = SUCCESS
        return base

    # No-progress stop or timeout — drone never reached the goal.
    # If a directional check is configured, prefer it: any wrong-direction
    # aperture crossing demotes to MISS_GATE even if the drone re-entered
    # the AABB later. Otherwise fall back to the legacy
    # "GOAL_NOT_REACHED if any state inside AABB" rule.
    if directional_result is not None:
        if directional_result.wrong_crossings > 0:
            base["outcome"] = MISS_GATE
        elif directional_result.correct_crossings > 0:
            base["outcome"] = GOAL_NOT_REACHED
        else:
            base["outcome"] = MISS_GATE
    else:
        base["outcome"] = GOAL_NOT_REACHED if tr.transited else MISS_GATE
    return base


# Reverse mapping for callers that want a FailureType enum equivalent of
# the post-hoc outcome (e.g. recovery-trigger filtering). SUCCESS maps to
# `None`; SKIPPED_GATE maps to MISS_GATE for trigger purposes.
OUTCOME_TO_FAILURE_TYPE: dict[str, Optional[FailureType]] = {
    SUCCESS:            None,
    MISS_GATE:          FailureType.MISS_GATE,
    GOAL_NOT_REACHED:   FailureType.GOAL_NOT_REACHED,
    COLLISION_GATE:     FailureType.COLLISION_GATE,
    COLLISION_OTHER:    FailureType.COLLISION_OTHER,
    OUT_OF_BOUNDS:      FailureType.OUT_OF_BOUNDS,
    EXCESSIVE_VELOCITY: FailureType.EXCESSIVE_VELOCITY,
    EXCESSIVE_TILT:     FailureType.EXCESSIVE_TILT,
    ERROR:              None,
}
