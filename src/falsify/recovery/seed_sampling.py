"""Sample a recovery seed state from the detector's safe-state history.

The orchestrator hands the recovery planner a single `DroneState` to plan
from. Historically that was the *last* safe state — the one immediately
before the failure. That worked for late-stage clips of the gate (small
correction near the gate fixes the trajectory) but is a poor seed for
failure modes where the entire approach drifted off course (miss-gate,
collision with the table) — there, restarting from a state deep in the
trajectory leaves no runway to recover. It's an equally poor seed for
`GOAL_NOT_REACHED`, where the drone already transited successfully —
replanning from before the gate would just re-fly the part that worked.

`sample_recovery_seed` is the single function that picks the right seed
for each failure mode. See `src/falsify/recovery/CLAUDE.md` for the
canonical bias / scope table, data flow, and persisted-metadata contract.

Quick reference (full table lives in `CLAUDE.md`)::

  Failure type          Scope of draw                       Beta(α, β)
  ──────────────────    ────────────────────────────────    ──────────
  COLLISION_GATE        full safe_history                   (3, 1)   late
  MISS_GATE             full safe_history                   (1, 3)   early
  COLLISION_OTHER       full safe_history                   (1, 3)   early
  OUT_OF_BOUNDS         full safe_history                   (1, 3)   early
  GOAL_NOT_REACHED      states with state.t >= transit_time (1, 3)   early-in-window
  (anything else)       full safe_history                   (1, 3)   early

`transit_time` is plumbed from `MissGateCriterion._transit_t` →
`Violation.extra["transit_time"]` →
`FailureRecord.extra["transit_time"]` →
`run_episode → sample_recovery_seed(transit_time=…)`. Don't introduce
a new detector→orchestrator channel; reuse this passthrough when adding
scoped sampling for future failure types.

Beta sampling is reproducible via the supplied `numpy.random.Generator`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from falsify.safety.records import FailureType
from falsify.sim.dynamics_state import DroneState


# Failure types where the rollout was likely fine until late — pick a
# *late* safe state so the recovery starts from near where things went
# wrong. The MPC then only has to do a small correction.
_BIAS_LATE: frozenset[FailureType] = frozenset({
    FailureType.COLLISION_GATE,
})

# Default α, β for the Beta distribution used to pick `s ∈ [0,1]`. Mean
# 0.25 (early) or 0.75 (late); both shapes are unimodal so the mass
# concentrates rather than uniformly spread.
_BETA_EARLY = (1.0, 3.0)
_BETA_LATE = (3.0, 1.0)


def sample_recovery_seed(
    safe_history: list[tuple[int, DroneState]],
    failure_type: Optional[FailureType],
    rng: np.random.Generator,
    *,
    trim_tail: int = 0,
    # The following kwargs are accepted for back-compat with older
    # call sites; the phase-scoping responsibility has moved to the
    # caller (orchestrator / Phase 2 collector), so these are ignored
    # here. Will be removed once all in-tree callers are migrated.
    transit_time: Optional[float] = None,
    gate_1_transit_time: Optional[float] = None,
    failure_phase: Optional[str] = None,
    pre_gate_bypass_time: Optional[float] = None,
) -> tuple[int, DroneState]:
    """Pick one ``(step, DroneState)`` from the safe history.

    See `src/falsify/recovery/CLAUDE.md § Replanning seed sampling` for
    the canonical bias / scope table and data flow.

    Parameters
    ----------
    safe_history
        Ordered list of ``(step, DroneState)`` from the detector — i.e.
        ``FailureRecord.safe_history``. Must be non-empty (caller's
        responsibility; orchestrator falls back to ``last_safe_state``
        on empty history).
    failure_type
        Drives both the *scope* of the draw and the Beta shape. See the
        table in `seed_sampling.py`'s module docstring. Unrecognised /
        ``None`` defaults to **early** bias (conservative: only push
        *toward* the failure when sure the approach was good).
    rng
        Source of randomness. Seed it for reproducible replays.
    transit_time
        Only consulted when ``failure_type == GOAL_NOT_REACHED``. When
        set, the draw is scoped to states with ``state.t >= transit_time``
        and the existing bias is applied **inside that window** (so the
        seed sits just after the gate crossing). Plumbed from
        ``MissGateCriterion._transit_t`` →
        ``Violation.extra["transit_time"]`` →
        ``FailureRecord.extra["transit_time"]`` → here.
    gate_1_transit_time
        Compositional-task scoping: when set AND ``transit_time`` is not
        already triggering the GOAL_NOT_REACHED scope, the draw is
        scoped to states with ``state.t >= gate_1_transit_time`` —
        i.e. only safe states that occurred AFTER the drone cleared
        gate 1. Used so a recovery replan after a gate-2 failure (mid-
        compositional) doesn't restart from before gate 1, throwing
        away the work of clearing it. Plumbed from
        ``OrderedMissGateCriterion._transit_t_1`` →
        ``Violation.extra["transit_time_1"]`` (stamped on every
        post-gate-1 stuck / goal-reached violation) →
        ``FailureRecord.extra["transit_time_1"]`` → here. The existing
        failure-type bias still applies inside the scoped window.

    Fallbacks (silent — no exceptions):

    - Singleton ``safe_history`` ⇒ that one state is returned.
    - ``GOAL_NOT_REACHED`` with no ``transit_time`` ⇒ falls through to
      ``gate_1_transit_time`` scoping if available, else full history.
    - Scoped window empty (rare — failure fired immediately after the
      transit before a single safe step was recorded post-crossing) ⇒
      falls back to full history with the type-driven bias.
    """
    if not safe_history:
        raise ValueError("safe_history is empty — no safe seed to sample from")

    # Tail trim: drop the last ``trim_tail`` entries before sampling.
    # The runtime collision criterion uses the drone OBB, so the entries
    # just before a collision are still "safe" by OBB-vs-points but are
    # spatially right next to the obstacle. SplatPlan inflates each
    # Gaussian by the drone clearance ``radius`` when building its voxel
    # occupancy grid, so a safe state at the gate's outer halo lands in
    # an occupied voxel and SplatPlan refuses to start. Trimming the
    # tail buys back the spatial margin SplatPlan needs.
    if trim_tail > 0 and len(safe_history) > trim_tail:
        safe_history = safe_history[:-trim_tail]

    # Phase scoping is now the caller's responsibility. By the time the
    # caller hands ``safe_history`` in, it has already been filtered to
    # entries in a single phase (e.g. the failure's post-trim phase).
    # The historical scoping kwargs (``transit_time`` / ``gate_1_transit_time``
    # / ``pre_gate_bypass_time``) are accepted for back-compat but no
    # longer alter sampling.
    candidates = safe_history

    if len(candidates) == 1:
        return candidates[0]

    if failure_type in _BIAS_LATE:
        alpha, beta = _BETA_LATE
    else:
        alpha, beta = _BETA_EARLY

    s = float(rng.beta(alpha, beta))
    idx = int(np.clip(np.floor(s * len(candidates)), 0, len(candidates) - 1))
    return candidates[idx]


def bias_for(failure_type: Optional[FailureType],
             failure_phase: Optional[str] = None) -> str:
    """Return ``"late"`` or ``"early"`` — used for logging /
    auditing the per-trial summary's ``recovery_seed.bias`` field.
    ``failure_phase`` is accepted (forwarded by the orchestrator) but
    not currently consulted; reserved for future phase-aware bias
    tables."""
    return "late" if failure_type in _BIAS_LATE else "early"
