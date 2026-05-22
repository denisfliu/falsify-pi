"""Cross-criterion phase enrichment: when ANY criterion fires (e.g. a
collision), the FailureDetector merges every other criterion's
``phase_snapshot()`` into the record's ``extra`` so the recovery
sampler can scope safe-history by compositional phase even for
failures that didn't originate from ``OrderedMissGateCriterion``.
"""

from __future__ import annotations

import numpy as np
import pytest

from falsify.geometry import Frame, FrameGraph, Point
from falsify.safety import FailureDetector, FailureType, OrderedMissGateCriterion
from falsify.safety.criteria import SafetyCriterion, Violation
from falsify.sim.dynamics_state import DroneState


MOCAP = Frame("mocap")


def _fg():
    g = FrameGraph(); g.register_frame(MOCAP); return g


def _state(pos, t: float = 0.0):
    return DroneState(
        pos=Point(np.asarray(pos, dtype=np.float64), frame=MOCAP),
        vel=np.zeros(3),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        t=float(t),
    )


def _ordered_miss_gate():
    """Same two side-by-side gates as test_ordered_miss_gate_aperture_gate."""
    def _corners(x):
        return np.array([
            [x, -0.5, 1.0], [x, +0.5, 1.0],
            [x, +0.5, 2.0], [x, -0.5, 2.0],
        ], dtype=np.float64)
    return OrderedMissGateCriterion(
        corners_1=_corners(1.0), corners_2=_corners(3.0),
        frame_name="mocap", margin_m=0.0,
        eval_stop_mode=True,
        transit_aabb_1_min=np.array([0.95, -1.0, 0.5]),
        transit_aabb_1_max=np.array([1.05, +1.0, 2.5]),
        transit_aabb_2_min=np.array([2.95, -1.0, 0.5]),
        transit_aabb_2_max=np.array([3.05, +1.0, 2.5]),
    )


class _StubCollisionAt(SafetyCriterion):
    """Stubbed criterion that fires COLLISION_GATE on the Nth call."""
    name = "stub_collision"
    operates_in_frame = "mocap"

    def __init__(self, fire_on_step: int):
        self._step = 0
        self._fire = fire_on_step

    def check(self, state):
        self._step += 1
        if self._step == self._fire:
            return Violation(
                description="stub gate clip",
                value=1.0, threshold=0.0,
                failure_type=FailureType.COLLISION_GATE,
                extra={"label": "gate"},
            )
        return None


def test_collision_post_gate_1_inherits_transit_times_from_ordered_miss_gate():
    """A collision criterion fires AFTER the drone has cleared
    gate_1. The resulting FailureRecord.extra must include the
    OrderedMissGateCriterion's transit_time_1 + transit_time_1_exit
    (cross-criterion enrichment) AND keep the collision's own
    `label='gate'` (violation extras win on key collision)."""
    omg = _ordered_miss_gate()
    # Fire the stubbed collision on the 5th detector.update() call —
    # after the drone has threaded gate_1.
    coll = _StubCollisionAt(fire_on_step=5)
    det = FailureDetector(criteria=(omg, coll), frame_graph=_fg())

    trajectory = [
        _state((0.0, 0.0, 1.5), t=0.0),
        _state((1.0, 0.0, 1.5), t=0.1),   # crosses gate_1 plane inside aperture
        _state((1.5, 0.0, 1.5), t=0.2),   # exits gate_1 AABB
        _state((2.0, 0.0, 1.5), t=0.3),
        _state((2.5, 0.0, 1.5), t=0.4),   # collision fires here
    ]

    rec = None
    for step, st in enumerate(trajectory):
        rec = det.update(st, step=step)
        if rec is not None:
            break
    assert rec is not None and rec.failure_type == FailureType.COLLISION_GATE
    # Cross-criterion enrichment populated the gate-1 transit info.
    assert rec.extra.get("transit_time_1") is not None
    assert rec.extra.get("transit_time_1_exit") is not None
    assert rec.extra.get("phase") == "between_gates"
    assert rec.extra.get("ever_inside_aabb_1") is True
    assert rec.extra.get("ever_inside_aabb_2") is False
    # The collision criterion's own extras still win on key collision.
    assert rec.extra.get("label") == "gate"


def test_collision_pre_gate_1_has_phase_pre_gate_1_no_transit():
    """A collision that fires BEFORE the drone touches gate_1: phase
    snapshot reports `pre_gate_1` and `transit_time_1 = None`. Sampler
    will then NOT scope (because no transit time) — full history."""
    omg = _ordered_miss_gate()
    coll = _StubCollisionAt(fire_on_step=2)
    det = FailureDetector(criteria=(omg, coll), frame_graph=_fg())
    trajectory = [
        _state((0.0, 0.0, 1.5), t=0.0),
        _state((0.5, 0.0, 1.5), t=0.1),   # still pre-gate-1; collision fires here
        _state((1.0, 0.0, 1.5), t=0.2),
    ]
    rec = None
    for step, st in enumerate(trajectory):
        rec = det.update(st, step=step)
        if rec is not None:
            break
    assert rec is not None and rec.failure_type == FailureType.COLLISION_GATE
    assert rec.extra.get("phase") == "pre_gate_1"
    assert rec.extra.get("transit_time_1") is None
    assert rec.extra.get("transit_time_1_exit") is None


def test_collision_inside_gate_1_aabb_without_aperture_transit_is_pre_gate_1():
    """Regression for the "skip gate_1" replanning bug: a drone that
    *clips* a post of gate_1 has `_ever_inside_aabb_1=True` at the
    moment of impact, but `_transited_1=False` (no aperture crossing).
    Phase MUST report `pre_gate_1` — using AABB latches misclassified
    these as `between_gates`, causing the recovery planner to trim
    pre-gate_1 waypoints and replan straight toward gate_2."""
    omg = _ordered_miss_gate()
    coll = _StubCollisionAt(fire_on_step=3)
    det = FailureDetector(criteria=(omg, coll), frame_graph=_fg())
    # Trajectory: approach gate_1 OUTSIDE the aperture (y=0.8 > hv=0.5),
    # enter AABB, collision fires before any aperture crossing.
    trajectory = [
        _state((0.0, 0.8, 1.5), t=0.0),
        _state((0.95, 0.8, 1.5), t=0.1),  # enters gate_1 AABB at +y edge
        _state((1.0, 0.8, 1.5), t=0.2),   # inside AABB; crosses plane OUTSIDE aperture
        _state((1.05, 0.8, 1.5), t=0.3),  # collision fires here
    ]
    rec = None
    for step, st in enumerate(trajectory):
        rec = det.update(st, step=step)
        if rec is not None:
            break
    assert rec is not None and rec.failure_type == FailureType.COLLISION_GATE
    assert rec.extra.get("phase") == "pre_gate_1"
    assert rec.extra.get("transit_time_1") is None
    # AABB latch did fire (drone was inside the AABB) — but the phase
    # must NOT be derived from that.
    assert rec.extra.get("ever_inside_aabb_1") is True


def test_detector_without_ordered_miss_gate_does_not_break():
    """Criteria list without an OrderedMissGateCriterion ⇒ no
    phase_snapshot() to call ⇒ extra carries no compositional keys.
    Regression guard: the enrichment loop must not crash on
    criteria that lack the hook."""
    coll = _StubCollisionAt(fire_on_step=1)
    det = FailureDetector(criteria=(coll,), frame_graph=_fg())
    rec = det.update(_state((0.0, 0.0, 1.5)), step=0)
    assert rec is not None and rec.failure_type == FailureType.COLLISION_GATE
    assert "phase" not in rec.extra
    assert "transit_time_1" not in rec.extra
