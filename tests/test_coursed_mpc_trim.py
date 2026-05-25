"""Unit tests for the legacy compositional-phase trim. Skipped — the
trim is now driven by phase metadata + ``Course.target_waypoint`` +
``trim_course_to_target``; the old name-prefix table
(``_PHASE_FIRST_WAYPOINT_PREFIX``) was deleted in the phase-driven
recovery refactor."""

from __future__ import annotations

from dataclasses import replace as dc_replace

import numpy as np
import pytest

from falsify.planning.waypoints import Course, Waypoint
from falsify.recovery.coursed_mpc import CoursedMpcPlanner

pytestmark = pytest.mark.skip(
    reason="trim_course_for_phase replaced by Course.target_waypoint + "
           "trim_course_to_target (phase-metadata driven). The old name-prefix "
           "table no longer drives recovery routing."
)


def _make_course(waypoint_names: list[str]) -> Course:
    waypoints = tuple(
        Waypoint(name=n, p=np.array([float(i), 0.0, 1.5]), yaw=None, t=None)
        for i, n in enumerate(waypoint_names)
    )
    return Course(
        name="test",
        frame="mocap",
        fps=10,
        total_time_s=10.0,
        yaw_mode="tangent",
        waypoints=waypoints,
    )


@pytest.fixture
def planner():
    # Don't actually initialise the FrameGraph; the trim helper doesn't
    # use it. Side-step __init__ via __new__ + minimal attribute set.
    p = CoursedMpcPlanner.__new__(CoursedMpcPlanner)
    return p


def test_no_trim_when_phase_is_none(planner):
    course = _make_course(["start", "approach_1", "gate_1", "pre_gate_2", "gate_2", "hover"])
    out = planner._trim_course_for_phase(course, None)
    assert tuple(w.name for w in out.waypoints) == tuple(w.name for w in course.waypoints)


def test_no_trim_for_pre_gate_1_phase(planner):
    """Pre-gate-1 failures should plan the whole course (no waypoints
    have been passed)."""
    course = _make_course(["start", "approach_1", "gate_1", "pre_gate_2", "gate_2", "hover"])
    out = planner._trim_course_for_phase(course, "pre_gate_1")
    assert len(out.waypoints) == len(course.waypoints)


def test_between_gates_trims_to_pre_gate_2(planner):
    course = _make_course([
        "start", "approach_1", "gate_1", "post_gate_1",
        "pre_gate_2", "gate_2", "post_gate_2", "arc_left", "hover", "hover_hold",
    ])
    out = planner._trim_course_for_phase(course, "between_gates")
    names = [w.name for w in out.waypoints]
    assert names[0] == "pre_gate_2"
    # Subsequent waypoints retain original order.
    assert names == ["pre_gate_2", "gate_2", "post_gate_2", "arc_left",
                     "hover", "hover_hold"]


def test_post_gate_2_trims_to_hover(planner):
    course = _make_course([
        "start", "approach_1", "gate_1", "post_gate_1",
        "pre_gate_2", "gate_2", "post_gate_2", "arc_left", "hover", "hover_hold",
    ])
    out = planner._trim_course_for_phase(course, "post_gate_2")
    names = [w.name for w in out.waypoints]
    assert names == ["hover", "hover_hold"]


def test_phase_with_no_matching_waypoint_is_noop(planner):
    """Single-gate course that gets a compositional phase by accident
    (shouldn't happen, but if it does, don't crash) ⇒ no-op."""
    course = _make_course(["start", "approach", "gate", "post_gate", "hover", "hover_hold"])
    out = planner._trim_course_for_phase(course, "between_gates")
    # No waypoint starts with "pre_gate_2" ⇒ course returned unchanged.
    assert tuple(w.name for w in out.waypoints) == tuple(w.name for w in course.waypoints)


def test_unknown_phase_is_noop(planner):
    course = _make_course(["start", "gate_1", "hover"])
    out = planner._trim_course_for_phase(course, "some_future_phase")
    assert len(out.waypoints) == len(course.waypoints)


def test_compositional_waypoint_already_at_index_0_is_noop(planner):
    """If the target waypoint is already the first waypoint (after
    earlier trims), don't trim it away."""
    course = _make_course(["pre_gate_2", "gate_2", "hover"])
    out = planner._trim_course_for_phase(course, "between_gates")
    assert len(out.waypoints) == 3
