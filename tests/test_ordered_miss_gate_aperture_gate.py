"""Test that ``OrderedMissGateCriterion`` in ``eval_stop_mode`` fires
GOAL_REACHED only after the drone has actually crossed BOTH aperture
planes inside the rectangle — clipping the AABB edge without aperture
transit is no longer enough.

Regression for the compositional-SUCCESS bug: a drone could enter both
gate AABBs at the corners without ever threading either aperture, end
up near the goal, and be classified as SUCCESS.
"""

from __future__ import annotations

import numpy as np
import pytest

from falsify.geometry import Frame, FrameGraph, Point
from falsify.safety import FailureType, OrderedMissGateCriterion
from falsify.sim.dynamics_state import DroneState


MOCAP = Frame("mocap")


def _fg() -> FrameGraph:
    """Tiny FrameGraph with only MOCAP registered. The criterion's
    operates_in_frame == 'mocap' and our test states are already in
    MOCAP, so check_with_graph short-circuits without needing any
    transforms."""
    g = FrameGraph()
    g.register_frame(MOCAP)
    return g


def _state(pos, t: float = 0.0) -> DroneState:
    return DroneState(
        pos=Point(np.asarray(pos, dtype=np.float64), frame=MOCAP),
        vel=np.zeros(3),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        t=float(t),
    )


def _make_criterion(
    *,
    eval_stop_mode: bool = True,
    goal: tuple[float, float, float] = (5.0, 0.0, 1.5),
    goal_tolerance_m: float = 0.5,
):
    """Two side-by-side axis-aligned gates:
      gate_1 plane: x = 1.0, aperture rectangle y ∈ [-0.5, 0.5], z ∈ [1.0, 2.0]
      gate_2 plane: x = 3.0, aperture rectangle y ∈ [-0.5, 0.5], z ∈ [1.0, 2.0]
    Drone flies +x; transit direction is +x, normal is +x for both.

    AABBs are slightly larger than the aperture rectangles in y and z so
    we can construct trajectories that ENTER an AABB at the corner
    without crossing the aperture plane within the rectangle.

      gate_1 AABB: x ∈ [0.95, 1.05], y ∈ [-1.0, 1.0], z ∈ [0.5, 2.5]
      gate_2 AABB: x ∈ [2.95, 3.05], y ∈ [-1.0, 1.0], z ∈ [0.5, 2.5]
    """
    # Corner ordering: 4 corners tracing the aperture rectangle (in u/v).
    # For an x-normal plane, u = +y, v = +z is the natural local frame.
    def _corners(x_plane: float):
        return np.array([
            [x_plane, -0.5, 1.0],
            [x_plane, +0.5, 1.0],
            [x_plane, +0.5, 2.0],
            [x_plane, -0.5, 2.0],
        ], dtype=np.float64)

    return OrderedMissGateCriterion(
        corners_1=_corners(1.0),
        corners_2=_corners(3.0),
        frame_name="mocap",
        margin_m=0.0,
        goal_position=np.asarray(goal, dtype=np.float64),
        goal_tolerance_m=goal_tolerance_m,
        min_progress_window_s=None,         # disable stuck-check for these tests
        eval_stop_mode=eval_stop_mode,
        transit_aabb_1_min=np.array([0.95, -1.0, 0.5]),
        transit_aabb_1_max=np.array([1.05, +1.0, 2.5]),
        transit_aabb_2_min=np.array([2.95, -1.0, 0.5]),
        transit_aabb_2_max=np.array([3.05, +1.0, 2.5]),
    )


def _run(criterion, trajectory):
    """Feed a list of states via check_with_graph (which manages the
    _prev_xyz the plane-cross check needs); return the FIRST Violation
    or None."""
    g = _fg()
    fired = None
    for i, st in enumerate(trajectory):
        v = criterion.check_with_graph(st, g)
        if v is not None:
            fired = (i, v)
            break
    return fired


def test_clipping_both_aabbs_without_aperture_transit_does_not_fire_goal_reached():
    """Drone passes through both AABBs OUTSIDE the rectangle (y > 0.5),
    ends up at the goal. Before the fix this would have latched
    both `_ever_inside_aabb_*` and fired GOAL_REACHED → SUCCESS.
    After the fix, aperture transit is required ⇒ no GOAL_REACHED."""
    c = _make_criterion()
    # Trajectory: enter gate_1 AABB at the +y edge (y = 0.8, outside
    # aperture which is |y| ≤ 0.5), exit, enter gate_2 AABB similarly,
    # exit, drift to goal.
    traj = [
        _state((0.0, 0.8, 1.5), t=0.0),
        _state((1.0, 0.8, 1.5), t=0.1),   # crosses gate_1 plane at y=0.8 — OUTSIDE aperture
        _state((2.0, 0.8, 1.5), t=0.2),
        _state((3.0, 0.8, 1.5), t=0.3),   # crosses gate_2 plane at y=0.8 — OUTSIDE aperture
        _state((4.5, 0.0, 1.5), t=0.4),
        _state((5.0, 0.0, 1.5), t=0.5),   # at goal
    ]
    fired = _run(c, traj)
    assert fired is None, (
        f"GOAL_REACHED should NOT fire when drone clips AABBs without "
        f"aperture transit; got {fired[1].failure_type if fired else None} "
        f"at step {fired[0] if fired else None}"
    )
    # Sanity: AABB latches DID fire (both AABBs entered), but aperture
    # transits did NOT.
    assert c._ever_inside_aabb_1 is True
    assert c._ever_inside_aabb_2 is True
    assert c._transited_1 is False
    assert c._transited_2 is False


def test_aperture_transit_through_both_gates_fires_goal_reached():
    """Drone threads both apertures cleanly and reaches goal — must
    fire GOAL_REACHED (regression guard)."""
    c = _make_criterion()
    traj = [
        _state((0.0, 0.0, 1.5), t=0.0),
        _state((1.0, 0.0, 1.5), t=0.1),   # gate_1 plane crossing at (0, 1.5) — inside aperture
        _state((2.0, 0.0, 1.5), t=0.2),
        _state((3.0, 0.0, 1.5), t=0.3),   # gate_2 plane crossing — inside aperture
        _state((4.0, 0.0, 1.5), t=0.4),
        _state((5.0, 0.0, 1.5), t=0.5),   # at goal
    ]
    fired = _run(c, traj)
    assert fired is not None and fired[1].failure_type == FailureType.GOAL_REACHED
    assert fired[1].extra.get("both_aperture_transited") is True
    assert c._transited_1 is True
    assert c._transited_2 is True


def test_wrong_direction_gate_1_fires_immediate_miss_gate():
    """Drone first goes around gate_1's aperture (clipping outside
    the rectangle in +x direction), then loops back and crosses INSIDE
    the aperture in the -x direction. Expected dir (auto-inferred
    from initial -x side) is +1, observed is -1 ⇒ MISS_GATE
    wrong_direction_aperture, regardless of eval_stop_mode."""
    c = _make_criterion()
    traj = [
        _state((0.0, 0.8, 1.5), t=0.0),    # initial -x side, y=0.8 outside aperture
        _state((2.0, 0.8, 1.5), t=0.1),    # crosses plane OUTSIDE aperture, +x dir
        _state((0.5, 0.0, 1.5), t=0.2),    # crosses plane INSIDE aperture, -x dir → WRONG
    ]
    fired = _run(c, traj)
    assert fired is not None
    step, v = fired
    assert v.failure_type == FailureType.MISS_GATE
    assert v.extra.get("mode") == "wrong_direction_aperture"
    assert v.extra.get("which_gate") == "gate_1"
    assert v.extra.get("dir_sign") == -v.extra.get("expected_dir_sign")


def test_out_of_order_gate_2_before_gate_1_fires_miss_gate():
    """Drone threads gate_2's aperture WITHOUT first transiting
    gate_1. Must fire MISS_GATE out_of_order_aperture — the previous
    elif-branch silently ignored gate_2 crossings while gate_1 was
    still un-transited, allowing wrong-gate-first trajectories to
    eventually satisfy the success criterion."""
    c = _make_criterion()
    # Skip gate_1 entirely; go straight through gate_2.
    traj = [
        _state((0.0, 0.0, 1.5), t=0.0),
        _state((2.0, 0.0, 1.5), t=0.1),     # past gate_1 (no inside crossing — corner clip)
        _state((4.0, 0.0, 1.5), t=0.2),     # crosses gate_2 aperture INSIDE rectangle
    ]
    # Adjust trajectory so the drone skips gate_1's inside crossing.
    # gate_1 plane is at x=1 with aperture y∈[-0.5,+0.5], z∈[1,2]. We
    # cross the plane at y=0, z=1.5 which IS inside the aperture, so
    # the test as written would latch _transited_1. We need to go
    # AROUND gate_1's aperture rectangle: use y=0.8 (outside aperture).
    traj = [
        _state((0.0, 0.8, 1.5), t=0.0),
        _state((1.0, 0.8, 1.5), t=0.1),     # crosses gate_1 plane OUTSIDE aperture
        _state((2.0, 0.0, 1.5), t=0.2),     # drift back to y=0
        _state((3.0, 0.0, 1.5), t=0.3),     # crosses gate_2 plane INSIDE aperture (out of order!)
    ]
    fired = _run(c, traj)
    assert fired is not None
    step, v = fired
    assert v.failure_type == FailureType.MISS_GATE
    assert v.extra.get("mode") == "out_of_order_aperture"
    assert v.extra.get("which_gate") == "gate_2"
    assert c._transited_1 is False
    assert c._transited_2 is False


def test_only_first_aperture_transited_does_not_fire_goal_reached():
    """Drone threads gate_1 cleanly, clips gate_2 AABB at the edge
    without crossing its aperture, reaches goal. Should NOT fire
    GOAL_REACHED — gate_2's aperture was missed."""
    c = _make_criterion()
    traj = [
        _state((0.0, 0.0, 1.5), t=0.0),
        _state((1.0, 0.0, 1.5), t=0.1),   # gate_1 aperture transit OK
        _state((2.0, 0.0, 1.5), t=0.2),
        _state((2.5, 0.8, 1.5), t=0.3),   # drift to y=0.8 before gate_2 plane
        _state((3.0, 0.8, 1.5), t=0.4),   # crosses gate_2 plane at y=0.8 — OUTSIDE aperture
        _state((5.0, 0.0, 1.5), t=0.5),   # at goal
    ]
    fired = _run(c, traj)
    assert fired is None
    assert c._transited_1 is True
    assert c._transited_2 is False
    assert c._ever_inside_aabb_2 is True   # AABB latch fired
