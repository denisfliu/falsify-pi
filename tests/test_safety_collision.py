"""Tests for `PointCloudCollisionCriterion` and `MissGateCriterion`.

The collision criterion treats the drone as an oriented bounding box (OBB)
in body-FRD and tests for inclusion of labeled NED-frame point clouds.
The miss-gate criterion fires when the drone segment between two adjacent
states crosses the gate plane outside the aperture rectangle.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as _R

from falsify.geometry import NED, MOCAP, FrameGraph, Point, SE3
from falsify.safety import (
    DroneBody, FailureDetector, FailureType,
    MissGateCriterion, PointCloudCollisionCriterion,
)
from falsify.sim.dynamics_state import DroneState


def _state(pos=(0, 0, 0), q=(0, 0, 0, 1.0), t=0.0, frame=NED):
    return DroneState(
        pos=Point.of(*pos, frame),
        vel=np.zeros(3),
        quat_xyzw=np.asarray(q, dtype=np.float64),
        t=t,
    )


def _graph_ned_only():
    g = FrameGraph()
    g.register_frame(NED)
    return g


# ---------------------------------------------------------------------------
# DroneBody
# ---------------------------------------------------------------------------


def test_drone_body_rejects_non_positive_extents():
    with pytest.raises(ValueError):
        DroneBody(half_extents=np.array([0.0, 0.1, 0.1]))


def test_drone_body_bounding_radius_geometry():
    b = DroneBody(half_extents=np.array([0.3, 0.4, 0.0 + 0.0001]))
    # ||(0.3, 0.4, 0.0001)|| ≈ 0.5 (and a hair).
    assert abs(b.bounding_radius - 0.5) < 1e-3


# ---------------------------------------------------------------------------
# PointCloudCollisionCriterion — axis-aligned (identity quaternion)
# ---------------------------------------------------------------------------


def test_collision_no_points_no_violation():
    body = DroneBody(half_extents=np.array([0.2, 0.2, 0.1]))
    crit = PointCloudCollisionCriterion(
        body,
        labeled_clouds={"gate": np.empty((0, 3)), "other": np.empty((0, 3))},
    )
    assert crit.check(_state(pos=(0, 0, 1))) is None


def test_collision_far_from_clouds_no_violation():
    body = DroneBody(half_extents=np.array([0.2, 0.2, 0.1]))
    gate = np.array([[5.0, 5.0, 1.0]])
    crit = PointCloudCollisionCriterion(body, labeled_clouds={"gate": gate})
    assert crit.check(_state(pos=(0, 0, 1))) is None


def test_collision_with_gate_point_inside_box():
    body = DroneBody(half_extents=np.array([0.2, 0.2, 0.1]))
    gate = np.array([[0.05, 0.0, 1.0]])     # 5 cm in front along NED-x
    crit = PointCloudCollisionCriterion(body, labeled_clouds={"gate": gate})
    v = crit.check(_state(pos=(0, 0, 1)))
    assert v is not None
    assert v.failure_type is FailureType.COLLISION_GATE
    assert v.extra["hit_label"] == "gate"
    assert v.extra["n_hits"] == 1


def test_collision_with_only_other_point_classifies_as_other():
    body = DroneBody(half_extents=np.array([0.2, 0.2, 0.1]))
    other = np.array([[0.05, 0.0, 1.0]])
    crit = PointCloudCollisionCriterion(body, labeled_clouds={"table": other})
    v = crit.check(_state(pos=(0, 0, 1)))
    assert v is not None
    assert v.failure_type is FailureType.COLLISION_OTHER
    assert v.extra["hit_label"] == "table"


def test_collision_gate_wins_over_other_in_same_box():
    body = DroneBody(half_extents=np.array([0.3, 0.3, 0.2]))
    gate = np.array([[0.1, 0.0, 1.0]])
    other = np.array([[-0.1, 0.0, 1.0]])
    crit = PointCloudCollisionCriterion(
        body, labeled_clouds={"gate": gate, "table": other},
    )
    v = crit.check(_state(pos=(0, 0, 1)))
    assert v is not None
    # Both inside → gate wins the classification.
    assert v.failure_type is FailureType.COLLISION_GATE


def test_collision_obb_rotation_changes_containment():
    # Tall narrow box: hx=1.0 along forward, hy=0.05 (very thin sideways),
    # hz=0.1. A point at (0, 0.5, 1) sits along NED-y, 0.5 m off the body
    # origin. With identity quat (body forward = NED +x) it is *outside*
    # the half-extent of 0.05 along body-y. After yawing 90° about NED-z,
    # body forward = NED +y, so the body-y direction becomes NED -x, and
    # the same world point (0, 0.5, 1) lies along the box's long axis —
    # now *inside*.
    body = DroneBody(half_extents=np.array([1.0, 0.05, 0.1]))
    pt = np.array([[0.0, 0.5, 1.0]])
    crit = PointCloudCollisionCriterion(body, labeled_clouds={"gate": pt})

    # Identity quaternion: outside.
    v_id = crit.check(_state(pos=(0, 0, 1), q=(0, 0, 0, 1.0)))
    assert v_id is None

    # +90° yaw about NED-z (world-up in NED is -z; we rotate about +z to
    # align body-x with NED +y). For NED with z-down, a "yaw" rotation
    # around z is the same maths regardless of sign convention.
    q_yaw = _R.from_euler("z", np.pi / 2).as_quat()
    v_rot = crit.check(_state(pos=(0, 0, 1), q=q_yaw))
    assert v_rot is not None
    assert v_rot.failure_type is FailureType.COLLISION_GATE


def test_collision_criterion_detector_integration_uses_failure_type_from_violation():
    # The detector should read the failure_type override from the
    # Violation rather than falling back to _NAME_TO_TYPE lookup.
    g = _graph_ned_only()
    body = DroneBody(half_extents=np.array([0.2, 0.2, 0.1]))
    other = np.array([[0.05, 0.0, 1.0]])
    crit = PointCloudCollisionCriterion(body, labeled_clouds={"table": other})
    det = FailureDetector([crit], g)
    rec = det.update(_state(pos=(0, 0, 1)), step=0)
    assert rec is not None
    assert rec.failure_type is FailureType.COLLISION_OTHER


# ---------------------------------------------------------------------------
# MissGateCriterion
# ---------------------------------------------------------------------------


def _square_aperture_mocap(centre=(0, 0, 1.5), half_width=0.4, half_height=0.4):
    # Square aperture in the y=0 plane (mocap). Order the corners so that
    # corners[1]-corners[0] and corners[3]-corners[0] are orthogonal
    # adjacent edges.
    cx, _, cz = centre
    return np.array([
        [cx - half_width, 0.0, cz - half_height],   # 0: bottom-left
        [cx + half_width, 0.0, cz - half_height],   # 1: bottom-right (u edge)
        [cx + half_width, 0.0, cz + half_height],   # 2: top-right
        [cx - half_width, 0.0, cz + half_height],   # 3: top-left  (v edge)
    ], dtype=np.float64)


def test_miss_gate_no_crossing_no_violation():
    g = _graph_ned_only()
    g.register_frame(MOCAP)
    g.register_edge(SE3.identity(NED, MOCAP))
    crit = MissGateCriterion(_square_aperture_mocap(), frame_name="mocap")
    # Walk along x at y=-1 (never crosses y=0).
    assert crit.check_with_graph(_state(pos=(-1, -1, 1.5), frame=MOCAP), g) is None
    assert crit.check_with_graph(_state(pos=( 0, -1, 1.5), frame=MOCAP), g) is None
    assert crit.check_with_graph(_state(pos=( 1, -1, 1.5), frame=MOCAP), g) is None


def test_miss_gate_crossing_inside_aperture_marks_transited():
    g = _graph_ned_only()
    g.register_frame(MOCAP)
    g.register_edge(SE3.identity(NED, MOCAP))
    crit = MissGateCriterion(_square_aperture_mocap(), frame_name="mocap")
    # Approach + cross at the centre.
    assert crit.check_with_graph(_state(pos=(0, -1, 1.5), frame=MOCAP), g) is None
    assert crit.check_with_graph(_state(pos=(0,  1, 1.5), frame=MOCAP), g) is None
    # Crossing again should NOT re-fire — drone already transited.
    assert crit.check_with_graph(_state(pos=(0, -1, 1.5), frame=MOCAP), g) is None


def test_miss_gate_crossing_outside_aperture_fires_miss():
    g = _graph_ned_only()
    g.register_frame(MOCAP)
    g.register_edge(SE3.identity(NED, MOCAP))
    crit = MissGateCriterion(_square_aperture_mocap(), frame_name="mocap")
    # Approach + cross 1 m above the aperture (out of vertical extent).
    assert crit.check_with_graph(_state(pos=(0, -1, 3.0), frame=MOCAP), g) is None
    v = crit.check_with_graph(_state(pos=(0, 1, 3.0), frame=MOCAP), g)
    assert v is not None
    assert v.failure_type is FailureType.MISS_GATE
    assert v.extra["cv"] > v.extra["hv"]


def test_miss_gate_reset_clears_transit():
    g = _graph_ned_only()
    g.register_frame(MOCAP)
    g.register_edge(SE3.identity(NED, MOCAP))
    crit = MissGateCriterion(_square_aperture_mocap(), frame_name="mocap")
    crit.check_with_graph(_state(pos=(0, -1, 1.5), frame=MOCAP), g)
    crit.check_with_graph(_state(pos=(0,  1, 1.5), frame=MOCAP), g)
    assert crit._transited is True
    crit.reset()
    assert crit._transited is False
    assert crit._prev_xyz is None


def test_miss_gate_rejects_non_rectangular_corners():
    # Parallelogram, not a rectangle: corners[3] - corners[0] = (1, 0.1, 0)
    # which is not orthogonal to corners[1] - corners[0] = (1, 0, 0).
    bad = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.1, 0.0],
        [1.0, 0.1, 0.0],
    ], dtype=np.float64)
    with pytest.raises(ValueError, match="orthogonal"):
        MissGateCriterion(bad, frame_name="mocap")
