"""Tests for scene_edits — purely numpy / FrameGraph; no GPU required."""

from __future__ import annotations

import numpy as np
import pytest

from falsify.geometry import (
    COLMAP, MOCAP, NED, NS, FrameGraph, SE3, Sim3,
)
from falsify.sim.scene_edits import (
    RigidTransformAABB, apply_edits_to_arrays, load_scene_edits,
)


def _graph_perm5_with_ns_scale(scale: float = 0.5):
    """Minimal FrameGraph matching the falsify scene conventions:
    ned↔mocap via perm5 (diag(1,-1,-1)); mocap↔ns Sim3 with uniform scale.
    """
    g = FrameGraph()
    for f in (NED, MOCAP, NS):
        g.register_frame(f)
    g.register_edge(SE3(R=np.diag([1.0, -1.0, -1.0]), t=np.zeros(3), src=NED, dst=MOCAP))
    g.register_edge(Sim3(s=scale, R=np.eye(3), t=np.zeros(3), src=MOCAP, dst=NS))
    return g


def test_yaw_rotation_aligns_normals():
    """Source normal at (1, 0) → target at (0, 1) ⇒ rotation by +π/2 about z."""
    edit = RigidTransformAABB(
        name="t",
        target_aabb_frame="mocap",
        target_aabb_min=np.array([-1, -1, -1]),
        target_aabb_max=np.array([1, 1, 1]),
        source_anchor=np.array([0.0, 0.0, 0.0]),
        target_anchor=np.array([0.0, 0.0, 0.0]),
        source_normal=np.array([1.0, 0.0, 0.0]),
        target_normal=np.array([0.0, 1.0, 0.0]),
    )
    T = edit.transform_in_authored_frame()
    np.testing.assert_allclose(T.R, np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]]), atol=1e-12)
    np.testing.assert_allclose(T.t, np.zeros(3), atol=1e-12)


def test_transform_in_other_frame_roundtrip():
    """A no-op edit (source==target) should produce identity in any frame."""
    g = _graph_perm5_with_ns_scale(scale=0.5)
    edit = RigidTransformAABB(
        name="noop",
        target_aabb_frame="mocap",
        target_aabb_min=np.array([-1, -1, -1]),
        target_aabb_max=np.array([1, 1, 1]),
        source_anchor=np.array([0.3, 0.4, 0.5]),
        target_anchor=np.array([0.3, 0.4, 0.5]),
        source_normal=np.array([1.0, 0.0, 0.0]),
        target_normal=np.array([1.0, 0.0, 0.0]),
    )
    T_ns = edit.transform_in("ns", g)
    np.testing.assert_allclose(T_ns.R, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(T_ns.t, np.zeros(3), atol=1e-9)


def test_aabb_min_max_in_ns_scales_correctly():
    """AABB at MOCAP [0, 1] × [0, 1] × [0, 1] under Sim3 scale=0.5 → NS [0, 0.5]^3."""
    g = _graph_perm5_with_ns_scale(scale=0.5)
    edit = RigidTransformAABB(
        name="t",
        target_aabb_frame="mocap",
        target_aabb_min=np.array([0.0, 0.0, 0.0]),
        target_aabb_max=np.array([1.0, 1.0, 1.0]),
        source_anchor=np.array([0.5, 0.5, 0.5]),
        target_anchor=np.array([0.5, 0.5, 0.5]),
        source_normal=np.array([1.0, 0.0, 0.0]),
        target_normal=np.array([1.0, 0.0, 0.0]),
    )
    mn, mx = edit.aabb_min_max_in("ns", g)
    np.testing.assert_allclose(mn, np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(mx, np.array([0.5, 0.5, 0.5]), atol=1e-12)


def test_apply_to_arrays_translates_only_masked_points():
    g = _graph_perm5_with_ns_scale(scale=1.0)  # use scale=1 so NS == MOCAP for points
    # NS points: two inside the AABB, two outside.
    means = np.array([
        [0.5, 0.5, 0.5],     # inside
        [0.6, 0.4, 0.7],     # inside
        [5.0, 5.0, 5.0],     # outside
        [-3.0, -3.0, -3.0],  # outside
    ])
    edit = RigidTransformAABB(
        name="move",
        target_aabb_frame="mocap",
        target_aabb_min=np.array([0.0, 0.0, 0.0]),
        target_aabb_max=np.array([1.0, 1.0, 1.0]),
        source_anchor=np.array([0.5, 0.5, 0.5]),
        target_anchor=np.array([2.5, -0.25, 0.0]),
        # Pure translation: keep the same normal.
        source_normal=np.array([1.0, 0.0, 0.0]),
        target_normal=np.array([1.0, 0.0, 0.0]),
    )
    new_means, _ = apply_edits_to_arrays(means, None, [edit], g)
    # Inside points get translated by (target - source).
    delta = np.array([2.0, -0.75, -0.5])
    np.testing.assert_allclose(new_means[0], means[0] + delta, atol=1e-12)
    np.testing.assert_allclose(new_means[1], means[1] + delta, atol=1e-12)
    # Outside points untouched.
    np.testing.assert_allclose(new_means[2], means[2], atol=1e-12)
    np.testing.assert_allclose(new_means[3], means[3], atol=1e-12)


def test_apply_rotates_points_around_source_anchor():
    """A pure rotation in MOCAP (source==target anchors) should rotate
    masked points about that anchor."""
    g = _graph_perm5_with_ns_scale(scale=1.0)
    means = np.array([
        [1.0, 0.0, 0.0],  # source: at distance 1 along +x from anchor (0,0,0)
        [0.0, 1.0, 0.0],  # source: along +y from anchor
        [10.0, 10.0, 10.0],   # outside AABB; untouched
    ])
    edit = RigidTransformAABB(
        name="rotate",
        target_aabb_frame="mocap",
        target_aabb_min=np.array([-2.0, -2.0, -2.0]),
        target_aabb_max=np.array([2.0, 2.0, 2.0]),
        source_anchor=np.array([0.0, 0.0, 0.0]),
        target_anchor=np.array([0.0, 0.0, 0.0]),
        source_normal=np.array([1.0, 0.0, 0.0]),  # +x
        target_normal=np.array([0.0, 1.0, 0.0]),  # +y (90° CCW)
    )
    new_means, _ = apply_edits_to_arrays(means, None, [edit], g)
    # +x  → +y; +y → -x
    np.testing.assert_allclose(new_means[0], [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(new_means[1], [-1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(new_means[2], means[2], atol=1e-12)


def test_combined_rotate_and_translate():
    """Realistic gate-style edit: rotate the source normal to mocap -y,
    place the source anchor at a far target. Verify a point initially
    aligned with source_normal ends up at target_anchor + target_normal."""
    g = _graph_perm5_with_ns_scale(scale=1.0)
    source_anchor = np.array([0.86, 0.69, 0.07])
    target_anchor = np.array([2.5, -0.25, 0.0])
    source_normal = np.array([0.749, 0.663, 0.0])
    target_normal = np.array([0.0, -1.0, 0.0])
    edit = RigidTransformAABB(
        name="gate_move",
        target_aabb_frame="mocap",
        target_aabb_min=source_anchor - 1.0,
        target_aabb_max=source_anchor + 1.0,
        source_anchor=source_anchor,
        target_anchor=target_anchor,
        source_normal=source_normal,
        target_normal=target_normal,
    )
    # A point one meter out along the source normal from the source anchor.
    s_norm_unit = source_normal / np.linalg.norm(source_normal)
    pt = source_anchor + s_norm_unit
    means = np.stack([pt])
    new_means, _ = apply_edits_to_arrays(means, None, [edit], g)
    # Expected: target_anchor + target_normal (also unit).
    t_norm_unit = target_normal / np.linalg.norm(target_normal)
    expected = target_anchor + t_norm_unit
    np.testing.assert_allclose(new_means[0], expected, atol=1e-9)


def test_quaternion_pre_multiplied_for_rotated_points():
    """Quaternion update should match the rotation applied to means."""
    g = _graph_perm5_with_ns_scale(scale=1.0)
    means = np.array([[1.0, 0.0, 0.0]])
    # Identity quaternion (wxyz).
    quats = np.array([[1.0, 0.0, 0.0, 0.0]])
    edit = RigidTransformAABB(
        name="rot90",
        target_aabb_frame="mocap",
        target_aabb_min=np.array([-2.0, -2.0, -2.0]),
        target_aabb_max=np.array([2.0, 2.0, 2.0]),
        source_anchor=np.array([0.0, 0.0, 0.0]),
        target_anchor=np.array([0.0, 0.0, 0.0]),
        source_normal=np.array([1.0, 0.0, 0.0]),
        target_normal=np.array([0.0, 1.0, 0.0]),
    )
    _new_means, new_quats = apply_edits_to_arrays(means, quats, [edit], g)
    # +π/2 rotation about z, wxyz: w=cos(π/4)=√2/2, z=sin(π/4)=√2/2.
    expected = np.array([np.sqrt(2) / 2, 0.0, 0.0, np.sqrt(2) / 2])
    np.testing.assert_allclose(new_quats[0], expected, atol=1e-9)


def test_load_scene_edits_from_dict():
    cfg = {
        "scene_edits": [{
            "name": "gate_move",
            "type": "rigid_transform_aabb",
            "target_aabb_frame": "mocap",
            "target_aabb_min": [0.5, 0.2, 0.0],
            "target_aabb_max": [1.2, 1.1, 2.0],
            "transform": {
                "source_anchor": [0.86, 0.69, 0.07],
                "target_anchor": [2.5, -0.25, 0.0],
                "source_normal": [0.749, 0.663, 0.0],
                "target_normal": [0.0, -1.0, 0.0],
            },
            "applies_to_scene_objects": ["gate"],
        }]
    }
    edits = load_scene_edits(cfg)
    assert len(edits) == 1
    e = edits[0]
    assert e.name == "gate_move"
    assert e.applies_to_scene_objects == ("gate",)
    np.testing.assert_allclose(e.target_anchor, [2.5, -0.25, 0.0])


def test_load_scene_edits_empty():
    assert load_scene_edits({}) == []
    assert load_scene_edits({"scene_edits": None}) == []
    assert load_scene_edits({"scene_edits": []}) == []


# ---------------------------------------------------------------------------
# Oriented box tests (yaw-only rotation about z)
# ---------------------------------------------------------------------------


def test_oriented_box_contains_along_diagonal():
    """A 45° yawed box with a long x half-extent contains points along
    the +xy diagonal but not along +x alone."""
    from falsify.sim.scene_edits import _OrientedBox
    box = _OrientedBox(
        center=np.array([0.0, 0.0, 0.0]),
        half_extents=np.array([1.0, 0.2, 1.0]),
        yaw=np.pi / 4,
    )
    pts = np.array([
        [0.7, 0.7, 0.0],     # along long axis after rotation — inside
        [0.0, 0.0, 0.0],     # centre
        [1.0, 0.0, 0.0],     # off-axis — outside the narrow y half-extent
    ])
    inside = box.contains(pts)
    assert inside[0]
    assert inside[1]
    assert not inside[2]


def test_oriented_include_unions_with_aabbs():
    """Oriented include boxes union with axis-aligned ones in the
    applier's selection mask."""
    from falsify.sim.scene_edits import (
        _OrientedBox, RigidTransformAABB, apply_edits_to_arrays,
    )
    g = _graph_perm5_with_ns_scale(scale=1.0)   # NS == MOCAP for ease
    pts = np.array([
        [0.5, 0.5, 0.5],     # in main AABB
        [3.0, 3.0, 0.0],     # outside main AABB, inside oriented box
        [10.0, 10.0, 10.0],  # outside everything
    ])
    edit = RigidTransformAABB(
        name="t",
        target_aabb_frame="mocap",
        target_aabb_min=np.array([0.0, 0.0, 0.0]),
        target_aabb_max=np.array([1.0, 1.0, 1.0]),
        source_anchor=np.array([0.5, 0.5, 0.5]),
        target_anchor=np.array([10.0, 10.0, 10.0]),   # big translation
        source_normal=np.array([1.0, 0.0, 0.0]),
        target_normal=np.array([1.0, 0.0, 0.0]),
        oriented_include_aabbs=(_OrientedBox(
            center=np.array([3.0, 3.0, 0.0]),
            half_extents=np.array([0.5, 0.5, 0.5]),
            yaw=0.0,
        ),),
    )
    new_pts, _ = apply_edits_to_arrays(pts, None, [edit], g)
    moved = np.linalg.norm(new_pts - pts, axis=1)
    assert moved[0] > 1.0, "point in main AABB should move"
    assert moved[1] > 1.0, "point in oriented include should move"
    assert moved[2] < 1e-9, "outside-everything point should stay"


def test_precise_include_overrides_axis_exclude():
    """``include_aabbs`` are precise / hand-curated and override
    ``exclude_aabbs``. A point inside both should move (precise wins)."""
    from falsify.sim.scene_edits import (
        _Box, RigidTransformAABB, apply_edits_to_arrays,
    )
    g = _graph_perm5_with_ns_scale(scale=1.0)
    pts = np.array([
        [0.5, 0.5, 0.5],   # main AABB, NOT in precise, NOT in exclude   → moves
        [0.7, 0.5, 0.5],   # main AABB AND exclude (no precise rescue)    → stays
        [0.8, 0.5, 0.5],   # main AABB AND exclude AND precise            → moves (override)
    ])
    edit = RigidTransformAABB(
        name="t",
        target_aabb_frame="mocap",
        target_aabb_min=np.array([0.0, 0.0, 0.0]),
        target_aabb_max=np.array([1.0, 1.0, 1.0]),
        source_anchor=np.array([0.5, 0.5, 0.5]),
        target_anchor=np.array([10.0, 10.0, 10.0]),
        source_normal=np.array([1.0, 0.0, 0.0]),
        target_normal=np.array([1.0, 0.0, 0.0]),
        include_aabbs=(_Box(
            min=np.array([0.75, 0.0, 0.0]),
            max=np.array([0.85, 1.0, 1.0]),
        ),),
        exclude_aabbs=(_Box(
            min=np.array([0.6, 0.0, 0.0]),
            max=np.array([1.0, 1.0, 1.0]),
        ),),
    )
    new_pts, _ = apply_edits_to_arrays(pts, None, [edit], g)
    moved = np.linalg.norm(new_pts - pts, axis=1)
    assert moved[0] > 1.0, "main AABB, not excluded — should move"
    assert moved[1] < 1e-9, "main AABB AND excluded, no precise — should stay"
    assert moved[2] > 1.0, "in precise include — should override exclude and move"


def test_precise_include_overrides_oriented_exclude():
    """Same override rule for oriented exclude vs axis-aligned precise include."""
    from falsify.sim.scene_edits import (
        _Box, _OrientedBox, RigidTransformAABB, apply_edits_to_arrays,
    )
    g = _graph_perm5_with_ns_scale(scale=1.0)
    pts = np.array([
        [0.5, 0.5, 0.5],   # main + oriented_excl, no precise → stays
        [0.5, 0.5, 0.5],   # same point (duplicate to check determinism)
    ])
    # Point only stays if the oriented exclude actually catches it; with
    # yaw=0 a box centred at origin of side 2 covers [-1,1]^3.
    edit_stay = RigidTransformAABB(
        name="t",
        target_aabb_frame="mocap",
        target_aabb_min=np.array([0.0, 0.0, 0.0]),
        target_aabb_max=np.array([1.0, 1.0, 1.0]),
        source_anchor=np.array([0.5, 0.5, 0.5]),
        target_anchor=np.array([10.0, 10.0, 10.0]),
        source_normal=np.array([1.0, 0.0, 0.0]),
        target_normal=np.array([1.0, 0.0, 0.0]),
        oriented_exclude_aabbs=(_OrientedBox(
            center=np.array([0.5, 0.5, 0.5]),
            half_extents=np.array([0.4, 0.4, 0.4]),
            yaw=0.0,
        ),),
    )
    new_pts, _ = apply_edits_to_arrays(pts[:1], None, [edit_stay], g)
    assert np.linalg.norm(new_pts[0] - pts[0]) < 1e-9, "no precise rescue → stays"

    # Now add a precise include over the same point → must move.
    edit_move = RigidTransformAABB(
        **{**edit_stay.__dict__,
           "include_aabbs": (_Box(
               min=np.array([0.4, 0.4, 0.4]),
               max=np.array([0.6, 0.6, 0.6]),
           ),)},
    )
    new_pts, _ = apply_edits_to_arrays(pts[:1], None, [edit_move], g)
    assert np.linalg.norm(new_pts[0] - pts[0]) > 1.0, "precise include overrides oriented exclude"


def test_oriented_exclude_subtracts_from_inclusion():
    """An oriented exclude carves out a region that would otherwise be moved."""
    from falsify.sim.scene_edits import (
        _OrientedBox, RigidTransformAABB, apply_edits_to_arrays,
    )
    g = _graph_perm5_with_ns_scale(scale=1.0)
    pts = np.array([
        [0.5, 0.5, 0.5],     # in main AABB, NOT in oriented exclude → moves
        [0.2, 0.2, 0.5],     # in main AABB, INSIDE oriented exclude → stays
    ])
    edit = RigidTransformAABB(
        name="t",
        target_aabb_frame="mocap",
        target_aabb_min=np.array([0.0, 0.0, 0.0]),
        target_aabb_max=np.array([1.0, 1.0, 1.0]),
        source_anchor=np.array([0.5, 0.5, 0.5]),
        target_anchor=np.array([10.0, 10.0, 10.0]),
        source_normal=np.array([1.0, 0.0, 0.0]),
        target_normal=np.array([1.0, 0.0, 0.0]),
        oriented_exclude_aabbs=(_OrientedBox(
            center=np.array([0.2, 0.2, 0.5]),
            half_extents=np.array([0.1, 0.1, 0.5]),
            yaw=0.0,
        ),),
    )
    new_pts, _ = apply_edits_to_arrays(pts, None, [edit], g)
    moved = np.linalg.norm(new_pts - pts, axis=1)
    assert moved[0] > 1.0, "point outside oriented exclude should move"
    assert moved[1] < 1e-9, "point inside oriented exclude should stay"


def test_load_oriented_aabbs_from_yaml():
    cfg = {"scene_edits": [{
        "name": "t",
        "type": "rigid_transform_aabb",
        "target_aabb_frame": "mocap",
        "target_aabb_min": [0, 0, 0],
        "target_aabb_max": [1, 1, 1],
        "transform": {
            "source_anchor": [0.5, 0.5, 0.5],
            "target_anchor": [0.5, 0.5, 0.5],
            "source_normal": [1, 0, 0],
            "target_normal": [1, 0, 0],
        },
        "oriented_include_aabbs": [
            {"center": [1.5, 0.5, 1.0], "half_extents": [0.3, 0.05, 0.5], "yaw": 0.7},
        ],
        "oriented_exclude_aabbs": [
            {"center": [2.0, 0.0, 0.5], "half_extents": [0.1, 0.1, 0.1], "yaw": -1.2},
        ],
    }]}
    edits = load_scene_edits(cfg)
    assert len(edits) == 1
    e = edits[0]
    assert len(e.oriented_include_aabbs) == 1
    assert len(e.oriented_exclude_aabbs) == 1
    ob = e.oriented_include_aabbs[0]
    np.testing.assert_allclose(ob.center, [1.5, 0.5, 1.0])
    np.testing.assert_allclose(ob.half_extents, [0.3, 0.05, 0.5])
    assert ob.yaw == 0.7
