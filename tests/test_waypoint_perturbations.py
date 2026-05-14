"""Tests for the waypoint perturbation utilities."""

from __future__ import annotations

import numpy as np
import pytest

from falsify.planning import (
    Course, Waypoint, perturb_waypoint, sample_variants, save_course, load_course,
)


def _base_course() -> Course:
    return Course(
        name="test",
        frame="mocap",
        fps=10,
        total_time_s=4.0,
        yaw_mode="tangent",
        waypoints=(
            Waypoint(name="start",    p=np.array([0.0, 0.0, 1.5])),
            Waypoint(name="approach", p=np.array([1.0, 0.0, 1.5])),  # heading: +x
            Waypoint(name="end",      p=np.array([2.0, 0.0, 1.5])),
        ),
    )


def test_perturb_center_is_noop():
    c = _base_course()
    out = perturb_waypoint(c, "approach", "center", 0.5)
    np.testing.assert_allclose(out.waypoints[1].p, c.waypoints[1].p)


def test_perturb_up_down():
    c = _base_course()
    up = perturb_waypoint(c, "approach", "up", 0.3)
    down = perturb_waypoint(c, "approach", "down", 0.3)
    np.testing.assert_allclose(up.waypoints[1].p, [1.0, 0.0, 1.8])
    np.testing.assert_allclose(down.waypoints[1].p, [1.0, 0.0, 1.2])


def test_perturb_body_relative_left_right_along_x_heading():
    """With heading = +x, left should be +y and right should be -y (right-hand-rule about up)."""
    c = _base_course()
    left = perturb_waypoint(c, "approach", "left", 0.4)
    right = perturb_waypoint(c, "approach", "right", 0.4)
    np.testing.assert_allclose(left.waypoints[1].p, [1.0, 0.4, 1.5], atol=1e-9)
    np.testing.assert_allclose(right.waypoints[1].p, [1.0, -0.4, 1.5], atol=1e-9)


def test_perturb_body_relative_left_right_along_y_heading():
    """With heading = +y, left should be -x and right should be +x."""
    c = Course(
        name="t", frame="mocap", fps=10, total_time_s=2.0, yaw_mode="tangent",
        waypoints=(
            Waypoint(name="a", p=np.array([0.0, 0.0, 1.5])),
            Waypoint(name="b", p=np.array([0.0, 1.0, 1.5])),
            Waypoint(name="c", p=np.array([0.0, 2.0, 1.5])),
        ),
    )
    left = perturb_waypoint(c, "b", "left", 0.3)
    right = perturb_waypoint(c, "b", "right", 0.3)
    np.testing.assert_allclose(left.waypoints[1].p, [-0.3, 1.0, 1.5], atol=1e-9)
    np.testing.assert_allclose(right.waypoints[1].p, [0.3, 1.0, 1.5], atol=1e-9)


def test_perturb_does_not_mutate_base():
    c = _base_course()
    original = c.waypoints[1].p.copy()
    _ = perturb_waypoint(c, "approach", "left", 0.5)
    np.testing.assert_allclose(c.waypoints[1].p, original)


def test_sample_variants_reproducible():
    c = _base_course()
    a = sample_variants(c, "approach", n_per_mode=3, seed=42)
    b = sample_variants(c, "approach", n_per_mode=3, seed=42)
    assert [v.label for v in a] == [v.label for v in b]
    for va, vb in zip(a, b):
        assert va.magnitude_m == vb.magnitude_m
        np.testing.assert_allclose(va.course.waypoints[1].p, vb.course.waypoints[1].p)


def test_sample_variants_center_always_zero_magnitude():
    c = _base_course()
    variants = sample_variants(
        c, "approach",
        modes=("center", "left", "right"),
        magnitude_range_m=(0.1, 1.0),
        n_per_mode=2, seed=0,
    )
    for v in variants:
        if v.direction == "center":
            assert v.magnitude_m == 0.0
            np.testing.assert_allclose(v.course.waypoints[1].p, c.waypoints[1].p)
        else:
            assert 0.1 <= v.magnitude_m <= 1.0


def test_sample_variants_labels_sort_correctly():
    c = _base_course()
    variants = sample_variants(c, "approach", n_per_mode=12, seed=0)
    labels = [v.label for v in variants]
    # With zero-padded labels, lexicographic sort matches numeric sort within a mode.
    for mode in ("center", "up", "down", "left", "right"):
        mode_labels = [l for l in labels if l.startswith(mode + "_")]
        assert mode_labels == sorted(mode_labels)


def test_save_load_course_roundtrip(tmp_path):
    """save_course followed by load_course should produce a structurally-equal course."""
    c = _base_course()
    out = save_course(c, tmp_path / "course.yaml")
    loaded = load_course(out)
    assert loaded.name == c.name
    assert loaded.frame == c.frame
    assert loaded.fps == c.fps
    assert loaded.total_time_s == c.total_time_s
    assert loaded.yaw_mode == c.yaw_mode
    assert [w.name for w in loaded.waypoints] == [w.name for w in c.waypoints]
    for a, b in zip(loaded.waypoints, c.waypoints):
        np.testing.assert_allclose(a.p, b.p)
