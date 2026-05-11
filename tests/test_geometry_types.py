"""Frame-tag enforcement at module boundaries."""

import numpy as np
import pytest

from falsify.geometry import (
    Frame, NED, MOCAP,
    Point, Pose, Trajectory, PointCloud,
    assert_frame,
)


def test_point_carries_frame():
    p = Point.of(1, 2, 3, NED)
    assert p.frame is NED
    assert p.xyz.shape == (3,)


def test_assert_frame_passes_on_match():
    p = Point.of(0, 0, 0, NED)
    assert_frame(p, NED)
    assert_frame(p, "ned")


def test_assert_frame_raises_on_mismatch():
    p = Point.of(0, 0, 0, NED)
    with pytest.raises(ValueError, match="frame mismatch"):
        assert_frame(p, MOCAP)
    with pytest.raises(ValueError, match="frame mismatch"):
        assert_frame(p, "mocap")


def test_pose_shape_validation():
    with pytest.raises(ValueError):
        Pose(R=np.eye(2), t=np.zeros(3), frame=NED)
    with pytest.raises(ValueError):
        Pose(R=np.eye(3), t=np.zeros(2), frame=NED)


def test_pose_round_trip_matrix():
    R = np.eye(3)
    t = np.array([1.0, 2.0, 3.0])
    p = Pose(R=R, t=t, frame=NED)
    M = p.as_matrix()
    assert M.shape == (4, 4)
    np.testing.assert_array_equal(M[:3, 3], t)
    p2 = Pose.from_matrix(M, NED)
    np.testing.assert_array_equal(p.R, p2.R)
    np.testing.assert_array_equal(p.t, p2.t)


def test_trajectory_validation():
    Trajectory(
        times=np.array([0.0, 1.0]),
        positions=np.array([[0, 0, 0], [1, 0, 0]], dtype=float),
        frame=NED,
    )
    with pytest.raises(ValueError):
        Trajectory(times=np.zeros(3), positions=np.zeros((2, 3)), frame=NED)


def test_pointcloud_validation():
    PointCloud(points=np.zeros((5, 3)), frame=NED)
    with pytest.raises(ValueError):
        PointCloud(points=np.zeros((5, 2)), frame=NED)


def test_frame_value_equality():
    f1 = Frame("custom")
    f2 = Frame("custom")
    assert f1 == f2
    assert hash(f1) == hash(f2)
