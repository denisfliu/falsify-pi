"""Visualization tests — ply writer + frame-aware episode dump."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from falsify.geometry import COLMAP, FrameGraph, MOCAP, NED, NS, Point, Sim3, SE3, Trajectory
from falsify.io import build_frame_graph
from falsify.policy import MockStraightLine, MockStraightLineConfig
from falsify.sensors import SensorRig, StateSensor
from falsify.sim import DroneState, Simulator, SimulatorConfig
from falsify.orchestrator import FalsificationEpisode
from falsify.visualization import (
    dump_episode, dump_trajectory_in_frames, trajectory_to_pointcloud, write_ply,
)


def _build_graph():
    """Tiny graph that mirrors the left_gate transforms enough for visualization."""
    g = FrameGraph()
    for f in (NED, MOCAP, COLMAP, NS):
        g.register_frame(f)
    g.register_edge(SE3(R=np.diag([1.0, -1.0, -1.0]), t=np.zeros(3), src=NED, dst=MOCAP))
    g.register_edge(Sim3(s=1.5, R=np.eye(3), t=np.array([0.1, 0.0, 0.0]), src=MOCAP, dst=COLMAP))
    g.register_edge(Sim3(s=0.4, R=np.eye(3), t=np.zeros(3), src=COLMAP, dst=NS))
    return g


def test_ply_writer_records_frame_in_header(tmp_path: Path):
    pc = trajectory_to_pointcloud(
        Trajectory(times=np.array([0, 1]), positions=np.array([[0, 0, 0], [1, 1, 1]]), frame=NED),
        color=(0.5, 0.5, 0.5),
    )
    p = write_ply(pc, tmp_path / "x.ply")
    assert p.exists()
    text = p.read_text()
    assert "comment falsify frame: ned" in text
    assert "element vertex 2" in text


def test_dump_trajectory_in_frames_writes_one_per_frame(tmp_path: Path):
    g = _build_graph()
    traj = Trajectory(
        times=np.linspace(0, 1, 5),
        positions=np.linspace([0, 0, 1], [1, 0, 1], 5),
        frame=NED,
    )
    written = dump_trajectory_in_frames(
        traj, g, tmp_path, target_frames=("ned", "mocap", "ns"),
    )
    assert set(written.keys()) == {"ned", "mocap", "ns"}
    for f, path in written.items():
        assert path.exists()
        assert f"comment falsify frame: {f}" in path.read_text()


def test_dump_episode_handles_minimal_inputs(tmp_path: Path):
    g = _build_graph()
    sim = Simulator(SimulatorConfig(hz=10, horizon_s=1.0, policy_hz=1), g)
    start = DroneState(
        pos=Point.of(0, 0, 1.0, NED), vel=np.zeros(3),
        quat_xyzw=np.array([0, 0, 0, 1.0]), t=0.0,
    )
    sim.reset(start)
    policy = MockStraightLine(MockStraightLineConfig(
        goal=Point.of(0.5, 0, 1.0, NED), speed=0.5, n_waypoints=10,
    ))
    rig = SensorRig([StateSensor()])
    trace = sim.rollout_with_policy(policy, rig)

    ep = FalsificationEpisode(
        scene_cfg={}, frame_cfg={}, episode_cfg={},
        trace=trace,
        goal=Point.of(0.5, 0.0, 1.0, NED),
    )
    written = dump_episode(ep, g, tmp_path, target_frames=("ned", "mocap", "ns"))
    assert "nominal" in written and "markers" in written
    for entity, paths in written.items():
        for f, p in paths.items():
            assert p.exists()


def test_html_replay_optional(tmp_path: Path):
    """The html viewer either writes a file (plotly present) or returns None."""
    from falsify.visualization import html_replay
    g = _build_graph()
    sim = Simulator(SimulatorConfig(hz=10, horizon_s=0.5, policy_hz=1), g)
    start = DroneState(
        pos=Point.of(0, 0, 1.0, NED), vel=np.zeros(3),
        quat_xyzw=np.array([0, 0, 0, 1.0]), t=0.0,
    )
    sim.reset(start)
    policy = MockStraightLine(MockStraightLineConfig(
        goal=Point.of(0.5, 0, 1.0, NED), speed=0.5, n_waypoints=5,
    ))
    rig = SensorRig([StateSensor()])
    trace = sim.rollout_with_policy(policy, rig)
    ep = FalsificationEpisode(
        scene_cfg={}, frame_cfg={}, episode_cfg={}, trace=trace,
        goal=Point.of(0.5, 0.0, 1.0, NED),
    )
    out = html_replay(ep, g, tmp_path / "ep.html", view_frame="ned")
    if out is not None:
        assert out.exists()
        # plotly was available; verify the html mentions the chosen frame.
        text = out.read_text()
        assert "ned" in text
