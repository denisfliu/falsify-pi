"""SplatNav recovery wrapper tests.

Uses a stub backend so we don't depend on splatnav/torch at test time. The
test asserts the wrapper's frame contract: NED in, NED out.
"""

from __future__ import annotations

import numpy as np
import pytest

from falsify.geometry import (
    COLMAP, MOCAP, NED, NS, Frame, FrameGraph, Point, SE3, Sim3, Trajectory,
)
from falsify.recovery import RecoveryConfig, RecoveryResult, SplatNavPlanner


def _build_graph():
    g = FrameGraph()
    for f in (NED, MOCAP, COLMAP, NS):
        g.register_frame(f)
    g.register_edge(SE3(R=np.diag([1.0, -1.0, -1.0]), t=np.zeros(3), src=NED, dst=MOCAP))
    g.register_edge(Sim3(s=1.7, R=np.eye(3), t=np.array([1.0, 2.0, -0.5]), src=MOCAP, dst=COLMAP))
    g.register_edge(Sim3(s=0.31, R=np.eye(3), t=np.zeros(3), src=COLMAP, dst=NS))
    return g


class _LineBackend:
    """Stub backend that returns a straight line of N waypoints in NS."""

    def __init__(self, n: int = 8):
        self.n = n
        self.calls = []

    def generate_path(self, x0_ns, xf_ns):
        self.calls.append((np.asarray(x0_ns).copy(), np.asarray(xf_ns).copy()))
        return np.linspace(x0_ns, xf_ns, self.n)


def test_recovery_plan_returns_ned_trajectory():
    g = _build_graph()
    backend = _LineBackend(n=10)
    planner = SplatNavPlanner(
        RecoveryConfig(bounds_lower_ned=[-2, -2, 0], bounds_upper_ned=[2, 2, 3]),
        g, backend=backend, horizon_s=2.0, hz=5,
    )
    start = Point.of(0.0, 0.0, 1.0, NED)
    goal = Point.of(1.0, 0.5, 1.2, NED)
    result = planner.plan(start, goal)
    assert isinstance(result, RecoveryResult)
    assert result.trajectory.frame.name == "ned"
    # Endpoint should be at (or very near) the goal — backend wrote a line
    # from start_ns to goal_ns; we convert back to NED so we recover goal.
    np.testing.assert_allclose(result.trajectory.positions[-1], goal.xyz, atol=1e-8)
    np.testing.assert_allclose(result.trajectory.positions[0], start.xyz, atol=1e-8)


def test_recovery_rejects_non_ned_inputs():
    g = _build_graph()
    planner = SplatNavPlanner(
        RecoveryConfig(bounds_lower_ned=[-1, -1, 0], bounds_upper_ned=[1, 1, 2]),
        g, backend=_LineBackend(),
    )
    with pytest.raises(ValueError, match="frame mismatch"):
        planner.plan(Point.of(0, 0, 0, MOCAP), Point.of(0, 0, 0, NED))


def test_recovery_backend_is_called_in_ns_frame():
    g = _build_graph()
    backend = _LineBackend(n=5)
    planner = SplatNavPlanner(
        RecoveryConfig(bounds_lower_ned=[-1, -1, 0], bounds_upper_ned=[1, 1, 2]),
        g, backend=backend,
    )
    start = Point.of(0.0, 0.0, 1.0, NED)
    goal = Point.of(0.5, 0.0, 1.0, NED)
    planner.plan(start, goal)
    assert len(backend.calls) == 1
    x0_ns, xf_ns = backend.calls[0]
    # Backend got NS coordinates — manually verify by converting the inputs.
    expected_x0 = g.convert(start, to="ns").xyz
    expected_xf = g.convert(goal, to="ns").xyz
    np.testing.assert_allclose(x0_ns, expected_x0, atol=1e-12)
    np.testing.assert_allclose(xf_ns, expected_xf, atol=1e-12)


def test_orchestrator_invokes_recovery_after_failure():
    """End-to-end: detector fires → recovery planner is invoked → trajectory
    attached to the FalsificationEpisode."""
    import numpy as np
    from pathlib import Path
    from falsify.io import build_frame_graph, load_yaml
    from falsify.orchestrator import EpisodeConfig, run_episode
    from falsify.policy import MockNoisy, MockNoisyConfig
    from falsify.safety import BoundsCriterion, FailureDetector

    repo = Path(__file__).resolve().parent.parent
    cfg = EpisodeConfig.from_yaml(
        scene_path=repo / "configs/scenes/left_gate.yaml",
        frame_path=repo / "configs/frames/carl_dual.yaml",
        episode_path=repo / "configs/policies/mock_noisy.yaml",
    )
    cfg.episode_cfg.setdefault("hz", 10)
    cfg.episode_cfg.setdefault("policy_hz", 1)
    cfg.episode_cfg.setdefault("horizon_s", 2.0)

    def policy_factory(goal, ep_cfg):
        return MockNoisy(MockNoisyConfig(
            goal=goal, speed=1.0, horizon_s=2.0, n_waypoints=20,
            position_noise_std=0.6, seed=1,
        ))

    def detector_factory(frame_graph, safety_cfg):
        ned = frame_graph.frame("ned")
        return FailureDetector(
            [BoundsCriterion(
                lower=Point.of(-0.2, -0.2, 1.0, ned),
                upper=Point.of(0.3, 0.3, 1.8, ned),
            )],
            frame_graph,
        )

    def recovery_factory(frame_graph, _cfg):
        return SplatNavPlanner(
            RecoveryConfig(bounds_lower_ned=[-2, -2, 0], bounds_upper_ned=[2, 2, 3]),
            frame_graph,
            backend=_LineBackend(n=12),
            horizon_s=2.0, hz=10,
        )

    ep = run_episode(
        cfg,
        policy_factory=policy_factory,
        detector_factory=detector_factory,
        recovery_factory=recovery_factory,
    )
    assert ep.failure is not None
    assert ep.recovery_trajectory is not None
    assert ep.recovery_trajectory.frame.name == "ned"
    # Recovery line connects last-safe → goal after a NED→NS→NED round-trip
    # through the real left_gate transforms; tolerance reflects Sim3 chaining.
    np.testing.assert_allclose(
        ep.recovery_trajectory.positions[0],
        ep.failure.last_safe_state.pos.xyz,
        atol=1e-6,
    )
    assert ep.goal is not None and ep.goal.frame.name == "ned"
    np.testing.assert_allclose(
        ep.recovery_trajectory.positions[-1],
        ep.goal.xyz,
        atol=1e-6,
    )
