"""End-to-end orchestrator smoke test using only repository configs.

Does NOT require FiGS / nerfstudio / acados — the mock policies declare no
camera requirements, so the sensor rig contains only `StateSensor`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from falsify.geometry import Point
from falsify.io import load_yaml
from falsify.orchestrator import EpisodeConfig, run_episode
from falsify.policy import MockStraightLine, MockStraightLineConfig


REPO = Path(__file__).resolve().parent.parent


def _factory_mock_straight(goal: Point, ep_cfg: dict):
    return MockStraightLine(MockStraightLineConfig(
        goal=goal,
        speed=0.5,
        horizon_s=ep_cfg.get("horizon_s", 5.0),
        n_waypoints=50,
    ))


def test_smoke_episode_runs_on_left_gate_config():
    cfg = EpisodeConfig.from_yaml(
        scene_path=REPO / "configs/scenes/left_gate.yaml",
        frame_path=REPO / "configs/frames/carl_dual.yaml",
        episode_path=REPO / "configs/policies/mock_straight_line.yaml",
    )
    cfg.episode_cfg.setdefault("hz", 10)
    cfg.episode_cfg.setdefault("policy_hz", 1)
    cfg.episode_cfg.setdefault("horizon_s", 3.0)
    ep = run_episode(cfg, policy_factory=_factory_mock_straight)

    # Every state in the trace is frame-tagged in NED.
    assert all(s.pos.frame.name == "ned" for s in ep.trace.states)
    # The trajectory progresses from start toward the converted goal.
    start = ep.trace.states[0].pos.xyz
    end = ep.trace.states[-1].pos.xyz
    assert ep.goal is not None and ep.goal.frame.name == "ned"
    goal = ep.goal.xyz
    assert np.linalg.norm(end - goal) <= np.linalg.norm(start - goal) + 1e-9
    # Frame-tagged full trajectory comes back.
    assert ep.trace.trajectory().frame.name == "ned"


def test_policy_requiring_cameras_without_renderer_fails_clearly():
    cfg = EpisodeConfig.from_yaml(
        scene_path=REPO / "configs/scenes/left_gate.yaml",
        frame_path=REPO / "configs/frames/carl_dual.yaml",
        episode_path=REPO / "configs/policies/mock_straight_line.yaml",
    )

    class HungryPolicy(MockStraightLine):
        required_modalities = frozenset({"images.forward"})

    def factory(goal: Point, ep_cfg: dict):
        return HungryPolicy(MockStraightLineConfig(goal=goal))

    with pytest.raises(ValueError, match="no renderer"):
        run_episode(cfg, policy_factory=factory)
