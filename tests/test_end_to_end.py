"""Full-pipeline end-to-end test (no FiGS / CUDA / OpenPI / splatnav required).

Exercises: scene YAML → FrameGraph → sensor rig → mock policy → perturbations
→ failure detection → stubbed recovery → frame-aware visualization dumps.

This is the test the CI runs to make sure the architecture is wired correctly
on every commit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from falsify.geometry import Point
from falsify.orchestrator import EpisodeConfig, run_episode
from falsify.perturbations import (
    PerturbationSuite, PositionBias, PositionNoise, StateNoise,
)
from falsify.policy import MockNoisy, MockNoisyConfig
from falsify.recovery import RecoveryConfig, SplatNavPlanner
from falsify.safety import BoundsCriterion, FailureDetector
from falsify.visualization import dump_episode, html_replay


REPO = Path(__file__).resolve().parent.parent


class _StubBackend:
    def __init__(self, n=20):
        self.n = n
    def generate_path(self, x0_ns, xf_ns):
        return np.linspace(np.asarray(x0_ns), np.asarray(xf_ns), self.n)


def test_full_pipeline_with_failure_and_stub_recovery(tmp_path):
    """One-shot: configure, run, dump, verify every artifact carries a frame."""
    cfg = EpisodeConfig.from_yaml(
        scene_path=REPO / "configs/scenes/left_gate.yaml",
        frame_path=REPO / "configs/frames/carl_dual.yaml",
        episode_path=REPO / "configs/policies/mock_noisy.yaml",
    )
    cfg.episode_cfg.setdefault("hz", 10)
    cfg.episode_cfg.setdefault("policy_hz", 1)
    cfg.episode_cfg.setdefault("horizon_s", 3.0)

    def policy_factory(goal: Point, ep_cfg):
        return MockNoisy(MockNoisyConfig(
            goal=goal, speed=1.0, horizon_s=3.0, n_waypoints=30,
            position_noise_std=0.4, seed=2,
        ))

    def detector_factory(fg, _cfg):
        ned = fg.frame("ned")
        return FailureDetector(
            [BoundsCriterion(
                lower=Point.of(-0.2, -0.2, 1.0, ned),
                upper=Point.of(0.4, 0.4, 1.7, ned),
            )],
            fg,
        )

    def recovery_factory(fg, _cfg):
        return SplatNavPlanner(
            RecoveryConfig(bounds_lower_ned=[-2, -2, 0], bounds_upper_ned=[2, 2, 3]),
            fg, backend=_StubBackend(n=25),
        )

    def perturbations_factory(_fg, _cfg):
        return PerturbationSuite(
            observation=[StateNoise(std=0.01)],
            action=[PositionNoise(std=0.02), PositionBias(bias_xyz=(0.0, 0.05, 0.0))],
            seed=0,
        )

    ep = run_episode(
        cfg,
        policy_factory=policy_factory,
        detector_factory=detector_factory,
        recovery_factory=recovery_factory,
        perturbations_factory=perturbations_factory,
    )

    # ---- frame-awareness assertions -----------------------------------

    # 1. Trace is all NED.
    assert all(s.pos.frame.name == "ned" for s in ep.trace.states)

    # 2. Every policy output is NED (perturbations preserve frame too).
    assert all(t.frame.name == "ned" for t in ep.trace.policy_outputs)

    # 3. Failure record's snapshots carry the frame tag.
    assert ep.failure is not None
    assert ep.failure.failure_state.pos.frame.name == "ned"
    assert ep.failure.last_safe_state.pos.frame.name == "ned"

    # 4. Recovery trajectory is NED.
    assert ep.recovery_trajectory is not None
    assert ep.recovery_trajectory.frame.name == "ned"

    # 5. Goal is a frame-tagged Point on the episode.
    assert ep.goal is not None and ep.goal.frame.name == "ned"

    # 6. Perturbation manifest is in metadata.
    assert ep.metadata["perturbations"]["seed"] == 0
    assert any(p["type"] == "PositionBias"
               for p in ep.metadata["perturbations"]["action"])

    # ---- visualization end-to-end -------------------------------------

    from falsify.io import build_frame_graph
    fg = build_frame_graph(cfg.scene_cfg, base_path=cfg.scene_cfg_dir)

    plys = dump_episode(ep, fg, tmp_path / "frames",
                        target_frames=("ned", "mocap", "ns"))
    # 7. Every entity is dumped in every target frame.
    for entity in ("nominal", "recovery", "markers"):
        assert entity in plys
        for f in ("ned", "mocap", "ns"):
            assert f in plys[entity]
            assert plys[entity][f].exists()
            # Frame name written into the PLY header.
            text = plys[entity][f].read_text()
            assert f"comment falsify frame: {f}" in text

    # 8. html_replay returns a path when plotly is present (else None — both OK).
    html = html_replay(ep, fg, tmp_path / "ep.html", view_frame="mocap")
    if html is not None:
        assert html.exists()
        assert "mocap" in html.read_text()


def test_smoke_cli_help_does_not_import_heavy_deps():
    """Importing the CLI module must not pull in CUDA / FiGS / openpi."""
    # If the module imports cleanly, lazy-imports are working as intended.
    import falsify.cli.smoke_test as st
    assert hasattr(st, "main")
