"""`run_episode` — one falsification episode from configs to result.

For v0 we exercise the vertical slice: scene YAML → `FrameGraph` → simulator
+ mock policy + sensor rig → trace. Failure detection + recovery are
plumbed in via optional parameters so Phase 4 / 5 can drop in without
churning callers.

Frame contract
--------------
Initial state is built in ``"ned"``. Goal positions in the scene config are
declared in ``"mocap"`` and converted to ``"ned"`` here via the `FrameGraph`.
Trajectories returned by the policy and stored on `EpisodeTrace` are always
in ``"ned"``. The whole episode is therefore self-consistent and the
geometry layer guarantees nothing leaks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from falsify.geometry import (
    FrameGraph,
    Point,
    Trajectory,
    assert_frame,
)
from falsify.io import load_yaml, build_frame_graph
from falsify.policy import Policy
from falsify.sensors import build_sensor_rig
from falsify.sim import DroneState, Simulator, SimulatorConfig, EpisodeTrace
from falsify.sim.poses import camera_to_world_pose

from .episode import FalsificationEpisode


# ---------------------------------------------------------------------------
# Episode configuration
# ---------------------------------------------------------------------------


@dataclass
class EpisodeConfig:
    """Top-level run config.

    Either pass the dicts directly or use `EpisodeConfig.from_yaml`.
    """
    scene_cfg: dict
    frame_cfg: dict
    episode_cfg: dict
    scene_cfg_dir: Path = field(default_factory=Path.cwd)

    @classmethod
    def from_yaml(
        cls,
        scene_path: str | Path,
        frame_path: str | Path,
        episode_path: str | Path,
    ) -> "EpisodeConfig":
        scene_path = Path(scene_path)
        return cls(
            scene_cfg=load_yaml(scene_path),
            frame_cfg=load_yaml(frame_path),
            episode_cfg=load_yaml(episode_path),
            scene_cfg_dir=scene_path.parent,
        )


# ---------------------------------------------------------------------------
# Wiring helpers — split out so each is easy to test in isolation.
# ---------------------------------------------------------------------------


def build_initial_state(scene_cfg: dict, frame_graph: FrameGraph) -> DroneState:
    """Translate the scene's MOCAP start position into a NED `DroneState`."""
    start_mocap = scene_cfg["start_position_mocap"]
    start = Point.of(*start_mocap, frame_graph.frame("mocap"))
    start_ned = frame_graph.convert(start, to="ned")
    assert_frame(start_ned, "ned")
    return DroneState(
        pos=start_ned,
        vel=np.zeros(3),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        t=0.0,
    )


def goal_in_ned(scene_cfg: dict, frame_graph: FrameGraph) -> Point:
    goal_mocap = Point.of(*scene_cfg["goal_position_mocap"], frame_graph.frame("mocap"))
    return frame_graph.convert(goal_mocap, to="ned")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_episode(
    cfg: EpisodeConfig,
    *,
    policy_factory: Callable[[Point, dict], Policy],
    renderer: Optional[Callable] = None,
    detector_factory: Optional[Callable[[FrameGraph, dict], Any]] = None,
    recovery_factory: Optional[Callable[[FrameGraph, dict], Any]] = None,
    perturbations_factory: Optional[Callable[[FrameGraph, dict], Any]] = None,
) -> FalsificationEpisode:
    """Run one episode end-to-end.

    Parameters
    ----------
    cfg
        Loaded `EpisodeConfig`.
    policy_factory
        Callable ``(goal_ned: Point, policy_cfg: dict) → Policy``. Decoupling
        policy construction from this function lets callers wire mock or VLA
        policies symmetrically.
    renderer
        Camera renderer (e.g. `GSplatRenderer.render`). Only needed if the
        policy declares image/depth modalities.
    detector
        Phase-4 placeholder; ignored for v0.
    recovery
        Phase-5 placeholder; ignored for v0.

    Returns
    -------
    A populated `FalsificationEpisode`.
    """
    frame_graph = build_frame_graph(cfg.scene_cfg, base_path=cfg.scene_cfg_dir)

    initial_state = build_initial_state(cfg.scene_cfg, frame_graph)
    goal = goal_in_ned(cfg.scene_cfg, frame_graph)

    policy = policy_factory(goal, cfg.episode_cfg.get("policy", {}))

    sensor_rig = build_sensor_rig(
        policy.required_modalities,
        frame_graph=frame_graph,
        frame_cfg=cfg.frame_cfg,
        renderer=renderer,
        body_to_world=camera_to_world_pose,
        prompt=cfg.scene_cfg.get("prompt"),
    )

    sim_cfg = SimulatorConfig(
        hz=int(cfg.episode_cfg.get("hz", 10)),
        horizon_s=float(cfg.episode_cfg.get("horizon_s", 5.0)),
        policy_hz=int(cfg.episode_cfg.get("policy_hz", 1)),
    )
    sim = Simulator(sim_cfg, frame_graph)
    sim.reset(initial_state)

    detector = None
    if detector_factory is not None:
        detector = detector_factory(frame_graph, cfg.episode_cfg.get("safety", {}))

    perturbations = None
    if perturbations_factory is not None:
        perturbations = perturbations_factory(frame_graph, cfg.episode_cfg.get("perturbations", {}))

    trace = sim.rollout_with_policy(
        policy, sensor_rig, detector=detector, perturbations=perturbations,
    )

    recovery_traj: Optional[Trajectory] = None
    recovery_trace: Optional[EpisodeTrace] = None
    if trace.failure is not None and recovery_factory is not None:
        planner = recovery_factory(frame_graph, cfg.episode_cfg.get("recovery", {}))
        last_safe = trace.failure.last_safe_state.pos
        result = planner.plan(last_safe, goal)
        recovery_traj = result.trajectory

    return FalsificationEpisode(
        scene_cfg=cfg.scene_cfg,
        frame_cfg=cfg.frame_cfg,
        episode_cfg=cfg.episode_cfg,
        trace=trace,
        goal=goal,
        failure=trace.failure,
        recovery_trajectory=recovery_traj,
        recovery_trace=recovery_trace,
        metadata={
            "perturbations": perturbations.manifest() if perturbations is not None else None,
        },
    )
