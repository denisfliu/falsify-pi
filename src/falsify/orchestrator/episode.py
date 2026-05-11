"""`FalsificationEpisode` — the per-episode record returned by `run_episode`.

For v0 we keep the schema small: configs used + the rollout trace + (later)
the failure record and recovery trace. Persistence to parquet/JSON is a
Phase-4+ concern (`falsify.io.episode_store`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from falsify.geometry import Point, Trajectory
from falsify.sim import EpisodeTrace


@dataclass
class FalsificationEpisode:
    """Result of one falsification episode.

    Attributes
    ----------
    scene_cfg, frame_cfg, episode_cfg
        Frozen copies of the inputs so the episode is self-describing.
    trace
        Time-stamped rollout trace from `Simulator.rollout_with_policy`.
    failure
        Set by the failure detector (Phase 4). ``None`` if no failure was
        detected.
    recovery_trajectory, recovery_trace
        Populated by the recovery planner (Phase 5).
    metadata
        Free-form dict for derived info (seed, wall-clock duration, etc.).
    """

    scene_cfg: dict
    frame_cfg: dict
    episode_cfg: dict
    trace: EpisodeTrace
    goal: Optional[Point] = None    # frame-tagged; ``None`` only for tests
    failure: Optional[Any] = None
    recovery_trajectory: Optional[Trajectory] = None
    recovery_trace: Optional[EpisodeTrace] = None
    metadata: dict = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.failure is None

    def summary(self) -> str:
        n = len(self.trace.states)
        out = [f"Episode: {n} state samples, {len(self.trace.policy_outputs)} policy queries"]
        if self.failure is not None:
            out.append(f"  FAILURE: {self.failure}")
            if self.recovery_trajectory is not None:
                out.append(f"  recovery: {len(self.recovery_trajectory)} waypoints")
        else:
            out.append("  no failure detected")
        return "\n".join(out)
