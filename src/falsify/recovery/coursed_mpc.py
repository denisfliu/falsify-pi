"""Course-driven MPC recovery planner.

When the failure detector fires on a falsification-triggered failure type
(``MISS_GATE``, ``COLLISION_*``, ``OUT_OF_BOUNDS``), this planner builds
a dynamically-feasible recovery trajectory by tracking the original
course's remaining waypoints from the drone's ``last_safe_state``.

Same ``planner.plan(start, goal) → RecoveryResult`` interface
``SplatNavPlanner`` exposes, so it drops into the orchestrator's
``recovery_factory`` slot with no further changes.

Internally:

1. Load the course YAML (e.g. ``configs/courses/through_left_gate.yaml``).
2. Replace the first waypoint with the drone's current position so the
   MPC starts at ``last_safe_state``. Velocity is treated as zero
   (we're "stuck" at the last safe pose); yaw is taken from the original
   first waypoint.
3. Hand the rewritten course to ``plan_mpc``. The MPC's min-snap
   reference tracks the gate, then the goal.
4. Return ``RecoveryResult(trajectory=NED traj)``.

Spline-mode is supported as a quick fallback (``--planner spline`` in the
recovery YAML) so the same wiring works without acados in CI / smoke
contexts.
"""

from __future__ import annotations

import dataclasses
from dataclasses import replace
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from falsify.geometry import FrameGraph, Point, Trajectory, assert_frame
from falsify.planning import (
    Course,
    Waypoint,
    load_course,
    plan_mpc,
    plan_spline,
)

from .planner import RecoveryResult


class CoursedMpcPlanner:
    """Recovery planner that tracks a course YAML from ``last_safe_state``."""

    def __init__(
        self,
        course_path: str | Path,
        frame_graph: FrameGraph,
        *,
        planner: Literal["mpc", "spline"] = "mpc",
        mpc_frame_cfg: dict | str | Path | None = None,
        prompt: str = "",
    ) -> None:
        self.course_path = Path(course_path)
        self.frame_graph = frame_graph
        self.planner_kind = planner
        self.mpc_frame_cfg = mpc_frame_cfg
        self.prompt = prompt
        self._course: Optional[Course] = None

    # ---- public API ---------------------------------------------------

    def plan(self, start: Point, goal: Point) -> RecoveryResult:
        assert_frame(start, "ned")
        assert_frame(goal, "ned")

        course = self._load_course()
        rewritten = self._replace_start_waypoint(course, start)

        if self.planner_kind == "spline":
            traj_training = plan_spline(rewritten, self.frame_graph, prompt=self.prompt)
        elif self.planner_kind == "mpc":
            # Build a 10-vector start state in NED (pos + zero vel + course yaw).
            start_state_ned = self._initial_state_from(course, start)
            # Recovery honours the course's authored total_time_s — disable
            # min-time-snap's time re-optimisation (`kT=None`). Otherwise
            # the planner compresses a 24 s schedule to ~8.7 s and the MPC
            # has to chase an over-aggressive reference.
            from falsify.planning.mpc import _DEFAULT_POLICY_CFG
            policy_cfg = {
                "plan": {"kT": None, "use_l2_time": False},
                "track": _DEFAULT_POLICY_CFG["track"],
            }
            traj_training = plan_mpc(
                rewritten,
                self.frame_graph,
                prompt=self.prompt,
                start_state_ned=start_state_ned,
                frame_cfg=self.mpc_frame_cfg,
                policy_cfg=policy_cfg,
            )
        else:
            raise ValueError(f"unknown planner {self.planner_kind!r}")

        # falsify.training.Trajectory has positions_ned + quats_xyzw fields;
        # the orchestrator wants a falsify.geometry.Trajectory with .frame
        # and .positions. Adapt.
        ned_frame = self.frame_graph.frame("ned")
        traj_ned = Trajectory(
            times=np.asarray(traj_training.times, dtype=np.float64),
            positions=np.asarray(traj_training.positions_ned, dtype=np.float64),
            frame=ned_frame,
            quaternions=np.asarray(traj_training.quaternions_xyzw, dtype=np.float64),
        )
        assert_frame(traj_ned, "ned")
        return RecoveryResult(trajectory=traj_ned, info={"planner": self.planner_kind})

    # ---- internals ----------------------------------------------------

    def _load_course(self) -> Course:
        if self._course is None:
            self._course = load_course(self.course_path)
        return self._course

    def _replace_start_waypoint(self, course: Course, start_ned: Point) -> Course:
        """Return a copy of ``course`` whose first waypoint sits at ``start_ned``.

        The course's authored frame may differ from NED; we convert
        ``start`` into the course's frame so the waypoint stays in the
        same coordinate system as the rest.
        """
        start_in_course_frame = self.frame_graph.convert(start_ned, to=course.frame).xyz
        original_first = course.waypoints[0]
        # Keep yaw + t (=0) from the original first waypoint — only swap pos.
        rewritten_first = Waypoint(
            name=original_first.name,
            p=np.asarray(start_in_course_frame, dtype=np.float64),
            yaw=original_first.yaw,
            t=original_first.t,
        )
        new_waypoints = (rewritten_first,) + tuple(course.waypoints[1:])
        return replace(course, waypoints=new_waypoints)

    def _initial_state_from(self, course: Course, start_ned: Point) -> np.ndarray:
        """Build a 10-vector ``[px, py, pz, vx, vy, vz, qx, qy, qz, qw]`` in
        NED. Velocity = 0 (drone is "stuck"); yaw from the course's first
        waypoint, converted through the FrameGraph."""
        # Reuse plan_spline's helper for yaw-frame conversion.
        from falsify.planning.spline import _to_ned_yaw, _yaw_to_quat_xyzw
        yaw_src = course.resolved_yaws()[0]
        yaw_ned = float(_to_ned_yaw(np.array([yaw_src]), course.frame, self.frame_graph)[0])
        x0 = np.zeros(10, dtype=np.float64)
        x0[0:3] = start_ned.xyz
        x0[3:6] = 0.0
        x0[6:10] = _yaw_to_quat_xyzw(yaw_ned)
        return x0
