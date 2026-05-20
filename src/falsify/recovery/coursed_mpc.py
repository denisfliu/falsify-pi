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
        gate_deltas: Optional[dict] = None,
        scene_cfg: Optional[dict] = None,
    ) -> None:
        """
        Parameters
        ----------
        gate_deltas
            Optional per-episode `GateRigidPerturbation` Δ, in the
            ``{anchor_mocap, delta_xyz_mocap, delta_yaw_rad}`` shape the
            orchestrator stamps on `episode.metadata['gate_deltas']`.
            When provided + `scene_cfg.gate_region` is set, every course
            waypoint whose nominal position falls inside the **gate's
            MOCAP AABB** is rigid-transformed by the same Δ so the
            recovery MPC threads through the *perturbed* gate, not the
            nominal one. Waypoints outside the gate AABB (start, post-gate
            corridor, hover) stay put. No-op when either kwarg is omitted.
        scene_cfg
            Parsed scene YAML — read for ``gate_region.aabb_min/max`` (the
            gate's MOCAP AABB) to decide which course waypoints to
            perturb. Optional; needed only when ``gate_deltas`` is set.
        """
        self.course_path = Path(course_path)
        self.frame_graph = frame_graph
        self.planner_kind = planner
        self.mpc_frame_cfg = mpc_frame_cfg
        self.prompt = prompt
        self.gate_deltas = gate_deltas
        self.scene_cfg = scene_cfg
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
            course = load_course(self.course_path)
            # If a gate perturbation is active, lift the course's
            # gate-region waypoints through the same MOCAP rigid transform
            # the renderer applies to the gate Gaussians. Without this,
            # the recovery MPC threads through the *nominal* gate, which
            # is no longer where the gate is in the rendered scene.
            if self.gate_deltas is not None and self.scene_cfg is not None:
                course = self._apply_gate_deltas_to_course(course)
            self._course = course
        return self._course

    def _apply_gate_deltas_to_course(self, course: Course) -> Course:
        """Rigid-transform every waypoint that lies inside the gate's
        MOCAP AABB by the active `gate_deltas`. Waypoints outside the
        AABB (start, post-gate corridor, hover targets) stay put."""
        region = (self.scene_cfg or {}).get("gate_region")
        if not region:
            return course
        if region.get("aabb_frame", "mocap") != "mocap":
            raise NotImplementedError(
                "CoursedMpcPlanner: gate_region.aabb_frame=='mocap' only"
            )
        aabb_min = np.asarray(region["aabb_min"], dtype=np.float64)
        aabb_max = np.asarray(region["aabb_max"], dtype=np.float64)
        # Transform parameters — all MOCAP. We rotate about the anchor,
        # then translate by delta_xyz (same convention as
        # `GateRigidPerturbation._build_edit`).
        anchor = np.asarray(self.gate_deltas["anchor_mocap"], dtype=np.float64)
        dxyz = np.asarray(self.gate_deltas["delta_xyz_mocap"], dtype=np.float64)
        dyaw = float(self.gate_deltas["delta_yaw_rad"])
        c, s = np.cos(dyaw), np.sin(dyaw)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

        # The course may be authored in a different frame; we transform
        # waypoints in MOCAP space.
        new_waypoints = []
        n_moved = 0
        for wp in course.waypoints:
            p_authored = np.asarray(wp.p, dtype=np.float64)
            p_mocap = self.frame_graph.convert(
                Point(p_authored, frame=self.frame_graph.frame(course.frame)),
                to="mocap",
            ).xyz
            inside = bool(((p_mocap >= aabb_min) & (p_mocap <= aabb_max)).all())
            if inside:
                moved_mocap = R @ (p_mocap - anchor) + anchor + dxyz
                moved_authored = self.frame_graph.convert(
                    Point(moved_mocap, frame=self.frame_graph.frame("mocap")),
                    to=course.frame,
                ).xyz
                # Yaw stays in the same coordinate system as the original
                # (course's authored frame). Yaw also rotates by dyaw if
                # the waypoint has an explicit yaw — keeps the
                # gate-passage orientation consistent with the perturbed
                # gate plane.
                new_yaw = wp.yaw if wp.yaw is None else float(wp.yaw + dyaw)
                new_waypoints.append(dataclasses.replace(
                    wp, p=np.asarray(moved_authored, dtype=np.float64),
                    yaw=new_yaw,
                ))
                n_moved += 1
            else:
                new_waypoints.append(wp)
        if n_moved > 0:
            print(f"[coursed_mpc] applied gate_deltas to {n_moved}/"
                  f"{len(course.waypoints)} waypoints inside gate AABB")
        return dataclasses.replace(course, waypoints=tuple(new_waypoints))

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
