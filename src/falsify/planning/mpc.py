"""FiGS-MPC trajectory planner.

Plans a dynamically-feasible trajectory through a :class:`Course` by:

1. Converting the course waypoints (declared in any frame) to NED.
2. Building a min-time-snap reference through them (via
   ``figs.tsplines.min_time_snap.MinTimeSnap``, invoked internally by
   ``VehicleRateMPC``).
3. Closing the loop with ``figs.control.vehicle_rate_mpc.VehicleRateMPC``
   tracking the reference, using a slim ``acados`` IRK integrator for the
   quadrotor rate-input model (``figs.dynamics.quadcopter_rate_model``).

The MPC's OCP and the integrator are both compiled lazily by ``acados``
on first call (~30 s). We isolate the generated C/.so artefacts in a
``tempfile.TemporaryDirectory`` so concurrent planners (e.g. a recovery
MPC built mid-rollout next to a VLA-side MPC) don't fight over the
``./c_generated_code/`` path that ``acados`` defaults to.

Pattern mirrors SousVide's ``FalsificationOrchestrator._rollout_recovery``.
Cost weights, horizon, and rate bounds are reused from that recipe (and
work for the dronevla v7 gate scenes); the drone physical parameters come
from a FiGS-schema frame JSON (default
``configs/frames/figs/carl.json``).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from falsify.geometry import FrameGraph, PointCloud
from falsify.training.trajectory import Trajectory as TrainingTrajectory

from .spline import _to_ned_yaw  # yaw frame conversion (preserve sign on z-flip)
from .waypoints import Course


# ---------------------------------------------------------------------------
# Defaults (SousVide-validated for carl quad)
# ---------------------------------------------------------------------------


_DEFAULT_POLICY_CFG: dict = {
    "plan": {"kT": 10.0, "use_l2_time": False},
    "track": {
        "hz": 10,
        "horizon": 40,
        # State cost diag: [px, py, pz, vx, vy, vz, qx, qy, qz, qw]
        "Qk": [100, 100, 100, 1, 1, 1, 10, 10, 10, 10],
        # Control cost diag: [u_thrust, wx, wy, wz]
        "Rk": [1.0, 0.1, 0.1, 0.01],
        "QN": [100, 100, 100, 1, 1, 1, 10, 10, 10, 10],
        "Ws": [10, 10, 10, 0.1, 0.1, 0.1, 0, 0, 0, 0],
        # Control bounds: [u_thrust_lower, wx, wy, wz] / upper
        "bounds": {"lower": [-1.0, -5.0, -5.0, -5.0],
                   "upper": [0.0, 5.0, 5.0, 5.0]},
    },
}

_DEFAULT_FRAME_PATH = Path("configs/frames/figs/carl.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _yaw_to_quat_xyzw(yaw: float) -> np.ndarray:
    return np.array([0.0, 0.0, np.sin(0.5 * yaw), np.cos(0.5 * yaw)])


def _waypoints_to_ned(course: Course, frame_graph: FrameGraph) -> np.ndarray:
    """Return the course waypoints' positions in NED, shape (N, 3)."""
    src = PointCloud(
        points=course.positions,
        frame=frame_graph.frame(course.frame),
    )
    return frame_graph.convert(src, to="ned").points


def _resolve_frame_cfg(frame_cfg: dict | str | Path | None) -> dict:
    """Load the FiGS-schema drone frame config from disk (or pass through if
    already a dict). Repo-relative paths resolve from the falsify root."""
    if isinstance(frame_cfg, dict):
        return frame_cfg
    path = Path(frame_cfg) if frame_cfg is not None else _DEFAULT_FRAME_PATH
    if not path.is_absolute():
        repo_root = Path(__file__).resolve().parents[3]
        path = repo_root / path
    if not path.is_file():
        raise FileNotFoundError(f"frame_cfg JSON not found: {path}")
    return json.loads(path.read_text())


def _build_keyframes(
    positions_ned: np.ndarray,
    yaws_ned: np.ndarray,
    times_s: np.ndarray,
    names: list[str],
) -> dict:
    """Build a ``MinTimeSnap``-style keyframes dict.

    Endpoints fix position + zero derivative (so the trajectory starts and
    ends at rest). Middle waypoints constrain position only; the snap
    minimization solves for smooth derivatives. Yaw is treated as the 4th
    flat-output channel and follows the same convention.
    """
    n = positions_ned.shape[0]
    if not (yaws_ned.shape[0] == n == times_s.shape[0] == len(names)):
        raise ValueError("positions/yaws/times/names must have matching length")
    keyframes: dict = {}
    for i in range(n):
        p = positions_ned[i]
        y = float(yaws_ned[i])
        is_endpoint = (i == 0) or (i == n - 1)
        if is_endpoint:
            fo = [
                [float(p[0]), 0.0],
                [float(p[1]), 0.0],
                [float(p[2]), 0.0],
                [y, 0.0],
            ]
        else:
            fo = [
                [float(p[0]), None, None, None],
                [float(p[1]), None, None, None],
                [float(p[2]), None, None, None],
                [y, None, None, None],
            ]
        keyframes[names[i]] = {"t": float(times_s[i]), "fo": fo}
    return keyframes


def _initial_state(
    course: Course,
    frame_graph: FrameGraph,
    start_state_ned: Optional[np.ndarray],
    positions_ned_wps: np.ndarray,
    yaws_ned_wps: np.ndarray,
) -> np.ndarray:
    """Resolve x0 (10-vector: pos[3], vel[3], quat_xyzw[4]) for the loop.

    Either honours an explicit ``start_state_ned`` (used by recovery, which
    starts from ``last_safe_state``) or falls back to the course's first
    waypoint at rest with the course-resolved yaw.
    """
    if start_state_ned is not None:
        x = np.asarray(start_state_ned, dtype=np.float64).reshape(-1)
        if x.shape != (10,):
            raise ValueError(
                f"start_state_ned must be a 10-vector (pos, vel, quat_xyzw); got {x.shape}"
            )
        return x
    x = np.zeros(10, dtype=np.float64)
    x[0:3] = positions_ned_wps[0]
    x[3:6] = 0.0
    x[6:10] = _yaw_to_quat_xyzw(float(yaws_ned_wps[0]))
    return x


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plan_mpc(
    course: Course,
    frame_graph: FrameGraph,
    *,
    prompt: str = "",
    start_state_ned: Optional[np.ndarray] = None,
    total_time_s: Optional[float] = None,
    hz: Optional[int] = None,
    policy_cfg: Optional[dict] = None,
    frame_cfg: dict | str | Path | None = None,
    use_rti: bool = True,
) -> TrainingTrajectory:
    """Plan a dynamically-feasible trajectory through ``course``'s waypoints.

    Parameters
    ----------
    course
        Source course (declared in any frame).
    frame_graph
        Active scene FrameGraph. Used to convert waypoint positions
        and yaws to NED.
    prompt
        Forwarded to the resulting Trajectory's ``prompt`` field.
    start_state_ned
        Optional 10-vector ``[px, py, pz, vx, vy, vz, qx, qy, qz, qw]`` in
        NED. When set, the MPC starts from this state instead of the
        course's first waypoint at rest — the recovery use case.
    total_time_s, hz
        Override course defaults.
    policy_cfg
        Override the inline ``VehicleRateMPC`` policy dict. Default is
        ``_DEFAULT_POLICY_CFG`` (carl-quad tuning from SousVide).
    frame_cfg
        FiGS-schema drone frame. Either a dict, a path to JSON, or None
        (loads ``configs/frames/figs/carl.json``).
    use_rti
        Build the OCP solver in SQP-RTI mode (one SQP iteration per tick,
        ~5–10x faster than full SQP). Default True for closed-loop tracking
        of a smooth min-time-snap reference; flip to False if the recovery
        ever needs full SQP convergence per tick.

    Returns
    -------
    Trajectory
        NED-frame, ``hz``-sampled trajectory carrying the MPC's tracked
        state evolution. ``source = "mpc:<course.name>"``.
    """
    # Heavy imports stay inside the function — pulling acados/FiGS at module
    # import time slows down anything that imports `falsify.planning`.
    from acados_template import AcadosSim, AcadosSimSolver
    from figs.control.vehicle_rate_mpc import VehicleRateMPC
    from figs.dynamics import quadcopter_rate_model as qrm

    # 1) Resolve course timing + yaw.
    waypoint_ts = course.resolved_times().astype(np.float64)
    waypoint_yaws_src = course.resolved_yaws()
    yaws_ned_wps = _to_ned_yaw(waypoint_yaws_src, course.frame, frame_graph)

    positions_ned_wps = _waypoints_to_ned(course, frame_graph)
    names = [wp.name for wp in course.waypoints]

    if total_time_s is None:
        total_time_s = float(course.total_time_s)
    if hz is None:
        hz = int(course.fps)

    # 2) Build the inline FiGS configs.
    course_config = {
        "waypoints": {
            "Nco": 6,
            "keyframes": _build_keyframes(
                positions_ned_wps, yaws_ned_wps, waypoint_ts, names,
            ),
        },
        "forces": None,
    }
    pcfg = policy_cfg or _DEFAULT_POLICY_CFG
    if "track" in pcfg and "hz" in pcfg["track"]:
        # Honour the policy's hz over the function arg when both diverge —
        # the MPC's OCP horizon assumes that rate.
        hz = int(pcfg["track"]["hz"])
    fcfg = _resolve_frame_cfg(frame_cfg)

    # 3) Build the MPC + the integrator inside a fresh temp dir so the
    # acados-generated .c / .so / .json files stay isolated.
    saved_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="falsify_mpc_") as tmpd:
        os.chdir(tmpd)
        try:
            mpc = VehicleRateMPC(
                policy=pcfg, course=course_config, frame=fcfg,
                use_RTI=bool(use_rti),
            )

            sim_json = "falsify_mpc_integrator.json"
            sim = AcadosSim()
            sim.model = qrm.export_model()
            sim.parameter_values = np.zeros(sim.model.p.shape)
            sim.solver_options.T = 1.0 / hz
            sim.solver_options.integrator_type = "IRK"
            integrator = AcadosSimSolver(sim, json_file=sim_json, verbose=False)

            # 4) Closed-loop tick.
            mass = float(fcfg["mass"])
            kt = float(fcfg["motor_thrust_coeff"])
            g = 9.81
            n_rotors = int(fcfg["number_of_rotors"])

            x = _initial_state(course, frame_graph, start_state_ned,
                               positions_ned_wps, yaws_ned_wps)
            # Trim hover thrust as the initial command. SousVide uses
            # u0 = -(m*g)/(Nrtr*kt); the negative is because FiGS' convention
            # has u_thrust ≤ 0 (lower bound -1, upper 0 in policy bounds).
            u = np.array([-(mass * g) / (n_rotors * kt), 0.0, 0.0, 0.0])
            # Model parameters: [mass, kt, fx, fy, fz] (zero external forces).
            p_dyn = np.array([mass, kt, 0.0, 0.0, 0.0])

            n_steps = max(1, int(round(total_time_s * hz)))
            times = np.zeros(n_steps + 1, dtype=np.float64)
            states = np.zeros((n_steps + 1, 10), dtype=np.float64)
            states[0] = x
            times[0] = 0.0
            # Workspace divergence guard. The acados OCP solver can silently
            # fail and FiGS' VehicleRateMPC.control falls back to the last
            # (possibly garbage) u — that drives the integrator out to
            # hundreds of metres. We bound on state and solver status so a
            # bad solve truncates the trajectory instead of corrupting it.
            pos_abort_m = 25.0   # ~5x scene radius; well outside any working volume
            vel_abort_mps = 25.0
            n_solver_fail = 0
            truncated_at: Optional[int] = None
            abort_reason: Optional[str] = None
            for k in range(n_steps):
                t_cur = k / hz
                # Dummy rgb/dpt/fcr (MPC doesn't read them for this model).
                u, _ = mpc.control(t_cur, x, u, None, None,
                                   np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
                # acados status: 0=success; non-zero = QP/NLP failure.
                solver_status = int(getattr(mpc.solver, "status", 0) or 0)
                if solver_status != 0:
                    n_solver_fail += 1
                if not np.all(np.isfinite(u)):
                    abort_reason = f"NaN/inf in control at step {k}, t={t_cur:.2f}s"
                    truncated_at = k
                    break
                x_new = integrator.simulate(x=x, u=u, p=p_dyn)
                if not np.all(np.isfinite(x_new)):
                    abort_reason = f"NaN/inf in integrated state at step {k+1}, t={(k+1)/hz:.2f}s"
                    truncated_at = k
                    break
                pos_norm = float(np.linalg.norm(x_new[0:3]))
                vel_norm = float(np.linalg.norm(x_new[3:6]))
                if pos_norm > pos_abort_m or vel_norm > vel_abort_mps:
                    abort_reason = (
                        f"state diverged at step {k+1}, t={(k+1)/hz:.2f}s: "
                        f"|pos|={pos_norm:.1f}m (>{pos_abort_m}) "
                        f"|vel|={vel_norm:.1f}m/s (>{vel_abort_mps})"
                    )
                    truncated_at = k
                    break
                x = x_new
                states[k + 1] = x
                times[k + 1] = (k + 1) / hz
            else:
                truncated_at = n_steps

            if abort_reason is not None:
                print(f"[plan_mpc] ABORT: {abort_reason}; "
                      f"solver_failures={n_solver_fail}/{truncated_at+1}; "
                      f"returning truncated trajectory of length {truncated_at+1}",
                      flush=True)
            elif n_solver_fail > 0:
                print(f"[plan_mpc] {n_solver_fail}/{n_steps} OCP solves failed "
                      f"(status != 0); trajectory completed but may be degraded",
                      flush=True)
            # Truncate buffers to the last valid sample.
            states = states[: truncated_at + 1]
            times = times[: truncated_at + 1]
        finally:
            os.chdir(saved_cwd)

    # 5) Wrap as a falsify Trajectory.
    positions_ned = states[:, 0:3]
    velocities_ned = states[:, 3:6]
    quaternions_xyzw = states[:, 6:10]
    return TrainingTrajectory(
        times=times,
        positions_ned=positions_ned,
        velocities_ned=velocities_ned,
        quaternions_xyzw=quaternions_xyzw,
        prompt=prompt,
        source=f"mpc:{course.name}",
    )
