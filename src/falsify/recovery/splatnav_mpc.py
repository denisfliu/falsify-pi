"""SplatNav A* + spline → VehicleRateMPC recovery planner.

Replaces the legacy ``CoursedMpcPlanner`` for recovery collection. The
recovery strategy is now collision-aware by construction:

  1. Load the course YAML (e.g. ``configs/courses/through_left_gate.yaml``)
     and (when ``gate_deltas`` is set) warp its in-gate-AABB waypoints
     so the rest of the pipeline targets the *perturbed* gate, not the
     nominal one.
  2. Use the warped waypoints as anchors. Between every consecutive pair
     (start_safe_state → wp_1 → … → wp_N → goal), call SplatPlan's
     A\* + B-spline planner to produce a collision-free segment.
  3. Concatenate the segments, subsample to a Course-shaped sequence,
     hand to ``plan_mpc`` so the FiGS ``VehicleRateMPC`` makes the
     reference dynamically feasible.
  4. Post-MPC, run ``check_collision_posthoc`` against the scene's gate
     point cloud as a safety net — log a warning when the MPC's
     tracking error pushed the body into a post.

Why the share-the-pipeline dance?
---------------------------------
The renderer keeps the live gsplat ``Pipeline`` (means/quats mutated
in place by every per-trial ``GateRigidPerturbation``). To plan
collision-free recoveries through the *actually perturbed* gate, the
A\* layer must see those mutations. SplatPlan caches a voxel grid from
``gsplat.means``/``gsplat.covs`` snapshots at construction; we
therefore (a) build a ``_SharedGSplatLoader`` that points at the same
``Pipeline`` instead of double-loading, and (b) call ``refresh()``
before each ``plan()`` to re-snapshot tensors and rebuild the voxel
grid. The cost is a one-time GSplatLoader snapshot per trial plus a
voxel-grid rebuild — both fast relative to the eliminated 30-s reload.

Same ``plan(start, goal) → RecoveryResult`` interface as
``CoursedMpcPlanner`` so ``scripts/recovery/collect_recovery_trajectories.py``
is a one-line swap.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from falsify.geometry import FrameGraph, Point, Trajectory, assert_frame
from falsify.planning import Course, Waypoint, load_course, plan_mpc

from .coursed_mpc import (
    apply_gate_deltas_to_course,
    trim_course_to_target,
)
from .planner import RecoveryResult


# ---------------------------------------------------------------------------
# Shared-pipeline GSplatLoader + SplatPlan wrapper
# ---------------------------------------------------------------------------

def _load_standalone_pipeline(gsplat_config_path, data_cwd=None):
    """Load a fresh nerfstudio pipeline from a config.yml — same code path
    ``GSplatRenderer`` uses, but isolated so the planner can own the
    pipeline independently when the renderer has been freed.

    nerfstudio's dataparser resolves its ``data:`` field as a relative
    path against cwd; the gate-scene configs store it as a bare
    ``mocap_processed`` (sibling of the gsplat config dir), so we mirror
    GSplatRenderer's ``_chdir(data_cwd)`` context wrap. Caller should pass
    the scene's ``gsplat_data_cwd`` (resolved against the scene yaml dir)
    when set, otherwise the default for our gate scenes is the parent
    dir two levels above the config.yml.
    """
    import os
    from contextlib import contextmanager
    from figs.render.gsplat import GSplat as _FiGSGSplat  # type: ignore
    gsplat_config_path = Path(gsplat_config_path)
    if data_cwd is not None:
        data_cwd = Path(data_cwd)

    @contextmanager
    def _chdir(path):
        if path is None:
            yield; return
        prev = os.getcwd(); os.chdir(path)
        try: yield
        finally: os.chdir(prev)

    with _chdir(data_cwd):
        impl = _FiGSGSplat(gsplat_config_path)
    return impl.pipeline


def _build_shared_gsplat_loader(
    pipeline, device,
    opacity_threshold=0.1,
    bounds_lower_ns=None,
    bounds_upper_ns=None,
):
    """Return a ``GSplatLoader``-shaped object whose snapshot tensors are
    taken from ``pipeline.model`` (no second nerfstudio load). The
    ``.refresh()`` method re-clones the snapshots so per-trial edits to
    ``pipeline.model.means`` / ``.quats`` propagate.

    ``opacity_threshold`` filters out the long tail of low-alpha "halo"
    Gaussians the splat trainer leaves around dense regions. SplatPlan's
    voxel-grid build subdivides per-Gaussian bounding boxes until they
    fit a cell — the residual count doubles each iteration, so a 5%
    halo at 150k Gaussians can blow up to a 4 GB ``torch.cat`` (the
    failure mode that hit us on the gate scenes). Keeping only
    Gaussians with sigmoid(opacities) > 0.1 typically drops total
    Gaussian count by 30-60% while leaving the collision-significant
    structure (posts, frame, table) untouched.
    """
    import torch  # type: ignore
    from ellipsoids.covariance_utils import compute_cov  # type: ignore
    from ns_utils.nerfstudio_utils import SH2RGB         # type: ignore
    from splat.splat_utils import GSplatLoader           # type: ignore

    class _SharedGSplatLoader(GSplatLoader):
        def __init__(self, _pipeline, _device, _opacity_threshold,
                     _bounds_lower_ns, _bounds_upper_ns):
            self.device = _device
            self._pipeline = _pipeline
            self._opacity_threshold = float(_opacity_threshold)
            self._bounds_lower_ns = _bounds_lower_ns   # torch tensor or None
            self._bounds_upper_ns = _bounds_upper_ns
            self.refresh()

        def refresh(self) -> None:
            with torch.no_grad():
                m = self._pipeline.model
                opac = torch.sigmoid(m.opacities.detach())
                if opac.ndim == 2 and opac.shape[1] == 1:
                    opac_flat = opac[:, 0]
                else:
                    opac_flat = opac.reshape(-1)
                opacity_mask = opac_flat > self._opacity_threshold
                # Spatial mask: keep Gaussians inside the env_bounds (NS).
                # SplatPlan already discards out-of-bounds bbs after
                # computing them for ALL Gaussians; pre-filtering up here
                # avoids the explosion in `create_navigable_grid`'s
                # subdivision loop at 6M-Gaussian scenes.
                if self._bounds_lower_ns is not None and self._bounds_upper_ns is not None:
                    means_all = m.means.detach()
                    in_bounds = (
                        (means_all >= self._bounds_lower_ns).all(dim=1)
                        & (means_all <= self._bounds_upper_ns).all(dim=1)
                    )
                    mask = opacity_mask & in_bounds
                else:
                    mask = opacity_mask
                n_keep = int(mask.sum().item())
                n_total = int(mask.numel())
                self.means     = m.means.detach()[mask].clone()
                self.rots      = m.quats.detach()[mask].clone()
                self.scales    = torch.exp(m.scales.detach()[mask].clone())
                self.covs_inv  = compute_cov(self.rots, 1.0 / self.scales)
                self.covs      = compute_cov(self.rots, self.scales)
                self.colors    = SH2RGB(m.features_dc.detach()[mask].clone())
                self.opacities = opac[mask].clone()
                print(f"[splatnav_mpc] refresh: kept {n_keep:,}/{n_total:,} "
                      f"Gaussians (opacity > {self._opacity_threshold}"
                      + (" + inside NS env_bounds"
                         if self._bounds_lower_ns is not None else "")
                      + ")")

    return _SharedGSplatLoader(
        pipeline, device, opacity_threshold,
        bounds_lower_ns, bounds_upper_ns,
    )


def _build_splatplan(gsplat_loader, robot_config, env_config, device):
    """Construct a fresh ``SplatPlan`` against the current ``gsplat_loader``
    snapshots. Voxel grid is built once inside ``SplatPlan.__init__``;
    re-call to rebuild after a refresh."""
    from splatplan.splatplan import SplatPlan          # type: ignore
    from splatplan.spline_utils import SplinePlanner   # type: ignore

    spline_planner = SplinePlanner(spline_deg=6, N_sec=10, device=device)
    return SplatPlan(
        gsplat_loader,
        robot_config,
        {**env_config, "resolution": env_config["resolution"]},
        spline_planner,
        device,
    )


# ---------------------------------------------------------------------------
# Public planner
# ---------------------------------------------------------------------------


@dataclass
class _SplatNavMpcConfig:
    """Lightweight value-class for ``SplatNavMpcPlanner`` knobs."""
    drone_clearance_m: float = 0.175
    vmax: float = 2.0
    amax: float = 3.0
    # 100³ peaks at ~4.5 GB during the spline polytope solve — fights the
    # v9 VLA model already resident on the GPU. 50³ keeps the grid coarse
    # enough to plan our 80 cm gate aperture (cell ≈ 14 cm < drone width)
    # while leaving headroom for SplatPlan's spline solver. Bump back to
    # 100 when planning very tight passages or running on a dedicated GPU.
    voxel_resolution: int = 50
    bounds_lower_mocap: tuple[float, float, float] = (-3.0, -3.0, 0.0)
    bounds_upper_mocap: tuple[float, float, float] = (4.0, 3.0, 3.0)
    # MPC reference shape — how many Course-shaped waypoints we hand to
    # ``plan_mpc`` after concatenating the SplatPlan spline segments.
    n_reference_waypoints: int = 20
    # Total trajectory time for the synthesized Course. Matches the
    # default a single-gate course YAML authors (e.g. through_left_gate
    # is ~5 s start→hover).
    total_time_s: float = 5.0


class SplatNavMpcPlanner:
    """Two-stage recovery: SplatPlan A\* + spline → VehicleRateMPC tracking."""

    def __init__(
        self,
        course_path: str | Path,
        frame_graph: FrameGraph,
        *,
        pipeline: Any = None,              # nerfstudio Pipeline (shared mode)
        gsplat_config_path: Optional[str | Path] = None,  # standalone mode
        gsplat_data_cwd: Optional[str | Path] = None,     # standalone-mode chdir
        device: Any = None,
        cfg: Optional[_SplatNavMpcConfig] = None,
        prompt: str = "",
        gate_deltas: Optional[dict] = None,
        scene_cfg: Optional[dict] = None,
        mpc_frame_cfg: dict | str | Path | None = None,
    ) -> None:
        """Either ``pipeline=`` (share with the renderer) or
        ``gsplat_config_path=`` (own load) must be set. In standalone mode
        the planner owns the pipeline so it can ``snapshot_baseline`` and
        ``restore_baseline`` like ``GSplatRenderer`` does — used by the
        two-phase recovery collector where the renderer is freed before
        any planning happens.
        """
        import torch  # type: ignore

        if pipeline is None and gsplat_config_path is None:
            raise ValueError(
                "SplatNavMpcPlanner needs one of pipeline= (share with renderer) "
                "or gsplat_config_path= (load own gsplat)"
            )

        self.course_path = Path(course_path)
        self.frame_graph = frame_graph
        self.prompt = prompt
        self.gate_deltas = gate_deltas
        self.scene_cfg = scene_cfg
        self.mpc_frame_cfg = mpc_frame_cfg
        self.cfg = cfg or _SplatNavMpcConfig()
        self._course: Optional[Course] = None

        # Set up the SplatPlan stack once. Per trial we'll refresh tensors
        # + rebuild the voxel grid (cheap).
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        if pipeline is None:
            pipeline = _load_standalone_pipeline(
                gsplat_config_path, data_cwd=gsplat_data_cwd,
            )
        self._pipeline = pipeline
        # Baseline snapshot so ``apply_scene_edits`` is a "reset → apply"
        # idempotent op — mirrors the GSplatRenderer's pattern. Lazy: only
        # snapshotted the first time an edit is applied.
        self._baseline_means = None
        self._baseline_quats = None

        # MOCAP → NS linear scale. We use a unit basis vector; the
        # FrameGraph carries the perm5/sim3 transform so this captures
        # the per-scene scale automatically.
        mocap_frame = frame_graph.frame("mocap")
        ns_frame = frame_graph.frame("ns")
        p0_ns = frame_graph.convert(Point(np.zeros(3), frame=mocap_frame), to="ns").xyz
        p1_ns = frame_graph.convert(Point(np.array([1.0, 0.0, 0.0]), frame=mocap_frame), to="ns").xyz
        self._scale_mocap_to_ns = float(np.linalg.norm(p1_ns - p0_ns))
        self._radius_ns = self.cfg.drone_clearance_m * self._scale_mocap_to_ns

        # NS-frame bounds. Convert MOCAP bounds via FrameGraph + min/max
        # over the corners to handle frame flips (perm5 negates y/z).
        lo_mocap = np.asarray(self.cfg.bounds_lower_mocap, dtype=np.float64)
        hi_mocap = np.asarray(self.cfg.bounds_upper_mocap, dtype=np.float64)
        lo_ns = frame_graph.convert(Point(lo_mocap, frame=mocap_frame), to="ns").xyz
        hi_ns = frame_graph.convert(Point(hi_mocap, frame=mocap_frame), to="ns").xyz
        self._lo_ns = torch.tensor(np.minimum(lo_ns, hi_ns), device=device, dtype=torch.float32)
        self._hi_ns = torch.tensor(np.maximum(lo_ns, hi_ns), device=device, dtype=torch.float32)
        # Pre-filter the gsplat loader by env bounds. 6M-Gaussian scenes
        # otherwise overwhelm SplatPlan's bounding-box subdivision loop;
        # culling to those inside env_bounds drops typical gate scenes
        # to <200k Gaussians (the relevant collision structure).
        self._gsplat = _build_shared_gsplat_loader(
            pipeline, device,
            bounds_lower_ns=self._lo_ns, bounds_upper_ns=self._hi_ns,
        )

        self._robot_config = {
            "radius": self._radius_ns,
            "vmax": self.cfg.vmax,
            "amax": self.cfg.amax,
        }
        self._env_config = {
            "lower_bound": self._lo_ns,
            "upper_bound": self._hi_ns,
            "resolution": self.cfg.voxel_resolution,
        }
        # First-time build. ``plan()`` rebuilds per-trial.
        self._splatplan = _build_splatplan(
            self._gsplat, self._robot_config, self._env_config, device,
        )
        print(f"[splatnav_mpc] init: clearance={self.cfg.drone_clearance_m:.3f} m "
              f"MOCAP → {self._radius_ns:.4f} NS (scale={self._scale_mocap_to_ns:.4f})")

    # ---- public API ---------------------------------------------------

    def plan(
        self,
        start: Point,
        goal: Point,
        *,
        target_waypoint: Optional[str] = None,
        failure_phase: Optional[str] = None,  # back-compat, ignored
    ) -> RecoveryResult:
        """Plan a recovery from ``start`` to ``goal``.

        ``target_waypoint`` is the FIRST course waypoint the recovery
        should aim for after ``start``. Caller picks this via
        ``Course.target_waypoint(post_phase, seed_kind)`` based on the
        post-trim phase and seed kind (in_gate vs pre_gate). If
        ``target_waypoint`` is None, the planner uses the course's full
        waypoint list (start-replacement behaviour, retained as a fallback
        for legacy callers that don't pass phase info).
        """
        assert_frame(start, "ned")
        assert_frame(goal, "ned")

        # Per-trial: re-snapshot tensors from the (now-edited) pipeline
        # and rebuild SplatPlan's voxel grid against the new means. The
        # previous voxel grid + SplatPlan must be released *before* the
        # new one allocates — at ~100k Gaussians × resolution³ cells the
        # old grid keeps several GB of CUDA memory alive otherwise and
        # the second build hits OOM. Drop refs, force GC, empty CUDA
        # cache, *then* allocate.
        import gc
        import torch  # type: ignore
        self._splatplan = None
        gc.collect()
        if torch.cuda.is_available():
            # Two passes — first empty_cache reaps the dropped SplatPlan
            # arena; second runs after refresh to defragment between the
            # snapshot clone and the SplatPlan build. We also call
            # `synchronize` so any queued rasterizer ops from the
            # preceding VLA rollout actually release their workspaces
            # before we start grabbing memory.
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        self._gsplat.refresh()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._splatplan = _build_splatplan(
            self._gsplat, self._robot_config, self._env_config, self._device,
        )

        course = self._load_course()
        if target_waypoint is not None:
            # Phase-driven path: trim the course to the suffix starting at
            # ``target_waypoint``. The seed is prepended as a separate
            # anchor in the planner — the course is left otherwise pristine
            # (no W_0 overwrite). This is the new flow; legacy callers that
            # don't supply ``target_waypoint`` fall through to the
            # waypoint-overwrite path below.
            course = trim_course_to_target(course, target_waypoint)
            course_frame = self.frame_graph.frame(course.frame)
            anchors_ned = [np.asarray(start.xyz, dtype=np.float64)]
            for wp in course.waypoints:
                p_ned = self.frame_graph.convert(
                    Point(np.asarray(wp.p, dtype=np.float64), frame=course_frame),
                    to="ned",
                ).xyz
                anchors_ned.append(np.asarray(p_ned, dtype=np.float64))
        else:
            # Back-compat fallback path: overwrite the course's first
            # waypoint with the seed and walk the rest.
            from .coursed_mpc import replace_start_waypoint
            course = replace_start_waypoint(course, start, self.frame_graph)
            course_frame = self.frame_graph.frame(course.frame)
            anchors_ned = []
            for wp in course.waypoints:
                p_ned = self.frame_graph.convert(
                    Point(np.asarray(wp.p, dtype=np.float64), frame=course_frame),
                    to="ned",
                ).xyz
                anchors_ned.append(np.asarray(p_ned, dtype=np.float64))
        # Some courses end at the hover/goal already; only append goal
        # if it isn't a near-duplicate of the last anchor.
        if np.linalg.norm(anchors_ned[-1] - goal.xyz) > 1e-3:
            anchors_ned.append(np.asarray(goal.xyz, dtype=np.float64))

        # Plan SplatPlan segments between consecutive anchors.
        dense_path_ned = self._chain_segments(anchors_ned)

        # Down-sample to a Course-shape that ``plan_mpc`` can consume.
        ref_course = self._build_reference_course(
            dense_path_ned, course=course, start=start,
        )

        # MPC tracking. Reuse CoursedMpcPlanner's "honor authored time"
        # knob — no time re-optimisation, since we set total_time_s on
        # the synthesized course explicitly.
        from falsify.planning.mpc import _DEFAULT_POLICY_CFG
        policy_cfg = {
            "plan":  {"kT": None, "use_l2_time": False},
            "track": _DEFAULT_POLICY_CFG["track"],
        }
        start_state_ned = self._initial_state_from(course, start)
        traj_training = plan_mpc(
            ref_course, self.frame_graph,
            prompt=self.prompt,
            start_state_ned=start_state_ned,
            frame_cfg=self.mpc_frame_cfg,
            policy_cfg=policy_cfg,
        )

        ned_frame = self.frame_graph.frame("ned")
        traj_ned = Trajectory(
            times=np.asarray(traj_training.times, dtype=np.float64),
            positions=np.asarray(traj_training.positions_ned, dtype=np.float64),
            frame=ned_frame,
            quaternions=np.asarray(traj_training.quaternions_xyzw, dtype=np.float64),
        )
        assert_frame(traj_ned, "ned")

        info = {
            "planner": "splatnav_mpc",
            "n_anchors": len(anchors_ned),
            "n_dense_path_points": int(dense_path_ned.shape[0]),
            "n_reference_waypoints": len(ref_course.waypoints),
            "radius_ns": float(self._radius_ns),
            "drone_clearance_m": float(self.cfg.drone_clearance_m),
        }

        # Safety-net posthoc collision sweep (WARN only).
        collision = self._check_recovery_collision(traj_ned)
        if collision is not None:
            info["collision_sweep"] = {
                "n_collision_steps": collision.n_collision_steps,
                "first_collision_step": collision.first_collision_step,
                "last_collision_step":  collision.last_collision_step,
                "max_points_hit_in_step": collision.max_points_hit_in_step,
                "n_total_points_hit":   collision.n_total_points_hit,
            }
            if collision.n_collision_steps > 0:
                warnings.warn(
                    f"[splatnav_mpc] recovery trajectory clips gate Gaussians: "
                    f"n_collision_steps={collision.n_collision_steps} "
                    f"(first={collision.first_collision_step}, "
                    f"last={collision.last_collision_step}, "
                    f"max_pts/step={collision.max_points_hit_in_step}). "
                    f"Recovery NPZ saved with collision metadata; "
                    f"filter downstream if you want to exclude.",
                    stacklevel=2,
                )

        return RecoveryResult(trajectory=traj_ned, info=info)

    # ---- per-trial gsplat mutation (two-phase mode) ------------------

    def snapshot_baseline(self) -> None:
        """Clone the current ``pipeline.model.means`` / ``.quats`` once so
        ``restore_baseline`` can revert per-trial edits. Idempotent."""
        if self._baseline_means is not None:
            return
        import torch  # type: ignore
        with torch.no_grad():
            self._baseline_means = self._pipeline.model.means.detach().clone()
            self._baseline_quats = self._pipeline.model.quats.detach().clone()

    def restore_baseline(self) -> None:
        """Copy the baseline back into the live pipeline. No-op if no
        baseline has been taken yet (the pipeline IS the baseline)."""
        if self._baseline_means is None:
            return
        import torch  # type: ignore
        with torch.no_grad():
            self._pipeline.model.means.data.copy_(self._baseline_means)
            self._pipeline.model.quats.data.copy_(self._baseline_quats)

    def apply_scene_edits(self, edits) -> int:
        """Reset to baseline, then apply ``edits`` to the live pipeline.
        Mirrors ``GSplatRenderer.apply_dynamic_edits`` so per-trial
        ``GateRigidPerturbation`` edits don't compound. Returns the
        number of Gaussians modified.
        """
        from falsify.sim.scene_edits import apply_edits_to_pipeline
        self.snapshot_baseline()
        self.restore_baseline()
        return apply_edits_to_pipeline(self._pipeline, edits, self.frame_graph)

    # Alias matching GSplatRenderer's API — lets `GateRigidPerturbation.apply`
    # treat the planner as a renderer-shaped target.
    apply_dynamic_edits = apply_scene_edits

    # ---- internals ----------------------------------------------------

    def _load_course(self) -> Course:
        if self._course is None:
            course = load_course(self.course_path)
            if self.gate_deltas is not None and self.scene_cfg is not None:
                course = apply_gate_deltas_to_course(
                    course, scene_cfg=self.scene_cfg,
                    gate_deltas=self.gate_deltas, frame_graph=self.frame_graph,
                )
            self._course = course
        return self._course

    def _chain_segments(self, anchors_ned: list[np.ndarray]) -> np.ndarray:
        """Call SplatPlan on each consecutive anchor pair in NED, convert
        the NS-frame outputs back to NED, and concatenate."""
        import torch  # type: ignore

        ns_frame = self.frame_graph.frame("ns")
        ned_frame = self.frame_graph.frame("ned")
        all_points: list[np.ndarray] = []
        for i in range(len(anchors_ned) - 1):
            a_ned = anchors_ned[i]
            b_ned = anchors_ned[i + 1]
            a_ns = self.frame_graph.convert(Point(a_ned, frame=ned_frame), to="ns").xyz
            b_ns = self.frame_graph.convert(Point(b_ned, frame=ned_frame), to="ns").xyz
            x0 = torch.tensor(a_ns, dtype=torch.float32, device=self._device)
            xf = torch.tensor(b_ns, dtype=torch.float32, device=self._device)
            out = self._splatplan.generate_path(x0, xf)
            traj = out["traj"] if isinstance(out, dict) else out
            seg_ns = np.asarray(traj)[:, :3]
            # Drop the first sample of every segment after the first to
            # avoid the start ≡ previous-end duplicate.
            if i > 0 and seg_ns.shape[0] > 0:
                seg_ns = seg_ns[1:]
            for p_ns in seg_ns:
                p_ned = self.frame_graph.convert(
                    Point(np.asarray(p_ns, dtype=np.float64), frame=ns_frame),
                    to="ned",
                ).xyz
                all_points.append(np.asarray(p_ned, dtype=np.float64))
        return np.stack(all_points, axis=0) if all_points else np.empty((0, 3))

    def _build_reference_course(
        self,
        dense_path_ned: np.ndarray,
        *,
        course: Course,
        start: Point,
    ) -> Course:
        """Down-sample the dense NED path to ``cfg.n_reference_waypoints``
        evenly-spaced points; build a Course in the SAME authored frame
        as the input course so per-waypoint yaw handling stays
        consistent."""
        n_in = int(dense_path_ned.shape[0])
        n_out = max(2, int(self.cfg.n_reference_waypoints))
        if n_in == 0:
            raise RuntimeError("SplatNav returned an empty dense path")
        if n_in <= n_out:
            sample_idx = np.arange(n_in)
        else:
            sample_idx = np.linspace(0, n_in - 1, n_out).round().astype(int)

        ned_frame = self.frame_graph.frame("ned")
        course_frame_name = course.frame
        course_frame = self.frame_graph.frame(course_frame_name)

        # Carry the original course's start-yaw into the synthesized
        # course's first waypoint; intermediate waypoints inherit no
        # explicit yaw (let plan_mpc interpolate / let the optimiser pick).
        first_yaw = course.waypoints[0].yaw
        last_yaw  = course.waypoints[-1].yaw

        t_total = float(self.cfg.total_time_s)
        times = np.linspace(0.0, t_total, len(sample_idx))

        waypoints: list[Waypoint] = []
        for j, (idx, t) in enumerate(zip(sample_idx, times)):
            p_ned = dense_path_ned[idx]
            p_course = self.frame_graph.convert(
                Point(np.asarray(p_ned, dtype=np.float64), frame=ned_frame),
                to=course_frame_name,
            ).xyz
            if j == 0:
                yaw = first_yaw
                name = "start"
            elif j == len(sample_idx) - 1:
                yaw = last_yaw
                name = "goal"
            else:
                yaw = None
                name = f"wp_{j:02d}"
            waypoints.append(Waypoint(
                name=name,
                p=np.asarray(p_course, dtype=np.float64),
                yaw=yaw,
                t=float(t),
            ))
        return Course(
            frame=course_frame_name,
            waypoints=tuple(waypoints),
            total_time_s=t_total,
        )

    def _initial_state_from(self, course: Course, start_ned: Point) -> np.ndarray:
        """[px, py, pz, vx, vy, vz, qx, qy, qz, qw] in NED — vel zero, yaw
        from the course's first waypoint."""
        from falsify.planning.spline import _to_ned_yaw, _yaw_to_quat_xyzw
        yaw_src = course.resolved_yaws()[0]
        yaw_ned = float(_to_ned_yaw(np.array([yaw_src]), course.frame, self.frame_graph)[0])
        x0 = np.zeros(10, dtype=np.float64)
        x0[0:3] = start_ned.xyz
        x0[6:10] = _yaw_to_quat_xyzw(yaw_ned)
        return x0

    def _check_recovery_collision(self, traj_ned: Trajectory):
        """Run ``check_collision_posthoc`` on the planned trajectory against
        the scene's gate Gaussians (perturbation-aware via ``gate_deltas``).
        Returns ``None`` if no scene_cfg / no gate clouds available;
        otherwise a ``CollisionSweepResult``."""
        if self.scene_cfg is None:
            return None
        from falsify.safety.criteria import DroneBody
        from falsify.safety.posthoc import (
            apply_gate_deltas_to_cloud,
            check_collision_posthoc,
        )

        # Locate the gate PLY via scene_objects.
        gate_obj = next(
            (o for o in (self.scene_cfg.get("scene_objects") or [])
             if o.get("name") == "gate"),
            None,
        )
        if gate_obj is None:
            return None
        from falsify.visualization.pointcloud import read_ply
        ply_path = (self.course_path.parent / gate_obj["ply"]).resolve()
        if not ply_path.is_file():
            # Course path may not be in configs/scenes; try repo-relative.
            from . import coursed_mpc  # noqa: F401 — for module path
            repo_root = Path(__file__).resolve().parents[3]
            ply_path = (repo_root / gate_obj["ply"].lstrip("./")).resolve()
            if not ply_path.is_file():
                return None
        cloud = read_ply(ply_path, frame=gate_obj.get("frame", "mocap"))
        pts_mocap = np.asarray(cloud.points, dtype=np.float64)
        if self.gate_deltas is not None:
            pts_mocap = apply_gate_deltas_to_cloud(pts_mocap, self.gate_deltas)

        # Convert traj to MOCAP for the OBB sweep.
        ned_frame = self.frame_graph.frame("ned")
        pos_mocap = np.stack([
            self.frame_graph.convert(Point(p, frame=ned_frame), to="mocap").xyz
            for p in traj_ned.positions
        ])
        # Drone body from safety YAML if we can find it; else default.
        half_extents = np.array([0.175, 0.175, 0.075])
        safety = (self.scene_cfg.get("safety") or {})
        if isinstance(safety, dict):
            he = (safety.get("drone_body") or {}).get("half_extents")
            if he is not None:
                half_extents = np.asarray(he, dtype=np.float64)
        drone_body = DroneBody(half_extents=half_extents)
        return check_collision_posthoc(
            pos_mocap, traj_ned.quaternions,
            gate_cloud_mocap=pts_mocap, drone_body=drone_body,
        )
