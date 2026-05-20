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


def build_initial_state(
    scene_cfg: dict,
    frame_graph: FrameGraph,
    *,
    rng: "np.random.Generator | None" = None,
) -> DroneState:
    """Translate the scene's MOCAP start position into a NED `DroneState`.

    If ``scene_cfg["start_randomization"]`` is present and an ``rng`` is
    supplied, the start is drawn uniformly inside the axis-aligned
    half-widths box centred at ``start_position_mocap``. With ``rng=None``
    the nominal start is used verbatim — deterministic behaviour for
    legacy callers (tests, mock smoke runs).

    Schema::

        start_position_mocap: [x, y, z]
        start_randomization:
          half_widths_mocap: [hx, hy, hz]    # uniform U(-h_i, +h_i) per axis
    """
    nominal = np.asarray(scene_cfg["start_position_mocap"], dtype=np.float64)
    rand_cfg = scene_cfg.get("start_randomization") or {}
    half = rand_cfg.get("half_widths_mocap")
    if rng is not None and half is not None:
        half_arr = np.asarray(half, dtype=np.float64)
        if half_arr.shape != (3,):
            raise ValueError(
                f"start_randomization.half_widths_mocap must be (3,), got {half_arr.shape}"
            )
        offset = rng.uniform(-half_arr, +half_arr)
        start_mocap = nominal + offset
    else:
        start_mocap = nominal
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


def _extract_gate_deltas(suite) -> Optional[dict]:
    """Return ``{"delta_xyz": [...], "delta_yaw_rad": ...}`` for the active
    ``GateRigidPerturbation`` (if any), or None.

    Used by the safety layer to keep aperture / collision targets in
    sync with the moved gate Gaussians. Multiple gate perturbations on
    one episode are not yet supported — we take the first.
    """
    if suite is None:
        return None
    for pert in suite.environment_perts:
        delta_xyz = getattr(pert, "_delta_xyz", None)
        delta_yaw = getattr(pert, "_delta_yaw", None)
        if delta_xyz is None or delta_yaw is None:
            continue
        if np.allclose(delta_xyz, 0.0) and abs(delta_yaw) < 1e-12:
            continue
        # Anchor lives on the perturbation's scene_cfg.gate_region.
        scene_cfg_ref = getattr(pert, "scene_cfg", None)
        region = (scene_cfg_ref or {}).get("gate_region") or {}
        anchor = region.get("anchor")
        return {
            "delta_xyz_mocap": list(map(float, delta_xyz)),
            "delta_yaw_rad": float(delta_yaw),
            "anchor_mocap": list(map(float, anchor)) if anchor is not None else None,
        }
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_episode(
    cfg: EpisodeConfig,
    *,
    policy_factory: Callable[[Point, dict], Policy],
    renderer: Optional[Any] = None,
    detector_factory: Optional[Callable[[FrameGraph, dict], Any]] = None,
    recovery_factory: Optional[Callable[[FrameGraph, dict], Any]] = None,
    recovery_triggers: Optional[Any] = None,
    perturbations_factory: Optional[Callable[[FrameGraph, dict], Any]] = None,
    rng: "np.random.Generator | None" = None,
    initial_state_override: Optional[DroneState] = None,
    perturbation_overrides: Optional[dict] = None,
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

    if initial_state_override is not None:
        # Trial-card / replay path: use the absolute start state directly.
        assert_frame(initial_state_override.pos, "ned")
        initial_state = initial_state_override
    else:
        initial_state = build_initial_state(cfg.scene_cfg, frame_graph, rng=rng)
    goal = goal_in_ned(cfg.scene_cfg, frame_graph)

    policy = policy_factory(goal, cfg.episode_cfg.get("policy", {}))

    # `renderer` may be a callable (the legacy `.render` callable) or the
    # GSplatRenderer object itself — env perturbations need the object so
    # they can mutate the live gsplat (`apply_dynamic_edits`). Detect and
    # split here so sensor wiring keeps receiving the callable.
    render_callable = renderer
    renderer_obj = None
    if renderer is not None and not callable(renderer):
        renderer_obj = renderer
        render_callable = renderer.render
    elif renderer is not None and hasattr(renderer, "apply_dynamic_edits"):
        # Object that's also callable (unlikely but possible) — keep both refs.
        renderer_obj = renderer

    sensor_rig = build_sensor_rig(
        policy.required_modalities,
        frame_graph=frame_graph,
        frame_cfg=cfg.frame_cfg,
        renderer=render_callable,
        body_to_world=camera_to_world_pose,
        prompt=cfg.scene_cfg.get("prompt"),
    )

    sim_cfg = SimulatorConfig(
        hz=int(cfg.episode_cfg.get("hz", 10)),
        horizon_s=float(cfg.episode_cfg.get("horizon_s", 5.0)),
        policy_hz=int(cfg.episode_cfg.get("policy_hz", 1)),
        chunk_steps=(
            int(cfg.episode_cfg["chunk_steps"])
            if cfg.episode_cfg.get("chunk_steps") is not None else None
        ),
    )
    sim = Simulator(sim_cfg, frame_graph)
    sim.reset(initial_state)

    # Perturbations are constructed + reset *before* the detector so that
    # safety criteria can read the perturbation's resolved deltas (e.g.
    # GateRigidPerturbation's Δxyz/Δyaw) and transform aperture corners /
    # collision PLYs to match the moved gate Gaussians. Without this
    # ordering a gate-jitter trial would check against the un-moved
    # aperture and the miss-gate criterion would be wrong.
    perturbations = None
    if perturbations_factory is not None:
        # Inject the parsed scene_cfg so env perturbations (e.g.
        # `GateRigidPerturbation`) can read `gate_region:` at construction.
        pert_cfg = dict(cfg.episode_cfg.get("perturbations", {}))
        pert_cfg.setdefault("scene_cfg", cfg.scene_cfg)
        # Trial-card / replay path: pass absolute per-perturbation overrides
        # (keyed by perturbation name) into the factory so the suite can
        # bypass its own RNG sampling.
        if perturbation_overrides is not None:
            pert_cfg["overrides"] = perturbation_overrides
        perturbations = perturbations_factory(frame_graph, pert_cfg)
        perturbations.reset()
        if renderer_obj is not None:
            perturbations.apply_environment(renderer_obj)
        elif perturbations.environment_perts:
            raise ValueError(
                "Perturbation suite has environment perturbations but no "
                "renderer object was provided to run_episode. Pass the "
                "GSplatRenderer instance (not just its .render method)."
            )

    detector = None
    # Compute gate_deltas once — used both by the safety layer (collision
    # PLYs + aperture corners follow the moved gate) and by the post-hoc
    # classifier (gate AABB is transported through the same Δ).
    gate_deltas = _extract_gate_deltas(perturbations)
    if detector_factory is not None:
        safety_cfg = dict(cfg.episode_cfg.get("safety", {}))
        if gate_deltas is not None:
            safety_cfg["_gate_deltas"] = gate_deltas
        detector = detector_factory(frame_graph, safety_cfg)

    trace = sim.rollout_with_policy(
        policy, sensor_rig, detector=detector, perturbations=perturbations,
    )

    recovery_traj: Optional[Trajectory] = None
    recovery_trace: Optional[EpisodeTrace] = None
    recovery_seed_info: Optional[dict] = None
    if trace.failure is not None and recovery_factory is not None:
        # Optional failure-type filter: if recovery_triggers is set, skip
        # recovery for failure types outside it (e.g. don't recover from
        # EXCESSIVE_VELOCITY / EXCESSIVE_TILT — those are sim instabilities,
        # not falsifications).
        if (recovery_triggers is None
                or trace.failure.failure_type in recovery_triggers):
            planner = recovery_factory(frame_graph, cfg.episode_cfg.get("recovery", {}))
            # Choose the recovery seed from the detector's full safe-state
            # history with a failure-type bias (early for miss-gate / non-gate
            # collisions; late for gate clips). Falls back to last_safe if
            # the history is empty (very early failure) or no rng provided.
            from falsify.recovery import sample_recovery_seed, bias_for
            seed_rng = rng if rng is not None else np.random.default_rng(0)
            history = trace.failure.safe_history
            # MissGateCriterion stamps `transit_time` into Violation.extra
            # when GOAL_NOT_REACHED fires; the detector merges that into
            # FailureRecord.extra. Pass it through so the sampler can scope
            # the draw to post-transit safe states.
            transit_time = trace.failure.extra.get("transit_time")
            if history:
                seed_step, seed_state = sample_recovery_seed(
                    history, trace.failure.failure_type, seed_rng,
                    transit_time=transit_time,
                )
            else:
                seed_step = trace.failure.last_safe_step
                seed_state = trace.failure.last_safe_state
            n_post_transit = (
                sum(1 for _, st in history
                    if transit_time is not None and float(st.t) >= float(transit_time))
                if transit_time is not None else None
            )
            recovery_seed_info = {
                "step": int(seed_step),
                "bias": bias_for(trace.failure.failure_type),
                "n_safe": len(history),
                "transit_time": transit_time,
                "n_post_transit": n_post_transit,
            }
            result = planner.plan(seed_state.pos, goal)
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
            "recovery_seed": recovery_seed_info,
            "gate_deltas": gate_deltas,
        },
    )
