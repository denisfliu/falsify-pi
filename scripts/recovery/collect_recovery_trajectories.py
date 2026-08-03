"""Stream-sample gate perturbations and harvest MPC recovery trajectories
for one (policy, scene) pair.

For each iteration we draw a fresh trial card from the perturbation
recipe, roll the policy out via the orchestrator with the
``CoursedMpcPlanner`` recovery wired, and persist the recovery NPZ when
one fires. The loop stops as soon as ``--n-recoveries`` NPZs are saved
or ``--max-trials`` is exhausted.

The deliverable is the ``recoveries/`` directory holding canonical
Trajectory NPZs ready for ``falsify.cli.export_training_data
--trajectory`` (or ``falsify-orchestrate-batch`` for bulk parquet
generation against the same scene).

Output layout (auto-numbered ``run-NNN-<ts>`` per (policy, scene)):

    runs/recovery_collection/<policy_id>/<scene_key>/run-NNN-<ts>/
    ├── collection_manifest.json
    ├── policy_manifest.json
    ├── collection.log
    ├── recoveries/
    │   ├── recovery_000.npz       # numbered by collection order
    │   └── ...
    ├── <scene_key>/trial_NNN/     # per attempt (incl. SUCCESSes); diagnostic
    │   ├── trial_card.json
    │   ├── episode_summary.json
    │   ├── rollout_states.npz
    │   └── recovery_trajectory.npz   # only when recovery fired (copy)
    └── viz/
        └── trajectories.html

Usage (single scene; the convenience driver ``tools/collect_recoveries_*``
loops over multiple scenes):

    bash -c 'export PI_API_KEY=...; source tools/env.sh; \\
        PYTHONPATH=src python scripts/recovery/collect_recovery_trajectories.py \\
            --policy-config configs/policies/pi_gateway/<policy>.yaml \\
            --scene         configs/scenes/<scene>.yaml \\
            --safety        configs/safety/<scene>.yaml \\
            --recovery      configs/recovery/<scene>_mpc.yaml \\
            --frame         configs/frames/carl_dual.yaml \\
            --perturbation-recipe configs/eval_suite/gate_perturbed_small.yaml \\
            --prompt-name   <scene_key> \\
            --n-recoveries  50 \\
            --max-trials    500'
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _serialize_drone_state(state) -> dict:
    """JSON-friendly view of a ``DroneState`` for the Phase-1→Phase-2
    failure_record.json. NED-frame positions only — Phase 2 reconstructs
    the Point via the active FrameGraph."""
    return {
        "pos_ned": np.asarray(state.pos.xyz, dtype=np.float64).tolist(),
        "vel": np.asarray(state.vel, dtype=np.float64).tolist(),
        "quat_xyzw": np.asarray(state.quat_xyzw, dtype=np.float64).tolist(),
        "t": float(state.t),
    }


def _jsonable_extra(extra: dict) -> dict:
    """Strip numpy / non-JSON types out of FailureRecord.extra so it can
    be persisted into failure_record.json."""
    out = {}
    for k, v in extra.items():
        if isinstance(v, (np.floating, float)):
            out[k] = float(v)
        elif isinstance(v, (np.integer, int)):
            out[k] = int(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (str, bool, type(None))):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = [_jsonable_extra({"_": x})["_"] for x in v]
        elif isinstance(v, dict):
            out[k] = _jsonable_extra(v)
        else:
            out[k] = str(v)
    return out


def _resolve(p: str | Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (REPO_ROOT / pp).resolve()


_RUN_DIR_RE = re.compile(r"^run-(\d+)-")


def _next_run_number(root: Path) -> int:
    if not root.is_dir():
        return 1
    nums = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        m = _RUN_DIR_RE.match(child.name)
        if m:
            nums.append(int(m.group(1)))
    return (max(nums) + 1) if nums else 1


def _derive_out_dir(policy_config: Path, scene_key: str) -> Path:
    policy_id = policy_config.stem
    ts = time.strftime("%Y%m%d_%H%M%S")
    root = REPO_ROOT / "runs" / "recovery_collection" / policy_id / scene_key
    n = _next_run_number(root)
    return root / f"run-{n:03d}-{ts}"


def _card_gate_deltas(card: dict, scene_cfg: dict):
    """Trial-card gate perturbation → the `_gate_deltas` dict the detector
    factory uses to shift collision clouds / aperture corners (mirrors
    falsify.orchestrator._extract_gate_deltas)."""
    gp = card.get("gate_perturbation")
    if gp is None:
        return None
    region = scene_cfg.get("gate_region") or {}
    anchor = region.get("anchor")
    return {
        "delta_xyz_mocap": [float(v) for v in gp["delta_xyz"]],
        "delta_yaw_rad": float(gp["delta_yaw_rad"]),
        "anchor_mocap": ([float(v) for v in anchor]
                         if anchor is not None else None),
    }


class _Tee:
    """Tee stdout/stderr to a sink file while preserving the primary
    stream, mirroring run_eval_campaign.py's _Tee."""

    def __init__(self, primary, sink):
        self._primary = primary
        self._sink = sink

    def write(self, data):
        self._primary.write(data)
        try:
            self._sink.write(data)
        except Exception:
            pass
        return len(data)

    def flush(self):
        self._primary.flush()
        try:
            self._sink.flush()
        except Exception:
            pass

    def isatty(self):
        return getattr(self._primary, "isatty", lambda: False)()

    def fileno(self):
        return self._primary.fileno()


def _git_rev() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def _load_prompt_text(prompt_name: str) -> str:
    """Resolve a prompt_name against the same registries
    generate_eval_bundles.py uses (atomic + compositional)."""
    from falsify.io import load_yaml
    prompts: dict = {}
    for rel in ("configs/prompts/atomic_dataset_prompts.yaml",
                "configs/prompts/compositional_prompts.yaml"):
        p = _resolve(rel)
        if p.is_file():
            prompts.update(load_yaml(p).get("prompts", {}))
    entry = prompts.get(prompt_name)
    if entry is None:
        raise SystemExit(
            f"prompt_name {prompt_name!r} not in atomic/compositional "
            f"prompt registries (configs/prompts/*.yaml)"
        )
    return entry["task"]


def _build_perturbation_suite(recipe: dict, scene_cfg: dict, gate_pert: dict):
    """Identical shape to run_eval_campaign.py:91 but takes the recipe
    dict directly (no scenario wrapper)."""
    from falsify.perturbations import GateRigidPerturbation, PerturbationSuite
    gp_recipe = recipe.get("gate_perturbation") or {}
    half_xyz = tuple(gp_recipe.get("offset_half_widths", [0.0, 0.0, 0.0]))
    half_yaw = float(gp_recipe.get("yaw_half_width_rad", 0.0))
    pert = GateRigidPerturbation(
        offset_half_widths=half_xyz,
        yaw_half_width_rad=half_yaw,
        scene_cfg=scene_cfg,
        name="gate_rigid_perturbation",
    )
    pert.set_absolute_deltas(
        delta_xyz=gate_pert["delta_xyz"],
        delta_yaw_rad=gate_pert["delta_yaw_rad"],
    )
    return PerturbationSuite(environment=[pert], seed=0)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policy-config", required=True, type=Path,
                    help="Pi-gateway policy YAML.")
    ap.add_argument("--scene", required=True, type=Path,
                    help="Scene YAML (e.g. configs/scenes/left_gate.yaml).")
    ap.add_argument("--safety", required=True, type=Path,
                    help="Safety YAML for the scene.")
    ap.add_argument("--recovery", required=True, type=Path,
                    help="Recovery YAML (planner + course + trigger types). "
                         "Required — the whole point of this script is "
                         "collecting recovery NPZs.")
    ap.add_argument("--frame", required=True, type=Path,
                    help="Drone-frame YAML (e.g. configs/frames/carl_dual.yaml).")
    ap.add_argument("--perturbation-recipe", type=Path, default=None,
                    help="Scenario YAML whose `recipe.gate_perturbation` "
                         "block defines the Δxyz/Δyaw bounds. Optional: "
                         "when omitted (or when the recipe has "
                         "`gate_perturbation.enabled: false`), every "
                         "trial runs with the nominal gate location and "
                         "the collector relies on the policy's natural "
                         "failure rate (typical for compositional tasks).")
    ap.add_argument("--prompt-name", required=True,
                    help="Prompt registry key (atomic_dataset_prompts.yaml or "
                         "compositional_prompts.yaml).")
    ap.add_argument("--scene-key-override", default=None,
                    help="Override the scene_key from the scene YAML. Use "
                         "when one scene file feeds multiple collection "
                         "entries (e.g. center_gate.yaml drives both "
                         "center_from_left and center_from_right). The "
                         "override is used for the output dir name, the "
                         "trials/<scene_key>/ subdir, the trial card, and "
                         "posthoc's directional-transit suffix lookup.")
    ap.add_argument("--n-recoveries", type=int, default=50,
                    help="Target number of recovery NPZs to collect. "
                         "Default 50.")
    ap.add_argument("--max-trials", type=int, default=500,
                    help="Safety cap on total rollouts. Exits early even if "
                         "fewer than --n-recoveries have fired. Default 500.")
    ap.add_argument("--collection-seed", type=int, default=100000,
                    help="Master seed for the per-trial seed_for() hash. "
                         "Default 100000 so cards don't collide with the "
                         "eval bundles (master_seed=0).")
    ap.add_argument("--horizon-s", type=float, default=25.0,
                    help="Per-trial time budget (default 25s = 750 steps "
                         "at 30 Hz).")
    ap.add_argument("--no-rtc", action="store_true",
                    help="Override the policy YAML's use_rtc to False. "
                         "Speeds up rollouts ~22× by querying the VLA once "
                         "per chunk; not byte-identical to a deployed RTC "
                         "checkpoint.")
    ap.add_argument("--no-gripper-overlay", action="store_true",
                    help="Strip the wrist-cam gripper overlay from this "
                         "collection run, regardless of what the policy "
                         "YAML declares. Ablation knob; resize + BGR swap "
                         "still happen.")
    ap.add_argument("--execute-chunk-size", type=int, default=None,
                    help="Override the policy YAML's execute_chunk_size.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Explicit run directory (skips auto-derivation).")
    ap.add_argument("--no-viz", action="store_true",
                    help="Skip the trajectories HTML emit at the end.")
    args = ap.parse_args(argv)

    policy_path = _resolve(args.policy_config)
    scene_path = _resolve(args.scene)
    safety_path = _resolve(args.safety)
    recovery_path = _resolve(args.recovery)
    frame_path = _resolve(args.frame)
    recipe_path = _resolve(args.perturbation_recipe) if args.perturbation_recipe else None

    # ---- Lazy import after arg validation ----------------------------
    from falsify.cli.run_vla_episode import _smoke_imports
    _smoke_imports(policy_backend="pi_gateway")

    from falsify.cli.smoke_test import _build_detector_factory
    from falsify.eval.sampling import (
        sample_gate_perturbation, sample_start_mocap, seed_for,
    )
    from falsify.geometry import Point
    from falsify.io import build_frame_graph, load_yaml
    from falsify.orchestrator import EpisodeConfig, run_episode
    from falsify.policy import PiGatewayConfig, PiGatewayPolicy
    from falsify.recovery import CoursedMpcPlanner  # noqa: F401 — back-compat
    from falsify.recovery.splatnav_mpc import SplatNavMpcPlanner
    from falsify.safety import FailureType
    from falsify.sim import DroneState
    from falsify.sim.renderer import GSplatRenderer
    from falsify.training import save_trajectory
    from falsify.training.trajectory import Trajectory as TrainingTrajectory

    scene_cfg = load_yaml(scene_path)
    scene_dir = scene_path.parent
    scene_key = args.scene_key_override or scene_cfg.get("scene_key") or scene_path.stem

    safety_cfg = load_yaml(safety_path)
    recovery_cfg = load_yaml(recovery_path)
    frame_cfg = load_yaml(frame_path)

    # Recipe is optional; without one the collector never samples a
    # gate perturbation (every trial uses the nominal gate location).
    # Useful for compositional collection where natural failures are
    # plentiful and no perturbation is needed.
    if recipe_path is not None:
        recipe = load_yaml(recipe_path).get("recipe", {})
        recipe_pert_enabled = bool(
            (recipe.get("gate_perturbation") or {}).get("enabled", False)
        )
    else:
        recipe = {}
        recipe_pert_enabled = False

    prompt_text = _load_prompt_text(args.prompt_name)

    policy_cfg_yaml = load_yaml(policy_path)
    if policy_cfg_yaml.get("type") != "pi_gateway":
        raise SystemExit("only pi_gateway policy configs are supported "
                         "(see run_vla_episode.py for the openpi path).")
    policy_sha = hashlib.sha256(policy_path.read_bytes()).hexdigest()

    # ---- Output dir + log capture ------------------------------------
    if args.out is None:
        args.out = _derive_out_dir(policy_path, scene_key)
    args.out.mkdir(parents=True, exist_ok=True)
    log_fp = open(args.out / "collection.log", "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_fp)
    sys.stderr = _Tee(sys.__stderr__, log_fp)

    recoveries_dir = args.out / "recoveries"
    recoveries_dir.mkdir(exist_ok=True)

    print(f"[collect] policy={policy_path.name} scene_key={scene_key}")
    print(f"[collect] target n_recoveries={args.n_recoveries} "
          f"max_trials={args.max_trials} collection_seed={args.collection_seed}")
    print(f"[collect] out={args.out}")

    # ---- Pre-flight manifest -----------------------------------------
    (args.out / "policy_manifest.json").write_text(json.dumps({
        "policy_config_path": str(policy_path),
        "policy_config_sha256": policy_sha,
        "bridge_admin_url": policy_cfg_yaml.get("bridge_admin_url"),
        "bridge_policy_id": policy_cfg_yaml.get("bridge_policy_id"),
        "traceability": policy_cfg_yaml.get("traceability") or {},
    }, indent=2))

    recipe_sha = (
        hashlib.sha256(recipe_path.read_bytes()).hexdigest()
        if recipe_path is not None else None
    )
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")

    # Write a preliminary collection_manifest.json with the planning
    # fields up-front so a live dashboard can read `target_n_recoveries`
    # immediately rather than waiting for the run to finish. We rewrite
    # it at the end with the final `stats` + `finished_at`.
    def _write_manifest(final: bool, sampled_: int, recovered_: int,
                        n_errors_: int, by_outcome_: dict,
                        elapsed_total_: float):
        (args.out / "collection_manifest.json").write_text(json.dumps({
            "cli_argv": list(argv) if argv is not None else sys.argv[1:],
            "policy_id": policy_path.stem,
            "scene_key": scene_key,
            "scene_yaml": str(scene_path),
            "safety_yaml": str(safety_path),
            "recovery_yaml": str(recovery_path),
            "perturbation_recipe": (str(recipe_path) if recipe_path else None),
            "perturbation_recipe_sha256": recipe_sha,
            "perturbation_bounds": recipe.get("gate_perturbation"),
            "prompt_name": args.prompt_name,
            "target_n_recoveries": args.n_recoveries,
            "max_trials": args.max_trials,
            "collection_seed": args.collection_seed,
            "horizon_s": args.horizon_s,
            "no_rtc": bool(args.no_rtc),
            "execute_chunk_size_override": args.execute_chunk_size,
            "git_rev": _git_rev(),
            "started_at": started_at,
            "finished_at": (time.strftime("%Y-%m-%dT%H:%M:%S") if final else None),
            "elapsed_total_s": (elapsed_total_ if final else None),
            "stats": {
                "n_sampled": sampled_,
                "n_recovered": recovered_,
                "n_errors": n_errors_,
                "by_outcome": by_outcome_,
            },
            "recoveries": sorted(
                str(p.relative_to(args.out))
                for p in recoveries_dir.glob("recovery_*.npz")
            ),
        }, indent=2))

    _write_manifest(final=False, sampled_=0, recovered_=0, n_errors_=0,
                    by_outcome_={}, elapsed_total_=0.0)

    # ---- Shared, per-scene resources (build ONCE) --------------------
    fg = build_frame_graph(scene_cfg, base_path=scene_dir)
    renderer = GSplatRenderer.from_scene_cfg(scene_cfg, scene_dir=scene_dir)
    detector_factory = _build_detector_factory(scene_cfg, scene_dir)

    # Recovery wiring (constants — not per-trial).
    course_path_raw = recovery_cfg.get("course")
    if course_path_raw is None:
        raise SystemExit(f"recovery YAML {recovery_path} must set `course:`")
    course_path = _resolve(course_path_raw)
    planner_kind = recovery_cfg.get("planner", "mpc")
    triggers_raw = recovery_cfg.get(
        "trigger_failure_types",
        ["MISS_GATE", "COLLISION_GATE", "COLLISION_OTHER",
         "GOAL_NOT_REACHED", "OUT_OF_BOUNDS"],
    )
    # EXCESSIVE_VELOCITY / EXCESSIVE_TILT are sim instabilities, not
    # falsification targets. Drop them from the trigger set even if a
    # legacy recovery YAML lists them.
    triggers_raw = [
        t for t in triggers_raw
        if t not in ("EXCESSIVE_VELOCITY", "EXCESSIVE_TILT")
    ]
    recovery_triggers = {FailureType[name] for name in triggers_raw}

    region = scene_cfg.get("gate_region") or {}
    anchor_mocap = list(region["anchor"]) if region else None

    # ---- Phase 1: rollouts (no recovery) -----------------------------
    # We split the loop into two phases so the SplatPlan-based recovery
    # planner doesn't have to share the GPU with the renderer's gsplat
    # workspace (the original interleaved design hit OOM during SplatPlan's
    # spline solver). Phase 1 here runs all rollouts with recovery
    # disabled, persisting each failed trial's FailureRecord; Phase 2
    # below frees the renderer, constructs a standalone SplatNavMpcPlanner
    # with its own gsplat, and plans recoveries for the saved failures.
    sampled = 0
    n_failures_to_replan = 0   # counts failed-trials matching recovery_triggers
    recovered = 0              # set in Phase 2 below
    by_outcome: dict[str, int] = {}
    n_errors = 0
    failed_trials: list[tuple[int, Path, dict, dict]] = []   # (trial_idx, trial_dir, card, failure_record)
    t_start = time.time()

    trials_root = args.out / scene_key  # mirror eval-campaign layout for viz

    while n_failures_to_replan < args.n_recoveries and sampled < args.max_trials:
        trial_idx = sampled
        trial_seed = seed_for(args.collection_seed,
                              "recovery_collection",
                              scene_key, trial_idx)
        rng = np.random.default_rng(trial_seed)

        # 1) Sample card.
        start_mocap = sample_start_mocap(scene_cfg, rng, enabled=True)
        gate_pert = sample_gate_perturbation(recipe, rng, scene_cfg=scene_cfg)
        # Card schema mirrors generate_eval_bundles.py:314–333.
        start_pt = Point.of(*start_mocap, fg.frame("mocap"))
        start_ned = list(map(float, fg.convert(start_pt, to="ned").xyz))
        card = {
            "source": "recovery_collection",
            "scene": str(scene_path.relative_to(REPO_ROOT)),
            "scene_key": scene_key,
            "safety": str(safety_path.relative_to(REPO_ROOT)),
            "recovery": str(recovery_path.relative_to(REPO_ROOT)),
            "prompt_name": args.prompt_name,
            "prompt": prompt_text,
            "trial_index": trial_idx,
            "master_seed": args.collection_seed,
            "trial_seed": int(trial_seed),
            "start_position_mocap": start_mocap,
            "start_ned": start_ned,
            "gate_perturbation": gate_pert,
        }
        trial_dir = trials_root / f"trial_{trial_idx:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "trial_card.json").write_text(json.dumps(card, indent=2))

        # 2) Build the per-trial policy, detector, recovery, suite.
        record_dir = trial_dir / "vla_io"
        pgcfg = PiGatewayConfig.from_yaml(
            policy_path,
            prompt_override=prompt_text,
            execute_chunk_size_override=args.execute_chunk_size,
            use_rtc_override=(False if args.no_rtc else None),
            record_dir=record_dir,
        )
        if args.no_gripper_overlay:
            pgcfg.gripper_overlay_paths = {}
        effective_hz = pgcfg.hz
        effective_chunk = 1 if pgcfg.use_rtc else pgcfg.execute_chunk_size

        def policy_factory(_goal_ned, _ec):
            return PiGatewayPolicy(pgcfg, build_frame_graph(scene_cfg, base_path=scene_dir))

        gp_for_recovery = None
        if gate_pert is not None and region:
            gp_for_recovery = {
                "anchor_mocap": anchor_mocap,
                "delta_xyz_mocap": list(gate_pert["delta_xyz"]),
                "delta_yaw_rad": float(gate_pert["delta_yaw_rad"]),
            }

        # Phase 1 runs rollouts only — no recovery_factory is passed to
        # run_episode below. Recovery planning happens in Phase 2 (after
        # the renderer is freed and the GPU has room for SplatPlan).

        suite_factory = None
        override = None
        if gate_pert is not None:
            def suite_factory(_fg, _ec):
                return _build_perturbation_suite(recipe, scene_cfg, gate_pert)
            override = {"gate_rigid_perturbation": {
                "delta_xyz": gate_pert["delta_xyz"],
                "delta_yaw_rad": gate_pert["delta_yaw_rad"],
            }}

        episode_cfg = {
            "hz": effective_hz,
            "horizon_s": args.horizon_s,
            "chunk_steps": effective_chunk,
            "safety": safety_cfg,
        }
        ec = EpisodeConfig(
            scene_cfg=scene_cfg, frame_cfg=frame_cfg,
            episode_cfg=episode_cfg, scene_cfg_dir=scene_dir,
        )
        start_state = DroneState(
            pos=Point(xyz=np.asarray(start_ned, dtype=np.float64),
                      frame=fg.frame("ned")),
            vel=np.zeros(3),
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            t=0.0,
        )

        # 3) Run the episode.
        if gate_pert is not None:
            pert_tag = (f"Δxyz={[round(v, 3) for v in gate_pert['delta_xyz']]} "
                        f"Δyaw={gate_pert['delta_yaw_rad']:.3f}")
        else:
            pert_tag = "no-pert"
        print(f"[run] trial_{trial_idx:03d}: "
              f"start_ned={np.round(start_ned, 3).tolist()} {pert_tag}")
        t_trial = time.time()
        # Per-trial rng for the recovery seed sampler. Without this the
        # orchestrator falls back to default_rng(0) every trial, so
        # Beta(α,β) draws are identical → every recovery starts from
        # the same percentile of the safe history. Derived from the
        # trial seed so two trials with the same seed reproduce.
        episode_rng = np.random.default_rng(trial_seed ^ 0x9E37_79B9_7F4A_7C15)
        try:
            episode = run_episode(
                ec,
                policy_factory=policy_factory,
                renderer=renderer,
                detector_factory=detector_factory,
                # Recovery deferred to Phase 2 — see top of loop comment.
                recovery_factory=None,
                recovery_triggers=None,
                perturbations_factory=suite_factory,
                rng=episode_rng,
                initial_state_override=start_state,
                perturbation_overrides=override,
            )
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            print(f"[error] trial_{trial_idx:03d}: {e}")
            (trial_dir / "error.txt").write_text(tb)
            n_errors += 1
            sampled += 1
            continue
        dt = time.time() - t_trial

        # 4) Persist trial artifacts (rollout + summary), mirroring
        # run_eval_campaign.py:599–671.
        rollout_npz_path = trial_dir / "rollout_states.npz"
        traj = episode.trace.trajectory()
        np.savez(
            rollout_npz_path,
            times=traj.times.astype(np.float64),
            positions_ned=traj.positions.astype(np.float64),
            quaternions_xyzw=(
                traj.quaternions.astype(np.float64)
                if traj.quaternions is not None
                else np.tile(np.array([0., 0., 0., 1.]),
                             (len(traj.positions), 1))
            ),
            velocities=(
                traj.velocities.astype(np.float64)
                if traj.velocities is not None
                else np.zeros_like(traj.positions)
            ),
            failure_step=np.array(
                -1 if episode.failure is None else episode.failure.failure_step,
                dtype=np.int64,
            ),
            failure_type=np.array(
                ("NONE" if episode.failure is None
                 else episode.failure.failure_type.name),
                dtype=object,
            ),
        )

        # Posthoc classification (so episode_summary matches the eval-campaign
        # schema — the viz emitter reads `posthoc_outcome`).
        from falsify.safety.posthoc import classify_trajectory_posthoc
        positions_mocap = np.asarray(
            fg.convert(traj, to="mocap").positions, dtype=np.float64,
        )
        horizon_steps = int(round(args.horizon_s * effective_hz))
        expected_dy_sign = None
        if scene_key.endswith("_from_left"):
            expected_dy_sign = -1
        elif scene_key.endswith("_from_right"):
            expected_dy_sign = +1
        posthoc = classify_trajectory_posthoc(
            positions_mocap=positions_mocap,
            scene_cfg=scene_cfg,
            runtime_failure_type=(
                episode.failure.failure_type if episode.failure is not None
                else None
            ),
            horizon_steps=horizon_steps,
            n_states=len(episode.trace.states),
            gate_deltas_mocap=(episode.metadata or {}).get("gate_deltas"),
            expected_dy_sign=expected_dy_sign,
        )

        outcome = posthoc["outcome"]
        by_outcome[outcome] = by_outcome.get(outcome, 0) + 1

        recovery_fired = episode.recovery_trajectory is not None

        summary = {
            "scene": str(scene_path),
            "scene_key": scene_key,
            "trial_index": trial_idx,
            "prompt": prompt_text,
            "start_ned": start_ned,
            "gate_perturbation": gate_pert,
            "hz": effective_hz,
            "actions_per_chunk": effective_chunk,
            "horizon_s": args.horizon_s,
            "policy_config": str(policy_path),
            "policy_traceability": pgcfg.traceability,
            "n_states": len(episode.trace.states),
            "n_chunks": len(episode.trace.policy_outputs),
            "failure": (None if episode.failure is None else {
                "step": episode.failure.failure_step,
                "type": episode.failure.failure_type.name,
                "criterion": episode.failure.criterion_name,
                "description": episode.failure.description,
                # Includes phase, which_gate, transit_time_1/2 when set —
                # essential for auditing the phase-aware seed scoping.
                "extra": dict(episode.failure.extra or {}),
            }),
            "goal_ned": episode.goal.xyz.tolist() if episode.goal is not None else None,
            "vla_io_dir": str(record_dir),
            "rollout_states_npz": str(rollout_npz_path),
            "perturbations_manifest": episode.metadata.get("perturbations"),
            # What step did the sampler pick and from what scope?
            "recovery_seed": episode.metadata.get("recovery_seed"),
            "elapsed_s": float(dt),
            "posthoc_outcome": outcome,
            "transited": posthoc["transited"],
            "transit_first_step": posthoc["first_inside_step"],
            "transit_last_step": posthoc["last_inside_step"],
            "gate_aabb_mocap": posthoc["aabb_mocap"],
            "recovery_fired": recovery_fired,
        }
        if expected_dy_sign is not None:
            summary["expected_dy_sign"] = expected_dy_sign
            summary["correct_crossings"] = posthoc.get("correct_crossings")
            summary["wrong_crossings"] = posthoc.get("wrong_crossings")
            summary["gate_plane_y_mocap"] = posthoc.get("gate_plane_y_mocap")
        (trial_dir / "episode_summary.json").write_text(json.dumps(summary, indent=2))

        # 5a) Persist FailureRecord for Phase 2 replanning. We only enqueue
        # trials whose failure_type is in recovery_triggers — that filters
        # sim instabilities (EXCESSIVE_VELOCITY/TILT) from the replay set,
        # matching the orchestrator's old in-loop behaviour.
        if (episode.failure is not None
                and episode.failure.failure_type in recovery_triggers):
            failure_record = {
                "failure_type": episode.failure.failure_type.name,
                "failure_step": int(episode.failure.failure_step),
                "criterion": episode.failure.criterion_name,
                "description": episode.failure.description,
                "extra": _jsonable_extra(episode.failure.extra or {}),
                "last_safe_step": int(episode.failure.last_safe_step or 0),
                "last_safe_state": _serialize_drone_state(
                    episode.failure.last_safe_state
                ),
                "safe_history": [
                    {"step": int(s), **_serialize_drone_state(st)}
                    for s, st in (episode.failure.safe_history or [])
                ],
                "goal_ned": (
                    episode.goal.xyz.tolist() if episode.goal is not None else None
                ),
            }
            (trial_dir / "failure_record.json").write_text(
                json.dumps(failure_record, indent=2)
            )
            failed_trials.append((trial_idx, trial_dir, card, failure_record))
            n_failures_to_replan += 1

        # 5) Harvest the recovery NPZ if it fired. (Phase 1: never fires
        # because recovery is disabled; kept for symmetry.)
        if recovery_fired:
            rt = episode.recovery_trajectory
            quats = (rt.quaternions if rt.quaternions is not None
                     else np.tile(np.array([0., 0., 0., 1.]),
                                  (len(rt.positions), 1)))
            traj_to_save = TrainingTrajectory(
                times=rt.times,
                positions_ned=rt.positions,
                quaternions_xyzw=quats,
                prompt=prompt_text,
                source="recovery",
            )
            # Same harvest guard as Phase 2 — never save a recovery whose
            # flight violates the perturbation-aligned safety criteria.
            from falsify.planning import validate_trajectory
            vres = validate_trajectory(
                traj_to_save, fg,
                scene_cfg=scene_cfg, scene_dir=scene_dir,
                safety_cfg=safety_cfg,
                gate_deltas=_card_gate_deltas(card, scene_cfg),
            )
            if not vres.ok:
                print(f"   [reject] trial_{trial_idx:03d}: recovery failed "
                      f"validation — {vres.summary()}")
                continue
            save_trajectory(recoveries_dir / f"recovery_{recovered:03d}.npz",
                            traj_to_save)
            # Also drop a copy inside the trial dir for context.
            save_trajectory(trial_dir / "recovery_trajectory.npz", traj_to_save)
            recovered += 1

        sampled += 1
        enq_tag = ("→queued for replan" if (
            episode.failure is not None
            and episode.failure.failure_type in recovery_triggers
        ) else "no-failure")
        print(f"   → outcome={outcome} {enq_tag} "
              f"(failures={n_failures_to_replan}/{args.n_recoveries})  "
              f"elapsed={dt:.1f}s")

        # Live-checkpoint the manifest every iteration so a dashboard or
        # tail watcher sees current progress without waiting for the final
        # write. ~kB of disk per trial; negligible vs the rollout cost.
        _write_manifest(
            final=False, sampled_=sampled, recovered_=recovered,
            n_errors_=n_errors, by_outcome_=by_outcome,
            elapsed_total_=time.time() - t_start,
        )

    phase1_elapsed = time.time() - t_start
    print(f"\n[collect] Phase 1 done: {sampled} trials, "
          f"{n_failures_to_replan} failures queued, {n_errors} errors "
          f"in {phase1_elapsed:.0f}s ({phase1_elapsed/max(sampled,1):.1f}s/trial)")

    # ---- Phase 1 → Phase 2 transition --------------------------------
    # Hand off the renderer's loaded nerfstudio pipeline to Phase 2's
    # planner BEFORE freeing the renderer wrapper. Loading a second
    # pipeline from disk costs ~30 s AND consumes ~10 GB of GPU memory
    # we don't have — the renderer's pipeline is already exactly what
    # SplatPlan needs (just with the rasterizer/Tw2g machinery on top
    # we won't use again).
    pipeline_for_phase2 = renderer.pipeline
    print("[collect] handing pipeline to Phase 2; freeing renderer wrapper + "
          "rasterizer/camera caches …")
    try:
        # Drop renderer-specific state we no longer need (camera cache,
        # baseline snapshots, FiGS impl wrapper) while keeping the
        # pipeline tensors alive via pipeline_for_phase2.
        renderer._cameras.clear()
        renderer._baseline_means = None
        renderer._baseline_quats = None
        del renderer
    except Exception:
        pass
    try:
        del detector_factory
    except Exception:
        pass
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass

    # ---- Phase 2: SplatNav + MPC recovery planning -------------------
    t_phase2 = time.time()
    if failed_trials:
        print(f"[collect] Phase 2: planning recoveries for {len(failed_trials)} "
              f"failed trials with standalone SplatNavMpcPlanner")
        from falsify.io import build_frame_graph
        from falsify.perturbations.environment import GateRigidPerturbation
        from falsify.recovery.seed_sampling import sample_recovery_seed
        from falsify.geometry import Point as _Pt
        from falsify.sim.dynamics_state import DroneState as _DroneState

        # Resolve the gsplat config path the same way GSplatRenderer does
        # (scene-cfg-relative). The standalone planner loads its own
        # nerfstudio pipeline from this path.
        # Re-use the freed-renderer's surviving pipeline tensors (no second
        # 30-s load, no second copy of ~10 GB of nerfstudio state).
        planner = SplatNavMpcPlanner(
            course_path=course_path,
            frame_graph=fg,
            pipeline=pipeline_for_phase2,
            prompt=prompt_text,
            scene_cfg=scene_cfg,
        )

        for (trial_idx, trial_dir, card, fr) in failed_trials:
            try:
                # Apply this trial's gate perturbation to the planner's
                # gsplat — same RigidTransformAABB the renderer applies,
                # via the renderer-compatible `apply_dynamic_edits` alias.
                if card.get("gate_perturbation") is not None:
                    pert = GateRigidPerturbation(
                        offset_half_widths=(0.0, 0.0, 0.0),
                        yaw_half_width_rad=0.0,
                        scene_cfg=scene_cfg,
                    )
                    pert.set_absolute_deltas(
                        card["gate_perturbation"]["delta_xyz"],
                        float(card["gate_perturbation"]["delta_yaw_rad"]),
                    )
                    pert.reset(np.random.default_rng(0))   # latches absolute deltas
                    pert.apply(planner)
                else:
                    planner.restore_baseline()

                # Reconstruct safe_history + last_safe_state in NED.
                ned_frame = fg.frame("ned")
                safe_history = [
                    (int(entry["step"]), _DroneState(
                        pos=_Pt(np.asarray(entry["pos_ned"], dtype=np.float64),
                                frame=ned_frame),
                        vel=np.asarray(entry["vel"], dtype=np.float64),
                        quat_xyzw=np.asarray(entry["quat_xyzw"], dtype=np.float64),
                        t=float(entry["t"]),
                    ))
                    for entry in fr["safe_history"]
                ]
                last_safe = _DroneState(
                    pos=_Pt(np.asarray(fr["last_safe_state"]["pos_ned"], dtype=np.float64),
                            frame=ned_frame),
                    vel=np.asarray(fr["last_safe_state"]["vel"], dtype=np.float64),
                    quat_xyzw=np.asarray(fr["last_safe_state"]["quat_xyzw"], dtype=np.float64),
                    t=float(fr["last_safe_state"]["t"]),
                )

                # ---- Phase-driven seed + target selection ------------
                # 1. Classify pre-trim phase from full safe_history.
                # 2. Trim the tail (drone OBB safe ≠ SplatPlan-radius safe;
                #    last ~0.5 s sits in an occupied voxel).
                # 3. Classify post-trim phase from the trimmed history.
                # 4. Scope safe_history to entries in post_phase.
                # 5. Seed = scope[-1] for collisions; Beta(1,3) sampled
                #    for non-collision failures.
                # 6. Target waypoint:
                #      Case A (pre_phase != post_phase): "in_gate" (push
                #          back across the boundary the trim regressed).
                #      Case B + collision: "in_gate" (re-attempt the gate).
                #      Case B + Beta(1,3): "pre_gate" (retry approach).
                ftype = FailureType[fr["failure_type"]]
                seed_rng = np.random.default_rng(
                    int(card["trial_seed"]) ^ 0xA1B2_C3D4_E5F6_0789
                )
                _trim_tail = int(round(0.5 * effective_hz))

                # Phase classification needs the full trajectory in MOCAP.
                from falsify.safety.posthoc import (
                    classify_phases_along_trajectory,
                    apertures_from_safety_cfg,
                )
                # Apply this trial's gate_deltas to the aperture corners
                # so the phase classifier sees where the gates ACTUALLY
                # were during the rollout.
                apertures = apertures_from_safety_cfg(safety_cfg)
                if card.get("gate_perturbation") is not None and apertures:
                    from falsify.safety.posthoc import apply_gate_deltas_to_cloud
                    gp = card["gate_perturbation"]
                    delta_dict = {
                        "anchor_mocap": anchor_mocap,
                        "delta_xyz_mocap": list(gp["delta_xyz"]),
                        "delta_yaw_rad": float(gp["delta_yaw_rad"]),
                    }
                    apertures = [
                        apply_gate_deltas_to_cloud(a, delta_dict)
                        for a in apertures
                    ]
                positions_mocap_full = np.stack([
                    fg.convert(st.pos, to="mocap").xyz
                    for _, st in safe_history
                ]) if safe_history else np.empty((0, 3))

                from falsify.planning.waypoints import load_course
                course = load_course(course_path)
                if not course.phases:
                    raise SystemExit(
                        f"course {course_path} has no `phases:` block — "
                        "the recovery pipeline requires it. See "
                        "configs/courses/through_left_gate.yaml for a "
                        "minimal example."
                    )

                if positions_mocap_full.shape[0] and apertures:
                    phases_per_step = classify_phases_along_trajectory(
                        positions_mocap_full, apertures,
                    )
                    pre_n = int(phases_per_step[-1])
                    trim_n = max(0, len(safe_history) - _trim_tail)
                    post_n = int(phases_per_step[trim_n - 1]) if trim_n > 0 else 0
                else:
                    pre_n = post_n = 0

                pre_phase  = course.phase_label(pre_n)
                post_phase = course.phase_label(post_n)
                phase_changed = (pre_phase != post_phase)

                # Trimmed safe-history scoped to post_phase entries.
                trimmed = safe_history[:max(0, len(safe_history) - _trim_tail)]
                if apertures and trimmed:
                    scope = [
                        (s, st) for (s, st), p in zip(trimmed, phases_per_step[:len(trimmed)])
                        if int(p) == post_n
                    ]
                else:
                    scope = trimmed

                is_collision = ftype in (
                    FailureType.COLLISION_GATE, FailureType.COLLISION_OTHER,
                )
                if not scope:
                    # Fallback: take the trimmed-history tail or the
                    # last_safe_state as a single seed.
                    seed_step, seed_state = (
                        (fr["last_safe_step"], last_safe) if not trimmed
                        else trimmed[-1]
                    )
                elif is_collision:
                    # No sampling for collisions — just the last state in scope.
                    seed_step, seed_state = scope[-1]
                else:
                    seed_step, seed_state = sample_recovery_seed(
                        scope, ftype, seed_rng, trim_tail=0,
                    )

                # Case A always routes to in_gate; Case B + collision
                # also routes to in_gate; everything else (Case B +
                # Beta-sampled) routes to pre_gate.
                seed_kind = (
                    "in_gate" if (phase_changed or is_collision)
                    else "pre_gate"
                )
                target_waypoint = course.target_waypoint(post_phase, seed_kind)

                # Plan.
                goal_pt = _Pt(
                    np.asarray(fr["goal_ned"], dtype=np.float64),
                    frame=ned_frame,
                )
                t_plan = time.time()
                # Re-bind per-trial gate_deltas onto the planner (it uses
                # them for course-waypoint warping + posthoc collision cloud).
                planner.gate_deltas = (
                    {
                        "anchor_mocap": anchor_mocap,
                        "delta_xyz_mocap": list(card["gate_perturbation"]["delta_xyz"]),
                        "delta_yaw_rad": float(card["gate_perturbation"]["delta_yaw_rad"]),
                    }
                    if card.get("gate_perturbation") is not None else None
                )
                planner._course = None  # force course reload with new gate_deltas
                print(f"   [phase] pre={pre_phase} post={post_phase} "
                      f"changed={phase_changed} failure={ftype.name} "
                      f"seed_kind={seed_kind} target={target_waypoint!r}")
                result = planner.plan(
                    seed_state.pos, goal_pt,
                    target_waypoint=target_waypoint,
                )
                plan_dt = time.time() - t_plan

                # Bounds-safety check: reject recoveries whose planned
                # trajectory wanders outside the scene's MOCAP bounds.
                # The MPC tracker can occasionally blow up under tight
                # constraints (we saw SQP_RTI status-3 warnings) and
                # produce trajectories that overshoot well past the
                # scene. We use the same bounds the SplatPlan voxel grid
                # is built against, with a small margin. Outside ⇒ skip
                # this trial; don't pollute the training data.
                rt_check = result.trajectory
                pos_mocap_check = np.stack([
                    fg.convert(_Pt(np.asarray(p, dtype=np.float64),
                                   frame=fg.frame("ned")), to="mocap").xyz
                    for p in rt_check.positions
                ])
                bounds_margin_m = 0.5
                lo = np.asarray(planner.cfg.bounds_lower_mocap) - bounds_margin_m
                hi = np.asarray(planner.cfg.bounds_upper_mocap) + bounds_margin_m
                oob_mask = (
                    (pos_mocap_check < lo).any(axis=1)
                    | (pos_mocap_check > hi).any(axis=1)
                )
                if oob_mask.any():
                    n_oob = int(oob_mask.sum())
                    first_oob = int(np.where(oob_mask)[0][0])
                    worst = pos_mocap_check[np.argmax(
                        np.maximum(
                            (lo - pos_mocap_check).max(axis=1),
                            (pos_mocap_check - hi).max(axis=1),
                        )
                    )]
                    print(f"   [skip] trial_{trial_idx:03d}: recovery left "
                          f"bounds (n_oob={n_oob}/{len(pos_mocap_check)}, "
                          f"first_step={first_oob}, worst_pos_mocap="
                          f"{worst.round(2).tolist()}, "
                          f"bounds=[{lo.tolist()}, {hi.tolist()}]); "
                          "not saving — MPC likely diverged.")
                    summary_path = trial_dir / "episode_summary.json"
                    summary_disk = json.loads(summary_path.read_text())
                    summary_disk["recovery_fired"] = False
                    summary_disk["recovery_skipped_reason"] = (
                        f"out_of_bounds: n_oob={n_oob}, "
                        f"first_step={first_oob}"
                    )
                    summary_path.write_text(json.dumps(summary_disk, indent=2))
                    n_errors += 1
                    continue

                # Save recovery NPZ + update episode summary.
                # The saved trajectory is the FULL drone path: the original
                # VLA-flown prefix (steps 0..seed_step-1 from rollout_states.npz)
                # concatenated with the MPC-planned recovery (seed → goal).
                # This makes the NPZ a complete start-to-goal demonstration
                # — the prefix is real flight (so the policy can learn
                # "everything was fine here") and the suffix is the
                # corrective plan ("here's how to recover from this drift").
                rt = result.trajectory
                recovery_pos   = np.asarray(rt.positions, dtype=np.float64)
                recovery_quats = (
                    np.asarray(rt.quaternions, dtype=np.float64)
                    if rt.quaternions is not None
                    else np.tile(np.array([0., 0., 0., 1.]),
                                 (len(recovery_pos), 1))
                )
                recovery_times = np.asarray(rt.times, dtype=np.float64)

                # Load the original rollout prefix from disk.
                rollout = np.load(trial_dir / "rollout_states.npz",
                                  allow_pickle=True)
                prefix_pos    = np.asarray(rollout["positions_ned"][:seed_step], dtype=np.float64)
                prefix_times  = np.asarray(rollout["times"][:seed_step], dtype=np.float64)
                if "quaternions_xyzw" in rollout.files:
                    prefix_quats = np.asarray(
                        rollout["quaternions_xyzw"][:seed_step], dtype=np.float64
                    )
                else:
                    prefix_quats = np.tile(
                        np.array([0., 0., 0., 1.]), (seed_step, 1)
                    )

                # Offset recovery times so they're continuous with the prefix.
                # Recovery starts at the seed_state; the prefix ends just
                # before it (step seed_step-1). The recovery's first sample
                # IS the seed state, so it picks up exactly where the
                # prefix left off — concat directly, no duplicate cut.
                if prefix_times.size > 0:
                    t_offset = float(seed_state.t)
                    recovery_times = recovery_times + (t_offset - recovery_times[0])

                full_pos    = np.concatenate([prefix_pos, recovery_pos], axis=0)
                full_quats  = np.concatenate([prefix_quats, recovery_quats], axis=0)
                full_times  = np.concatenate([prefix_times, recovery_times], axis=0)

                traj_to_save = TrainingTrajectory(
                    times=full_times,
                    positions_ned=full_pos,
                    quaternions_xyzw=full_quats,
                    prompt=prompt_text,
                    source="recovery",
                )

                # Refuse to harvest a trajectory that violates the
                # (perturbation-aligned) safety criteria — a recovery that
                # clips the moved gate or the table would poison the
                # training set. Collision clouds are shifted by this
                # trial's gate deltas, exactly like the rollout detector's.
                from falsify.planning import validate_trajectory
                vres = validate_trajectory(
                    traj_to_save, fg,
                    scene_cfg=scene_cfg, scene_dir=scene_dir,
                    safety_cfg=safety_cfg,
                    gate_deltas=_card_gate_deltas(card, scene_cfg),
                )
                if not vres.ok:
                    print(f"   [reject] trial_{trial_idx:03d}: recovery "
                          f"failed validation — {vres.summary()}")
                    (trial_dir / "recovery_rejected.json").write_text(
                        json.dumps({
                            "failure_step": vres.failure_step,
                            "failure_type": vres.failure_type,
                            "description": vres.description,
                        }, indent=2))
                    continue

                save_trajectory(
                    recoveries_dir / f"recovery_{recovered:03d}.npz",
                    traj_to_save,
                )
                save_trajectory(
                    trial_dir / "recovery_trajectory.npz", traj_to_save,
                )

                # Patch episode_summary.json so recovery_fired + info are
                # reflected.
                summary_path = trial_dir / "episode_summary.json"
                summary_disk = json.loads(summary_path.read_text())
                summary_disk["recovery_fired"] = True
                summary_disk["recovery_seed"] = {
                    "step": int(seed_step),
                    "t": float(seed_state.t),
                }
                summary_disk["recovery_info"] = {
                    **result.info,
                    "prefix_steps": int(prefix_pos.shape[0]),
                    "suffix_steps": int(recovery_pos.shape[0]),
                    "total_steps":  int(full_pos.shape[0]),
                }
                summary_path.write_text(json.dumps(summary_disk, indent=2))
                recovered += 1
                print(f"   [recovery#{recovered-1:03d}] trial_{trial_idx:03d}: "
                      f"plan={plan_dt:.1f}s  "
                      f"prefix={prefix_pos.shape[0]}  recovery={recovery_pos.shape[0]}  "
                      f"info={result.info}")
            except Exception as e:  # noqa: BLE001
                tb = traceback.format_exc()
                (trial_dir / "phase2_error.txt").write_text(tb)
                print(f"[error] Phase 2 trial_{trial_idx:03d}: {e}")
                n_errors += 1

    phase2_elapsed = time.time() - t_phase2
    elapsed_total = phase1_elapsed + phase2_elapsed
    print(f"\n[collect] Phase 2 done: {recovered}/{len(failed_trials)} recoveries "
          f"in {phase2_elapsed:.0f}s")
    print(f"[collect] TOTAL: {recovered}/{args.n_recoveries} recoveries from "
          f"{sampled} trials ({n_errors} errors) in {elapsed_total:.0f}s "
          f"({elapsed_total/max(sampled,1):.1f}s/trial)")
    print(f"[collect] by_outcome: {by_outcome}")

    # ---- Final manifest with finished_at -----------------------------
    _write_manifest(
        final=True, sampled_=sampled, recovered_=recovered,
        n_errors_=n_errors, by_outcome_=by_outcome,
        elapsed_total_=elapsed_total,
    )

    # ---- Optional viz -------------------------------------------------
    if not args.no_viz and recovered > 0:
        try:
            from falsify.visualization.eval_report import emit_trajectories_html
            p = emit_trajectories_html(args.out)
            print(f"[viz] wrote {p.relative_to(args.out.parent)} "
                  f"({p.stat().st_size // 1024} KB)")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] viz emit failed: {e}")

    print(f"[collect] artifacts → {args.out}")
    return 0 if recovered >= args.n_recoveries else 1


if __name__ == "__main__":
    raise SystemExit(main())
