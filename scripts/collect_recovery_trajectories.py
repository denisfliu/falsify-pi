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
        PYTHONPATH=src python scripts/collect_recovery_trajectories.py \\
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


REPO_ROOT = Path(__file__).resolve().parent.parent


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
    from falsify.recovery import CoursedMpcPlanner
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
        ["MISS_GATE", "COLLISION_GATE", "COLLISION_OTHER", "OUT_OF_BOUNDS"],
    )
    recovery_triggers = {FailureType[name] for name in triggers_raw}

    region = scene_cfg.get("gate_region") or {}
    anchor_mocap = list(region["anchor"]) if region else None

    # ---- Streaming loop ----------------------------------------------
    sampled = 0
    recovered = 0
    by_outcome: dict[str, int] = {}
    n_errors = 0
    t_start = time.time()

    trials_root = args.out / scene_key  # mirror eval-campaign layout for viz

    while recovered < args.n_recoveries and sampled < args.max_trials:
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
        pgcfg = PiGatewayConfig(
            gateway_url=policy_cfg_yaml["gateway_url"],
            api_key=policy_cfg_yaml.get("api_key", ""),
            execute_chunk_size=int(
                args.execute_chunk_size
                if args.execute_chunk_size is not None
                else policy_cfg_yaml.get("execute_chunk_size", 25)
            ),
            prompt=prompt_text,
            hz=int(policy_cfg_yaml.get("hz", 30)),
            state_dim=int(policy_cfg_yaml.get("state_dim", 7)),
            action_dim=int(policy_cfg_yaml.get("action_dim", 7)),
            action_pos_slice=tuple(policy_cfg_yaml.get("action_pos_slice", (0, 3))),
            action_yaw_index=policy_cfg_yaml.get("action_yaw_index", 3),
            camera_map=dict(policy_cfg_yaml.get("camera_map") or {}),
            state_key=policy_cfg_yaml.get("state_key", "observation/state"),
            server_frame=policy_cfg_yaml.get("server_frame", "mocap"),
            bridge_admin_url=policy_cfg_yaml.get("bridge_admin_url"),
            bridge_policy_id=policy_cfg_yaml.get("bridge_policy_id"),
            use_rtc=(False if args.no_rtc
                     else bool(policy_cfg_yaml.get("use_rtc", False))),
            image_size=policy_cfg_yaml.get("image_size"),
            channel_order=str(policy_cfg_yaml.get("channel_order", "RGB")),
            traceability=dict(policy_cfg_yaml.get("traceability") or {}),
            record_dir=record_dir,
        )
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

        def recovery_factory(fg2, _episode_recovery_cfg):
            return CoursedMpcPlanner(
                course_path=course_path,
                frame_graph=fg2,
                planner=planner_kind,
                prompt=prompt_text,
                gate_deltas=gp_for_recovery,
                scene_cfg=scene_cfg,
            )

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
                recovery_factory=recovery_factory,
                recovery_triggers=recovery_triggers,
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

        # 5) Harvest the recovery NPZ if it fired.
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
            save_trajectory(recoveries_dir / f"recovery_{recovered:03d}.npz",
                            traj_to_save)
            # Also drop a copy inside the trial dir for context.
            save_trajectory(trial_dir / "recovery_trajectory.npz", traj_to_save)
            recovered += 1

        sampled += 1
        rec_tag = f"recovery#{recovered-1:03d}" if recovery_fired else "no-recovery"
        print(f"   → outcome={outcome} {rec_tag} "
              f"({recovered}/{args.n_recoveries})  elapsed={dt:.1f}s")

        # Live-checkpoint the manifest every iteration so a dashboard or
        # tail watcher sees current progress without waiting for the final
        # write. ~kB of disk per trial; negligible vs the rollout cost.
        _write_manifest(
            final=False, sampled_=sampled, recovered_=recovered,
            n_errors_=n_errors, by_outcome_=by_outcome,
            elapsed_total_=time.time() - t_start,
        )

    elapsed_total = time.time() - t_start
    print(f"\n[collect] done: collected {recovered}/{args.n_recoveries} recoveries "
          f"from {sampled} trials ({n_errors} errors) in {elapsed_total:.0f}s "
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
