"""Run an evaluation campaign over pre-generated trial cards.

Loads a scenario YAML + the corresponding bundle dir, then iterates every
trial card and runs one episode per card via the orchestrator. Start
position and gate perturbation come from the card (absolute values),
**not** from a seed — that's how we guarantee two policies see byte-
identical conditions on the same trial.

Per-trial outputs land under ``<out>/<scene_key>/trial_NNN/``; an
aggregate ``campaign_summary.json`` at the top of ``<out>`` records the
counts and policy metadata.

Usage:

    bash -c 'export PI_API_KEY=...; source tools/env.sh; \\
        source tools/pi_inference_env.sh; \\
        PYTHONPATH=src python scripts/run_eval_campaign.py \\
            --scenario configs/eval_suite/pure.yaml \\
            --bundle-dir runs/eval_bundles/pure \\
            --policy-config configs/policies/pi_gateway/history_h6jtbq0w_20k.yaml \\
            --frame configs/frames/carl_dual.yaml \\
            --out runs/eval_campaigns/pi07_history_pure'
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve(p: str | Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (REPO_ROOT / pp).resolve()


@dataclass
class TrialCard:
    path: Path
    data: dict

    @property
    def scene_key(self) -> str: return self.data["scene_key"]
    @property
    def trial_index(self) -> int: return int(self.data["trial_index"])
    @property
    def scene_yaml(self) -> Path: return _resolve(self.data["scene"])
    @property
    def safety_yaml(self) -> Path: return _resolve(self.data["safety"])
    @property
    def recovery_yaml(self) -> Optional[Path]:
        v = self.data.get("recovery")
        return _resolve(v) if v else None
    @property
    def prompt(self) -> str: return self.data["prompt"]
    @property
    def start_ned(self) -> np.ndarray: return np.asarray(self.data["start_ned"], dtype=np.float64)
    @property
    def gate_perturbation(self) -> Optional[dict]: return self.data.get("gate_perturbation")


def _load_trial_cards(bundle_dir: Path, scene_filter: Optional[list[str]],
                      trial_filter: Optional[list[int]]) -> list[TrialCard]:
    cards: list[TrialCard] = []
    for scene_dir in sorted(p for p in bundle_dir.iterdir() if p.is_dir()):
        if scene_filter and scene_dir.name not in scene_filter:
            continue
        for jf in sorted(scene_dir.glob("trial_*.json")):
            data = json.loads(jf.read_text())
            if trial_filter is not None and int(data["trial_index"]) not in trial_filter:
                continue
            cards.append(TrialCard(path=jf, data=data))
    return cards


def _build_perturbation_suite(scenario: dict, scene_cfg: dict,
                              gate_pert: Optional[dict]):
    """Construct a PerturbationSuite with one GateRigidPerturbation when the
    card carries gate-perturbation deltas. Otherwise return None.

    The half-widths from the scenario YAML are passed through purely for
    manifest purposes; `set_absolute_deltas` will override the sampled
    values on `reset()`.
    """
    if gate_pert is None:
        return None
    from falsify.perturbations import PerturbationSuite, GateRigidPerturbation

    gp_recipe = (scenario.get("recipe") or {}).get("gate_perturbation") or {}
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


def _build_initial_state(start_ned: np.ndarray):
    from falsify.geometry import Point
    from falsify.sim import DroneState
    return DroneState(
        pos=Point(xyz=start_ned, frame=None),   # frame filled below
        vel=np.zeros(3),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        t=0.0,
    )


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenario", required=True, type=Path,
                    help="Scenario YAML used to generate the bundles.")
    ap.add_argument("--bundle-dir", type=Path, default=None,
                    help="Path to the pre-generated trial-card directory. "
                         "Defaults to runs/eval_bundles/<scenario_name>.")
    ap.add_argument("--policy-config", required=True, type=Path,
                    help="Pi-gateway policy YAML "
                         "(configs/policies/pi_gateway/*.yaml).")
    ap.add_argument("--frame", required=True, type=Path,
                    help="Drone-frame YAML (e.g. configs/frames/carl_dual.yaml).")
    ap.add_argument("--out", required=True, type=Path,
                    help="Campaign output dir (will be created).")
    ap.add_argument("--horizon-s", type=float, default=25.0,
                    help="Per-trial time budget. Default 25 s = 750 steps "
                         "at the YAML's 30 Hz, matching the eval-spec "
                         "'hefty time budget' design.")
    ap.add_argument("--scenes", nargs="+", default=None,
                    help="Optional scene_key filter (e.g. --scenes left_gate).")
    ap.add_argument("--trials", nargs="+", type=int, default=None,
                    help="Optional trial-index filter (e.g. --trials 0 1 2).")
    ap.add_argument("--skip-flythrough", action="store_true",
                    help="Don't render the forward-camera mp4 flythrough — "
                         "saves ~30 s per trial.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip trials whose <trial_dir>/episode_summary.json "
                         "already exists.")
    ap.add_argument("--no-rtc", action="store_true",
                    help="Override the policy YAML's use_rtc to False. "
                         "With pi_local_bridge loaded against an RTC-capable "
                         "inference method (e.g. sample_actions_with_prefix), "
                         "this still works — bridge falls back to plain "
                         ".infer() when the client doesn't send "
                         "initial_noise/prefix_info. Speeds up eval ~22× by "
                         "querying the VLA once per chunk instead of once "
                         "per step. NOT byte-identical to a checkpoint "
                         "deployed under sample_actions_fixed_noise.")
    ap.add_argument("--execute-chunk-size", type=int, default=None,
                    help="Override the policy YAML's execute_chunk_size. "
                         "Set to 1 for MPC-style receding-horizon: query the "
                         "VLA every step, take the first action, re-observe. "
                         "Only takes effect when --no-rtc is also set (with "
                         "RTC, chunk is always 1).")
    ap.add_argument("--gif-trials-per-scene", type=int, default=0,
                    help="For the first N trials of each scene, render a "
                         "forward-camera GIF (`flythrough_forward.gif`) "
                         "alongside the trial's outputs. 0 = disabled. "
                         "Adds ~3–10s per selected trial.")
    ap.add_argument("--gif-every", type=int, default=3,
                    help="Subsample stride when --gif-trials-per-scene > 0. "
                         "Default 3 keeps GIFs under a few MB.")
    ap.add_argument("--gif-fps", type=int, default=10)
    recov = ap.add_mutually_exclusive_group()
    recov.add_argument("--no-recovery", dest="recovery_mode", action="store_const",
                       const="off", default="auto",
                       help="Skip recovery-trajectory planning even when trial "
                            "cards declare a recovery YAML. Useful for "
                            "evaluation-only sweeps where you don't want to "
                            "spend the MPC time on failed trials. NPZs are "
                            "not saved; per-trial summaries report "
                            "recovery=skipped.")
    recov.add_argument("--force-recovery", dest="recovery_mode", action="store_const",
                       const="force",
                       help="Plan a recovery for every failed trial, even "
                            "when the card has no recovery YAML — requires "
                            "--recovery-yaml-default to point at a fallback.")
    ap.add_argument("--recovery-yaml-default", type=Path, default=None,
                    help="Fallback recovery YAML used by --force-recovery "
                         "when a card's `recovery:` field is null.")
    args = ap.parse_args(argv)

    scenario_path = _resolve(args.scenario)
    scenario = _load_yaml_lite(scenario_path)
    scenario_name = scenario["name"]
    bundle_dir = args.bundle_dir or (REPO_ROOT / "runs" / "eval_bundles" / scenario_name)
    bundle_dir = _resolve(bundle_dir)
    if not bundle_dir.is_dir():
        raise SystemExit(f"bundle dir not found: {bundle_dir} "
                         f"(run scripts/generate_eval_bundles.py first)")

    cards = _load_trial_cards(bundle_dir, args.scenes, args.trials)
    if not cards:
        raise SystemExit(f"no trial cards matched in {bundle_dir}")

    print(f"[campaign] scenario={scenario_name} n_trials={len(cards)} "
          f"bundle_dir={bundle_dir}")
    print(f"[campaign] policy={args.policy_config}")
    print(f"[campaign] out={args.out}")
    n_cards_with_recovery = sum(1 for c in cards if c.recovery_yaml is not None)
    print(f"[campaign] recovery mode={args.recovery_mode!r}  "
          f"cards-with-recovery={n_cards_with_recovery}/{len(cards)}")

    # ---- Lazy imports (after smoke checks) -----------------------------
    from falsify.cli.run_vla_episode import _smoke_imports
    _smoke_imports(policy_backend="pi_gateway")

    from falsify.geometry import Point
    from falsify.io import build_frame_graph, load_yaml
    from falsify.orchestrator import EpisodeConfig, run_episode
    from falsify.policy import PiGatewayConfig, PiGatewayPolicy
    from falsify.sensors import build_sensor_rig  # noqa: F401  (validation)
    from falsify.sim import DroneState
    from falsify.sim.poses import camera_to_world_pose
    from falsify.sim.renderer import GSplatRenderer

    import hashlib
    policy_cfg_path_resolved = _resolve(args.policy_config)
    policy_cfg_yaml = load_yaml(policy_cfg_path_resolved)
    if policy_cfg_yaml.get("type") != "pi_gateway":
        raise SystemExit("only pi_gateway policy configs are supported by "
                         "run_eval_campaign.py (see run_vla_episode.py for the "
                         "openpi path).")
    policy_cfg_sha = hashlib.sha256(policy_cfg_path_resolved.read_bytes()).hexdigest()
    print(
        f"[campaign] policy: {policy_cfg_path_resolved.name} "
        f"sha256={policy_cfg_sha[:12]} "
        f"bridge_policy_id={policy_cfg_yaml.get('bridge_policy_id') or '(none)'} "
        f"bridge_admin_url={policy_cfg_yaml.get('bridge_admin_url') or '(none)'}"
    )
    frame_cfg = load_yaml(_resolve(args.frame))

    args.out.mkdir(parents=True, exist_ok=True)
    # Campaign-level policy manifest — one per campaign, since every trial
    # uses the same policy YAML. Per-trial drift would be a bug, so we
    # don't write one per trial.
    (args.out / "policy_manifest.json").write_text(json.dumps({
        "policy_config_path": str(policy_cfg_path_resolved),
        "policy_config_sha256": policy_cfg_sha,
        "bridge_admin_url": policy_cfg_yaml.get("bridge_admin_url"),
        "bridge_policy_id": policy_cfg_yaml.get("bridge_policy_id"),
        "traceability": policy_cfg_yaml.get("traceability") or {},
    }, indent=2))

    # ---- Group trials by scene so we build each renderer once ----------
    by_scene_key: dict[str, list[TrialCard]] = {}
    for c in cards:
        by_scene_key.setdefault(c.scene_key, []).append(c)

    aggregate: list[dict] = []
    t_total = time.time()
    for scene_key, scene_cards in by_scene_key.items():
        scene_yaml = scene_cards[0].scene_yaml
        scene_cfg = load_yaml(scene_yaml)
        scene_dir_p = scene_yaml.parent
        fg = build_frame_graph(scene_cfg, base_path=scene_dir_p)
        renderer = GSplatRenderer.from_scene_cfg(scene_cfg, scene_dir=scene_dir_p)
        # Build the forward-cam sensor once per scene if GIFs are requested
        # — `_render_flythrough` walks every state in the trace and re-renders
        # the forward cam at that pose. The first N trials in this scene
        # will get a flythrough_forward.gif alongside their outputs.
        fwd_sensor = None
        if args.gif_trials_per_scene > 0:
            from falsify.sensors.camera import make_camera_sensor_from_yaml
            from falsify.sim.poses import camera_to_world_pose as _c2w
            fwd_cam_yaml = frame_cfg["cameras"]["forward"]
            fwd_sensor = make_camera_sensor_from_yaml(
                "forward", fwd_cam_yaml, fg,
                renderer=renderer.render,
                body_to_world=_c2w,
            )
        gifs_rendered_this_scene = 0
        print(f"\n[campaign] === scene={scene_key} "
              f"({len(scene_cards)} trials) ===")

        for card in scene_cards:
            trial_dir = args.out / scene_key / f"trial_{card.trial_index:03d}"
            summary_path = trial_dir / "episode_summary.json"
            if args.resume and summary_path.is_file():
                print(f"[skip] {scene_key}/trial_{card.trial_index:03d} "
                      f"(episode_summary.json exists)")
                aggregate.append(json.loads(summary_path.read_text()))
                continue

            trial_dir.mkdir(parents=True, exist_ok=True)
            # Persist the card alongside outputs for traceability.
            (trial_dir / "trial_card.json").write_text(json.dumps(card.data, indent=2))

            # Build the policy fresh per trial — guarantees a clean reset on
            # the gateway side. record_dir is per-trial so vla_io stays
            # organised.
            record_dir = trial_dir / "vla_io"
            pgcfg = PiGatewayConfig(
                gateway_url=policy_cfg_yaml["gateway_url"],
                api_key=policy_cfg_yaml.get("api_key", ""),
                execute_chunk_size=int(
                    args.execute_chunk_size
                    if args.execute_chunk_size is not None
                    else policy_cfg_yaml.get("execute_chunk_size", 25)
                ),
                prompt=card.prompt,
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
                return PiGatewayPolicy(pgcfg, build_frame_graph(scene_cfg, base_path=scene_dir_p))

            # Safety detector (mirrors run_vla_episode's wiring).
            from falsify.cli.smoke_test import _build_detector_factory
            safety_cfg = load_yaml(card.safety_yaml)
            detector_factory = _build_detector_factory(scene_cfg, scene_dir_p)

            # Recovery planner. Three resolution paths driven by
            # --no-recovery / --force-recovery / (default):
            #   off    → never wire recovery, even if card has it
            #   force  → use card's recovery if present, else
            #            --recovery-yaml-default
            #   auto   → use card's recovery if present, else skip
            recovery_factory = None
            recovery_triggers = None
            recovery_cfg: dict = {}
            if args.recovery_mode == "off":
                effective_recovery_yaml = None
            elif args.recovery_mode == "force":
                effective_recovery_yaml = (
                    card.recovery_yaml or args.recovery_yaml_default
                )
                if effective_recovery_yaml is None:
                    raise SystemExit(
                        "--force-recovery requires either the trial card to "
                        "declare a `recovery:` YAML, or "
                        "--recovery-yaml-default to be set."
                    )
            else:
                effective_recovery_yaml = card.recovery_yaml

            if effective_recovery_yaml is not None:
                from falsify.recovery import CoursedMpcPlanner
                from falsify.safety import FailureType
                recovery_cfg = load_yaml(effective_recovery_yaml)
                course_path = recovery_cfg.get("course")
                if course_path is None:
                    raise SystemExit(
                        f"recovery YAML {effective_recovery_yaml} must set `course:`"
                    )
                if not Path(course_path).is_absolute():
                    course_path = REPO_ROOT / course_path
                planner_kind = recovery_cfg.get("planner", "mpc")
                triggers_raw = recovery_cfg.get(
                    "trigger_failure_types",
                    ["MISS_GATE", "COLLISION_GATE", "COLLISION_OTHER", "OUT_OF_BOUNDS"],
                )
                recovery_triggers = {FailureType[name] for name in triggers_raw}

                # If the trial card declares a gate perturbation, hand
                # the equivalent gate_deltas + scene_cfg to the planner
                # so course waypoints inside the gate AABB are
                # rigid-transformed onto the perturbed gate. Without
                # this the MPC plans through the un-perturbed gate.
                gp_for_recovery = None
                if card.gate_perturbation is not None:
                    region = scene_cfg.get("gate_region") or {}
                    if region:
                        gp_for_recovery = {
                            "anchor_mocap": list(region["anchor"]),
                            "delta_xyz_mocap": list(card.gate_perturbation["delta_xyz"]),
                            "delta_yaw_rad": float(card.gate_perturbation["delta_yaw_rad"]),
                        }

                def recovery_factory(fg2, _episode_recovery_cfg):
                    return CoursedMpcPlanner(
                        course_path=course_path,
                        frame_graph=fg2,
                        planner=planner_kind,
                        prompt=card.prompt,
                        gate_deltas=gp_for_recovery,
                        scene_cfg=scene_cfg,
                    )

            episode_cfg = {
                "hz": effective_hz,
                "horizon_s": args.horizon_s,
                "chunk_steps": effective_chunk,
                "safety": safety_cfg,
            }
            ec = EpisodeConfig(
                scene_cfg=scene_cfg, frame_cfg=frame_cfg,
                episode_cfg=episode_cfg, scene_cfg_dir=scene_dir_p,
            )

            # Absolute start state (NED).
            start_state = DroneState(
                pos=Point(xyz=card.start_ned, frame=fg.frame("ned")),
                vel=np.zeros(3),
                quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
                t=0.0,
            )

            # Perturbation suite + override (one or zero gate perturbations).
            suite_factory = None
            override = None
            if card.gate_perturbation is not None:
                # Build the suite inline so the orchestrator's factory call
                # just returns it. (Factory signature is
                # `(frame_graph, episode_cfg) -> PerturbationSuite`.)
                gp = card.gate_perturbation
                def suite_factory(_fg, _ec):
                    return _build_perturbation_suite(scenario, scene_cfg, gp)
                override = {"gate_rigid_perturbation": {
                    "delta_xyz": gp["delta_xyz"],
                    "delta_yaw_rad": gp["delta_yaw_rad"],
                }}

            print(f"[run] {scene_key}/trial_{card.trial_index:03d}: "
                  f"start_ned={card.start_ned.round(3).tolist()} "
                  f"gate_pert={'yes' if override else 'none'}")
            t_trial = time.time()
            try:
                episode = run_episode(
                    ec,
                    policy_factory=policy_factory,
                    renderer=renderer,
                    detector_factory=detector_factory,
                    recovery_factory=recovery_factory,
                    recovery_triggers=recovery_triggers,
                    perturbations_factory=suite_factory,
                    initial_state_override=start_state,
                    perturbation_overrides=override,
                )
            except Exception as e:  # noqa: BLE001
                tb = traceback.format_exc()
                print(f"[error] {scene_key}/trial_{card.trial_index:03d}: {e}")
                (trial_dir / "error.txt").write_text(tb)
                aggregate.append({
                    "scene_key": scene_key,
                    "trial_index": card.trial_index,
                    "error": str(e),
                })
                continue
            dt = time.time() - t_trial

            # ---- Persist the rollout trace so post-hoc scoring (CEM cost
            # functions, OOD analysis, etc.) can recompute any per-step
            # metric without re-running the episode. Cheap (~kB / trial).
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

            summary = {
                "scenario": scenario_name,
                "scene": str(scene_yaml),
                "scene_key": scene_key,
                "trial_index": card.trial_index,
                "prompt": card.prompt,
                "start_ned": card.start_ned.tolist(),
                "gate_perturbation": card.gate_perturbation,
                "hz": effective_hz,
                "actions_per_chunk": effective_chunk,
                "horizon_s": args.horizon_s,
                "policy_config": str(args.policy_config),
                "policy_traceability": pgcfg.traceability,
                "n_states": len(episode.trace.states),
                "n_chunks": len(episode.trace.policy_outputs),
                "failure": (None if episode.failure is None else {
                    "step": episode.failure.failure_step,
                    "type": episode.failure.failure_type.name,
                    "criterion": episode.failure.criterion_name,
                    "description": episode.failure.description,
                }),
                "goal_ned": episode.goal.xyz.tolist() if episode.goal is not None else None,
                "vla_io_dir": str(record_dir),
                "rollout_states_npz": str(rollout_npz_path),
                "perturbations_manifest": episode.metadata.get("perturbations"),
                "elapsed_s": float(dt),
            }

            # ---- Post-hoc classification --------------------------------
            # The runtime stack (MissGateCriterion in eval_stop_mode) only
            # stops the rollout; it does NOT decide SUCCESS vs MISS_GATE.
            # We do that here by walking the trial's positions in MOCAP
            # against the scene's gate_region AABB (with gate-perturbation
            # Δ applied so the AABB tracks the moved Gaussians).
            from falsify.safety.posthoc import classify_trajectory_posthoc
            mocap_frame = fg.frame("mocap")
            positions_mocap = np.asarray(
                fg.convert(traj, to="mocap").positions, dtype=np.float64,
            )
            horizon_steps = int(round(args.horizon_s * effective_hz))
            # Directional gate-transit enforcement: trial cards carrying
            # an explicit `*_from_left` / `*_from_right` scene_key
            # constrain the expected crossing direction through the gate
            # aperture. from_left ⇒ correct crossing is in -y (mocap);
            # from_right ⇒ +y. Any wrong-direction aperture crossing
            # demotes the outcome to MISS_GATE in posthoc. Other
            # scene_keys leave this unset and use the legacy
            # "any-AABB-touch counts" rule.
            expected_dy_sign = None
            if card.scene_key.endswith("_from_left"):
                expected_dy_sign = -1
            elif card.scene_key.endswith("_from_right"):
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
            summary["posthoc_outcome"]    = posthoc["outcome"]
            summary["transited"]          = posthoc["transited"]
            summary["transit_first_step"] = posthoc["first_inside_step"]
            summary["transit_last_step"]  = posthoc["last_inside_step"]
            summary["gate_aabb_mocap"]    = posthoc["aabb_mocap"]
            if expected_dy_sign is not None:
                summary["expected_dy_sign"]   = expected_dy_sign
                summary["correct_crossings"]  = posthoc.get("correct_crossings")
                summary["wrong_crossings"]    = posthoc.get("wrong_crossings")
                summary["gate_plane_y_mocap"] = posthoc.get("gate_plane_y_mocap")
            # Compositional phase ({pre_gate_1, between_gates,
            # post_gate_2}) — three sources, in priority order:
            #   1. OrderedMissGateCriterion stamps `phase` onto its
            #      Violation.extra (gate-1/gate-2 stuck or
            #      goal-reached). Detector merges into
            #      FailureRecord.extra → episode.failure.extra.
            #   2. Post-hoc derives the phase from the trajectory's
            #      AABB-latch replay against `scene_cfg.gate_regions`.
            #      Covers collisions / OOB / sim instabilities for which
            #      the runtime criterion doesn't know about gates.
            #   3. None if the scene isn't compositional (no
            #      gate_regions).
            phase = None
            if summary.get("failure"):
                phase = (episode.failure.extra or {}).get("phase")
            if phase is None:
                phase = posthoc.get("phase")
            if phase is None and posthoc["outcome"] == "SUCCESS":
                phase = "post_gate_2"
            summary["phase"]              = phase

            # ---- Recovery trajectory (falsification pipeline) -----------
            # We only persist recovery NPZs for trials that actually failed
            # and replanned — the user explicitly opted out of saving any
            # heavy artifacts for successful trials.
            if episode.recovery_trajectory is not None:
                from falsify.training import save_trajectory
                from falsify.training.trajectory import Trajectory as TrainingTrajectory
                rt = episode.recovery_trajectory
                quats = (rt.quaternions if rt.quaternions is not None
                         else np.tile(np.array([0., 0., 0., 1.]),
                                      (len(rt.positions), 1)))
                npz_path = trial_dir / "recovery_trajectory.npz"
                save_trajectory(
                    npz_path,
                    TrainingTrajectory(
                        times=rt.times,
                        positions_ned=rt.positions,
                        quaternions_xyzw=quats,
                        prompt=card.prompt,
                        source="recovery",
                    ),
                )
                seed_info = (episode.metadata or {}).get("recovery_seed") or {}
                summary["recovery"] = {
                    "course": str(course_path),
                    "planner": planner_kind,
                    "triggers": sorted(t.name for t in recovery_triggers),
                    "fired": True,
                    "trajectory_npz": str(npz_path),
                    "n_states": int(len(rt.positions)),
                    "duration_s": float(rt.times[-1] - rt.times[0]),
                    "seed_step": seed_info.get("step"),
                    "seed_bias": seed_info.get("bias"),
                    "n_safe_states": seed_info.get("n_safe"),
                }
            elif effective_recovery_yaml is not None:
                summary["recovery"] = {
                    "course": str(course_path),
                    "planner": planner_kind,
                    "triggers": sorted(t.name for t in recovery_triggers),
                    "fired": False,
                    "reason": (
                        "no failure" if episode.failure is None
                        else f"failure type {episode.failure.failure_type.name} not in triggers"
                    ),
                }
            elif args.recovery_mode == "off" and card.recovery_yaml is not None:
                # Card has recovery wired but --no-recovery overrode it.
                summary["recovery"] = {
                    "fired": False,
                    "reason": "disabled by --no-recovery",
                }
            # ---- Forward-cam GIF (first N trials per scene only) --------
            gif_path = None
            if (fwd_sensor is not None
                    and gifs_rendered_this_scene < args.gif_trials_per_scene
                    and episode.trace.states):
                from falsify.cli.run_vla_episode import _render_flythrough
                from falsify.sim.poses import camera_to_world_pose as _c2w
                gif_path = trial_dir / "flythrough_forward.gif"
                _render_flythrough(
                    episode.trace.states, renderer, fwd_sensor.spec, _c2w,
                    gif_path,
                    fps=args.gif_fps, every=args.gif_every,
                )
                summary["flythrough_gif"] = str(gif_path)
                gifs_rendered_this_scene += 1
            summary_path.write_text(json.dumps(summary, indent=2))
            aggregate.append(summary)
            if summary.get("recovery", {}).get("fired"):
                recovery_tag = "recovery=fired"
            elif args.recovery_mode == "off" and card.recovery_yaml is not None:
                recovery_tag = "recovery=off"
            elif effective_recovery_yaml is None:
                recovery_tag = "recovery=skip"
            else:
                recovery_tag = "recovery=miss"
            gif_tag = " gif=yes" if gif_path else ""
            transit_tag = " transit" if summary.get("transited") else ""
            print(f"  -> n_states={summary['n_states']}  "
                  f"outcome={summary.get('posthoc_outcome', 'UNKNOWN')}  "
                  f"stop={summary['failure']['type'] if summary['failure'] else 'HORIZON'}  "
                  f"{recovery_tag}{gif_tag}{transit_tag}  elapsed={dt:.1f}s")

    # ---- Aggregate campaign summary -----------------------------------
    n_recovery_fired = sum(
        1 for r in aggregate
        if (r.get("recovery") or {}).get("fired") is True
    )
    # Source of truth for "did this trial succeed?" is the post-hoc
    # classifier. The runtime `failure` field is just the stop signal.
    def _outcome_of(r: dict) -> str:
        if "error" in r:
            return "ERROR"
        o = r.get("posthoc_outcome")
        if o is not None:
            return o
        # Pre-eval_stop_mode trials: fall back to runtime failure_type or NONE.
        return "NONE" if r.get("failure") is None else r["failure"]["type"]

    cs = {
        "scenario": scenario_name,
        "policy_config": str(args.policy_config),
        "bundle_dir": str(bundle_dir),
        "n_trials_total": len(aggregate),
        "n_succeeded": sum(1 for r in aggregate if _outcome_of(r) == "SUCCESS"),
        "n_recovery_fired": n_recovery_fired,
        "recovery_npzs": [
            r["recovery"]["trajectory_npz"]
            for r in aggregate
            if (r.get("recovery") or {}).get("fired") is True
        ],
        "by_outcome": {},          # post-hoc histogram (authoritative)
        "by_failure_type": {},     # runtime stop-signal histogram (diagnostic)
        "by_phase": {},            # compositional: where in the two-gate sequence the trial ended
        "by_outcome_phase": {},    # cross-tab: outcome × phase (compositional diagnostic)
        "by_scene": {},
        "elapsed_total_s": float(time.time() - t_total),
        "trials": aggregate,
    }
    for r in aggregate:
        outcome = _outcome_of(r)
        cs["by_outcome"][outcome] = cs["by_outcome"].get(outcome, 0) + 1
        ftype = ("ERROR" if "error" in r
                 else ("NONE" if r.get("failure") is None
                       else r["failure"]["type"]))
        cs["by_failure_type"][ftype] = cs["by_failure_type"].get(ftype, 0) + 1
        # Phase (compositional) — None for single-gate scenes.
        phase = r.get("phase") or "unknown"
        cs["by_phase"][phase] = cs["by_phase"].get(phase, 0) + 1
        key = f"{outcome}@{phase}"
        cs["by_outcome_phase"][key] = cs["by_outcome_phase"].get(key, 0) + 1
        sk = r.get("scene_key", "_unknown")
        cs["by_scene"].setdefault(sk, {"n": 0, "succeeded": 0})
        cs["by_scene"][sk]["n"] += 1
        if outcome == "SUCCESS":
            cs["by_scene"][sk]["succeeded"] += 1

    (args.out / "campaign_summary.json").write_text(json.dumps(cs, indent=2))
    print(f"\n[campaign] done: {cs['n_succeeded']}/{cs['n_trials_total']} succeeded; "
          f"by_outcome={cs['by_outcome']}; "
          f"elapsed={cs['elapsed_total_s']:.0f}s")
    return 0


def _load_yaml_lite(path: Path) -> dict:
    """Tiny load_yaml that avoids pulling falsify deps before _smoke_imports
    runs."""
    import yaml
    with path.open() as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    raise SystemExit(main())
