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
    ap.add_argument("--horizon-s", type=float, default=30.0)
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

    policy_cfg_yaml = load_yaml(_resolve(args.policy_config))
    if policy_cfg_yaml.get("type") != "pi_gateway":
        raise SystemExit("only pi_gateway policy configs are supported by "
                         "run_eval_campaign.py (see run_vla_episode.py for the "
                         "openpi path).")
    frame_cfg = load_yaml(_resolve(args.frame))

    args.out.mkdir(parents=True, exist_ok=True)

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
                execute_chunk_size=int(policy_cfg_yaml.get("execute_chunk_size", 25)),
                prompt=card.prompt,
                hz=int(policy_cfg_yaml.get("hz", 30)),
                state_dim=int(policy_cfg_yaml.get("state_dim", 7)),
                action_dim=int(policy_cfg_yaml.get("action_dim", 7)),
                action_pos_slice=tuple(policy_cfg_yaml.get("action_pos_slice", (0, 3))),
                action_yaw_index=policy_cfg_yaml.get("action_yaw_index", 3),
                camera_map=dict(policy_cfg_yaml.get("camera_map") or {}),
                state_key=policy_cfg_yaml.get("state_key", "observation/state"),
                server_frame=policy_cfg_yaml.get("server_frame", "mocap"),
                use_rtc=bool(policy_cfg_yaml.get("use_rtc", False)),
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
                "perturbations_manifest": episode.metadata.get("perturbations"),
                "elapsed_s": float(dt),
            }
            summary_path.write_text(json.dumps(summary, indent=2))
            aggregate.append(summary)
            print(f"  -> n_states={summary['n_states']}  "
                  f"failure={summary['failure']['type'] if summary['failure'] else 'NONE'}  "
                  f"elapsed={dt:.1f}s")

    # ---- Aggregate campaign summary -----------------------------------
    cs = {
        "scenario": scenario_name,
        "policy_config": str(args.policy_config),
        "bundle_dir": str(bundle_dir),
        "n_trials_total": len(aggregate),
        "n_succeeded": sum(1 for r in aggregate if r.get("failure") is None and "error" not in r),
        "by_failure_type": {},
        "by_scene": {},
        "elapsed_total_s": float(time.time() - t_total),
        "trials": aggregate,
    }
    for r in aggregate:
        ftype = "NONE" if r.get("failure") is None else r["failure"]["type"]
        if "error" in r:
            ftype = "ERROR"
        cs["by_failure_type"][ftype] = cs["by_failure_type"].get(ftype, 0) + 1
        sk = r["scene_key"]
        cs["by_scene"].setdefault(sk, {"n": 0, "succeeded": 0})
        cs["by_scene"][sk]["n"] += 1
        if ftype == "NONE":
            cs["by_scene"][sk]["succeeded"] += 1

    (args.out / "campaign_summary.json").write_text(json.dumps(cs, indent=2))
    print(f"\n[campaign] done: {cs['n_succeeded']}/{cs['n_trials_total']} succeeded; "
          f"by_failure_type={cs['by_failure_type']}; "
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
