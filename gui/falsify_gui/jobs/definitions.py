"""Declarative job-type table.

Each JobType describes: the form the frontend renders (fields), how to turn
submitted args into a command line (build), and how to read progress off the
filesystem while the job runs (progress). Adding a workflow to the GUI means
adding an entry here — nothing else.

Form field `kind`s: select | multiselect | text | number | checkbox.
Select options come from `source` (a key in /api/configs) or static `options`.
Field values for config selects are repo-relative paths.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Callable, Optional

from ..paths import REPO_ROOT
from ..services import configs_enum
from . import progress as progress_mod
from .models import Job


@dataclass
class Built:
    script_args: list[str]          # passed to the falsify venv python
    out_dir: str | None = None      # repo-relative primary artifact path
    url: str | None = None          # services only
    label: str = ""


@dataclass
class JobType:
    name: str
    label: str
    kind: str                       # "job" | "service"
    gpu: bool
    fields: list[dict]
    build: Callable[[dict], Built]
    progress: Optional[Callable[[Job], dict]] = None
    # status inference for jobs whose exit code was lost (GUI restart):
    # return "succeeded"/"failed", or None to mark "orphaned"
    finalize: Optional[Callable[[Job], Optional[str]]] = None
    # log line printed as the script's last statement before returning.
    # Some scripts (eval campaign, recovery collect) leave non-daemon
    # gateway-client threads behind and never exit on their own; once this
    # marker appears the reaper may reap the lingering process as succeeded.
    done_marker: Optional[str] = None
    description: str = ""

    def schema(self) -> dict:
        return {"name": self.name, "label": self.label, "kind": self.kind,
                "gpu": self.gpu, "fields": self.fields,
                "description": self.description}


def _f(name: str, label: str, kind: str, **kw) -> dict:
    return {"name": name, "label": label, "kind": kind, **kw}


def _ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _require(args: dict, *names: str) -> None:
    missing = [n for n in names if not args.get(n)]
    if missing:
        raise ValueError(f"missing required fields: {missing}")


# ---------------------------------------------------------------- job types

def _build_plan_trajectory(args: dict) -> Built:
    _require(args, "course", "scene")
    planner = args.get("planner") or "spline"
    out = f"runs/courses/{Path(args['course']).stem}-{planner}-{_ts()}.npz"
    argv = ["-m", "falsify.cli.plan_trajectory",
            "--course", args["course"], "--scene", args["scene"],
            "--out", out, "--planner", planner]
    if planner == "mpc" and args.get("mpc_frame"):
        argv += ["--mpc-frame", args["mpc_frame"]]
    if args.get("prompt_name"):
        argv += ["--prompt", configs_enum.prompt_task(args["prompt_name"])]
    return Built(argv, out_dir=out,
                 label=f"{Path(args['course']).stem} ({planner})")


def _build_generate_bundles(args: dict) -> Built:
    _require(args, "scenario")
    argv = ["scripts/eval/generate_eval_bundles.py", "--scenario", args["scenario"]]
    out = f"runs/eval_bundles/{Path(args['scenario']).stem}"
    return Built(argv, out_dir=out, label=Path(args["scenario"]).stem)


JOB_TYPES: dict[str, JobType] = {}


def register(jt: JobType) -> None:
    JOB_TYPES[jt.name] = jt


register(JobType(
    name="plan_trajectory",
    label="Plan trajectory (course → NPZ)",
    kind="job", gpu=False,
    description="Turn a waypoint Course YAML into a canonical Trajectory NPZ "
                "(NED positions + quaternions + times) without running any "
                "policy. Two planners: 'spline' fits a cubic spline through "
                "the waypoints (<1 s, kinematic only); 'mpc' rolls FiGS' "
                "VehicleRateMPC over the course for a dynamically-feasible "
                "path (~30 s acados JIT on first run, ~10–30 s after). The "
                "NPZ lands in runs/courses/ and feeds straight into "
                "training-data export or visualization. No GPU, no policy "
                "server needed.",
    fields=[
        _f("course", "Course", "select", source="courses", required=True),
        _f("scene", "Scene", "select", source="scenes", required=True),
        _f("planner", "Planner", "select",
           options=["spline", "mpc"], default="spline"),
        _f("mpc_frame", "MPC drone frame", "select", source="mpc_frames",
           show_if={"planner": "mpc"}),
        _f("prompt_name", "Prompt (registry)", "select", source="prompts",
           help="optional; embedded in the NPZ"),
    ],
    build=_build_plan_trajectory,
))

def _build_eval_campaign(args: dict) -> Built:
    _require(args, "scenario", "policy_config", "frame")
    argv = ["scripts/eval/run_eval_campaign.py",
            "--scenario", args["scenario"],
            "--policy-config", args["policy_config"],
            "--frame", args["frame"]]
    if args.get("scenes"):
        argv += ["--scenes", *args["scenes"]]
    if args.get("trials"):
        argv += ["--trials", *str(args["trials"]).split()]
    if args.get("no_rtc", True):
        argv += ["--no-rtc"]
    rec = args.get("recovery_mode") or "default"
    if rec == "no-recovery":
        argv += ["--no-recovery"]
    elif rec == "force-recovery":
        argv += ["--force-recovery"]
    if args.get("skip_flythrough"):
        argv += ["--skip-flythrough"]
    if args.get("resume"):
        argv += ["--resume"]
    if args.get("execute_chunk_size"):
        argv += ["--execute-chunk-size", str(int(args["execute_chunk_size"]))]
    if args.get("out"):
        argv += ["--out", args["out"]]
    label = f"{Path(args['policy_config']).stem} × {Path(args['scenario']).stem}"
    # out dir is auto-generated by the script when --out is omitted; the
    # progress reader discovers it from the "[campaign] out=" log line
    return Built(argv, out_dir=args.get("out"), label=label)


def _finalize_eval_campaign(job: Job) -> str | None:
    if job.out_dir and (REPO_ROOT / job.out_dir / "campaign_summary.json").exists():
        return "succeeded"
    return None


def _build_recovery_collect(args: dict) -> Built:
    _require(args, "policy_config", "scene", "safety", "recovery", "frame",
             "prompt_name")
    configs_enum.prompt_task(args["prompt_name"])  # assert registry membership
    argv = ["scripts/recovery/collect_recovery_trajectories.py",
            "--policy-config", args["policy_config"],
            "--scene", args["scene"],
            "--safety", args["safety"],
            "--recovery", args["recovery"],
            "--frame", args["frame"],
            "--prompt-name", args["prompt_name"],
            "--n-recoveries", str(int(args.get("n_recoveries") or 50)),
            "--max-trials", str(int(args.get("max_trials") or 500))]
    if args.get("perturbation_recipe"):
        argv += ["--perturbation-recipe", args["perturbation_recipe"]]
    if args.get("collection_seed"):
        argv += ["--collection-seed", str(int(args["collection_seed"]))]
    if args.get("no_rtc", True):
        argv += ["--no-rtc"]
    if args.get("execute_chunk_size"):
        argv += ["--execute-chunk-size", str(int(args["execute_chunk_size"]))]
    if args.get("out"):
        argv += ["--out", args["out"]]
    label = f"{Path(args['policy_config']).stem} / {Path(args['scene']).stem}"
    return Built(argv, out_dir=args.get("out"), label=label)


def _build_vla_episode(args: dict) -> Built:
    _require(args, "scene", "frame", "prompt_name", "policy_config")
    configs_enum.prompt_task(args["prompt_name"])  # assert registry membership
    out = f"runs/vla_episodes/{Path(args['scene']).stem}-{_ts()}"
    argv = ["-m", "falsify.cli.run_vla_episode",
            "--scene", args["scene"],
            "--frame", args["frame"],
            "--prompt-name", args["prompt_name"],
            "--policy-config", args["policy_config"],
            "--out", out]
    for cli_flag, key, cast in (("--horizon-s", "horizon_s", float),
                                ("--hz", "hz", int),
                                ("--seed", "seed", int)):
        if args.get(key) is not None and args.get(key) != "":
            argv += [cli_flag, str(cast(args[key]))]
    for cli_flag, key in (("--safety", "safety"), ("--recovery", "recovery"),
                          ("--perturbations", "perturbations")):
        if args.get(key):
            argv += [cli_flag, args[key]]
    label = f"{Path(args['policy_config']).stem} @ {Path(args['scene']).stem}"
    return Built(argv, out_dir=out, label=label)


def _finalize_vla_episode(job: Job) -> str | None:
    if job.out_dir and (REPO_ROOT / job.out_dir / "episode_summary.json").exists():
        return "succeeded"
    return None


def _build_export_training_data(args: dict) -> Built:
    _require(args, "source_path", "scene", "frame", "embodiment", "out")
    src_flag = {"trajectory": "--trajectory", "run-dir": "--run-dir",
                "trajectories-dir": "--trajectories-dir"}[args.get("source_kind") or "trajectory"]
    argv = ["-m", "falsify.cli.export_training_data",
            src_flag, args["source_path"],
            "--scene", args["scene"],
            "--frame", args["frame"],
            "--embodiment", args["embodiment"],
            "--out", args["out"]]
    for cli_flag, key in (("--episode-index", "episode_index"),
                          ("--index-offset", "index_offset"),
                          ("--task-index", "task_index"),
                          ("--chunk-steps", "chunk_steps")):
        if args.get(key) is not None and args.get(key) != "":
            argv += [cli_flag, str(int(args[key]))]
    if args.get("hz"):
        argv += ["--hz", str(float(args["hz"]))]
    if args.get("prompt_name"):
        argv += ["--prompt", configs_enum.prompt_task(args["prompt_name"])]
    return Built(argv, out_dir=args["out"],
                 label=f"export → {Path(args['out']).name}")


register(JobType(
    name="generate_eval_bundles",
    label="Generate eval bundles (trial cards)",
    kind="job", gpu=False,
    description="Pre-sample the trial cards (absolute start jitter + gate "
                "perturbation per trial) for an eval-suite scenario into "
                "runs/eval_bundles/<scenario>/. Deterministic: same scenario "
                "YAML + master_seed always produces byte-identical cards, "
                "which is what makes campaigns reproducible across runs and "
                "machines. Takes <1 s, no GPU. Required once per scenario "
                "before any eval campaign; re-run only if the scenario YAML "
                "changes.",
    fields=[
        _f("scenario", "Scenario", "select", source="eval_suite", required=True),
    ],
    build=_build_generate_bundles,
))

register(JobType(
    name="eval_campaign",
    label="Eval campaign",
    kind="job", gpu=True,
    description="The main evaluation workflow: roll out one policy on every "
                "pre-sampled trial card of a scenario (live gsplat renders → "
                "policy server queries → failure detection → posthoc SUCCESS/"
                "MISS_GATE/COLLISION classification). Needs the scenario's "
                "bundle generated and the policy's bridge reachable; the "
                "policy checkpoint is swapped in automatically via the bridge "
                "admin handshake. Writes per-trial outputs plus "
                "campaign_summary.json and the trajectories/outcome-chart "
                "HTMLs under runs/eval_campaigns/<policy_id>/. Roughly 1–5 "
                "min per trial on the GPU; use the scenes/trials filters for "
                "smoke runs. Keep --no-rtc on for reproducibility.",
    fields=[
        _f("scenario", "Scenario", "select", source="eval_suite", required=True),
        _f("policy_config", "Policy", "select", source="policies", required=True),
        _f("frame", "Drone frame", "select", source="frames",
           default="configs/frames/carl_dual.yaml", required=True),
        _f("scenes", "Scenes (default: all in bundle)", "multiselect",
           source="bundle_scenes"),
        _f("trials", "Trial indices (e.g. “0 1 2”; default: all)", "text"),
        _f("no_rtc", "--no-rtc (deterministic; keep ON for evals)", "checkbox",
           default=True),
        _f("recovery_mode", "Recovery", "select",
           options=["default", "no-recovery", "force-recovery"],
           default="default"),
        _f("skip_flythrough", "Skip flythrough MP4s (faster)", "checkbox"),
        _f("resume", "Resume into existing out dir", "checkbox"),
        _f("execute_chunk_size", "Execute chunk size override", "number"),
        _f("out", "Out dir (blank = auto under runs/eval_campaigns/)", "text"),
    ],
    build=_build_eval_campaign,
    progress=progress_mod.eval_campaign,
    finalize=_finalize_eval_campaign,
    done_marker="[campaign] artifacts →",
))

register(JobType(
    name="recovery_collect",
    label="Collect recovery trajectories",
    kind="job", gpu=True,
    description="Build corrective-maneuver training data: stream-sample gate "
                "perturbations, roll the policy until it fails, then have the "
                "MPC recovery planner fly from the last safe state to the "
                "goal — each successful recovery becomes one canonical "
                "Trajectory NPZ ('I'm off; recover' demonstration). Two "
                "phases: rollouts to collect failures, then MPC planning. "
                "Writes recoveries/recovery_NNN.npz + per-attempt diagnostics "
                "under runs/recovery_collection/<policy_id>/<scene_key>/. "
                "Long-running (often hours for 50 recoveries); progress = "
                "harvested NPZs vs target. Hand the output to 'Export "
                "training data' afterwards.",
    fields=[
        _f("policy_config", "Policy", "select", source="policies", required=True),
        _f("scene", "Scene", "select", source="scenes", required=True),
        _f("safety", "Safety", "select", source="safety", required=True,
           help="usually matches the scene"),
        _f("recovery", "Recovery planner", "select", source="recovery",
           required=True, help="usually <scene>_mpc"),
        _f("frame", "Drone frame", "select", source="frames",
           default="configs/frames/carl_dual.yaml", required=True),
        _f("prompt_name", "Prompt (registry)", "select", source="prompts",
           required=True),
        _f("perturbation_recipe", "Perturbation recipe (scenario YAML)",
           "select", source="eval_suite"),
        _f("n_recoveries", "Target # recoveries", "number", default=50),
        _f("max_trials", "Max trials", "number", default=500),
        _f("collection_seed", "Collection seed", "number"),
        _f("no_rtc", "--no-rtc", "checkbox", default=True),
        _f("execute_chunk_size", "Execute chunk size override", "number"),
        _f("out", "Out dir (blank = auto)", "text"),
    ],
    build=_build_recovery_collect,
    progress=progress_mod.recovery_collect,
    done_marker="[collect] artifacts →",
))

register(JobType(
    name="vla_episode",
    label="Single VLA episode",
    kind="job", gpu=True,
    description="One live rollout of a policy against a scene — the quickest "
                "way to eyeball how a checkpoint behaves. Renders dual-camera "
                "gsplat observations, queries the policy server chunk by "
                "chunk, and writes episode_summary.json, rollout_states.npz, "
                "a flythrough MP4, and per-query vla_io/ dumps (exact images "
                "+ state + actions sent/received) to an auto-named dir under "
                "runs/vla_episodes/. Optional safety/recovery/perturbation "
                "YAMLs wire in the failure detector and MPC recovery. A 25 s "
                "horizon takes ~5–10 min including gsplat load.",
    fields=[
        _f("scene", "Scene", "select", source="scenes", required=True),
        _f("frame", "Drone frame", "select", source="frames",
           default="configs/frames/carl_dual.yaml", required=True),
        _f("prompt_name", "Prompt (registry)", "select", source="prompts",
           required=True),
        _f("policy_config", "Policy", "select", source="policies", required=True),
        _f("horizon_s", "Horizon (s)", "number"),
        _f("hz", "Control Hz", "number"),
        _f("seed", "Seed", "number"),
        _f("safety", "Safety (optional)", "select", source="safety"),
        _f("recovery", "Recovery (optional)", "select", source="recovery"),
        _f("perturbations", "Perturbations (optional)", "select",
           source="perturbations"),
    ],
    build=_build_vla_episode,
    finalize=_finalize_vla_episode,
))

def _build_render_recoveries(args: dict) -> Built:
    _require(args, "recovery_run_dir", "scene", "frame", "embodiment", "out")
    argv = ["scripts/recovery/render_recoveries_to_dataset.py",
            "--recovery-run-dir", args["recovery_run_dir"],
            "--scene", args["scene"],
            "--frame", args["frame"],
            "--embodiment", args["embodiment"],
            "--out", args["out"]]
    for cli_flag, key in (("--episode-index-base", "episode_index_base"),
                          ("--index-offset", "index_offset"),
                          ("--task-index", "task_index")):
        if args.get(key) is not None and args.get(key) != "":
            argv += [cli_flag, str(int(args[key]))]
    return Built(argv, out_dir=args["out"],
                 label=f"render recoveries → {Path(args['out']).name}")


def _build_combine_lerobot(args: dict) -> Built:
    _require(args, "src", "out", "tasks")
    argv = ["-m", "falsify.cli.combine_lerobot",
            "--src", args["src"], "--out", args["out"]]
    for spec in str(args["tasks"]).split(";"):
        spec = spec.strip()
        if spec:
            argv += ["--task", spec]
    if args.get("drop_last_pattern"):
        argv += ["--drop-last-pattern", args["drop_last_pattern"]]
    if args.get("fps"):
        argv += ["--fps", str(int(args["fps"]))]
    if args.get("overwrite"):
        argv += ["--overwrite"]
    return Built(argv, out_dir=args["out"],
                 label=f"combine → {Path(args['out']).name}")


register(JobType(
    name="render_recoveries",
    label="Render recoveries → dataset",
    kind="job", gpu=True,
    description="Turn a recovery-collection run into training parquets: for "
                "each harvested recovery NPZ, re-apply that trial's gate "
                "perturbation to the gsplat and render the recovery flight — "
                "so the camera frames show the gate where it actually was "
                "(the generic exporter can't do this). One LeRobot episode "
                "parquet per recovery, ~50 ms/frame on the GPU. This is the "
                "step between 'Collect recovery trajectories' and 'Combine "
                "LeRobot datasets' in a DAgger round.",
    fields=[
        _f("recovery_run_dir", "Recovery run dir", "text", required=True,
           help="runs/recovery_collection/<policy>/<scene>/run-NNN-…"),
        _f("scene", "Scene", "select", source="scenes", required=True),
        _f("frame", "Drone frame", "select", source="frames",
           default="configs/frames/carl_dual.yaml", required=True),
        _f("embodiment", "Embodiment", "select", source="embodiments",
           required=True),
        _f("out", "Out dir", "text", required=True,
           help="e.g. data/atomic_datasets/_staging/dagger2_left"),
        _f("episode_index_base", "Episode index base", "number"),
        _f("index_offset", "Global index offset", "number"),
        _f("task_index", "Task index", "number"),
    ],
    build=_build_render_recoveries,
    progress=progress_mod.export_training_data,
))

register(JobType(
    name="combine_lerobot",
    label="Combine LeRobot datasets",
    kind="job", gpu=False,
    description="Merge multiple LeRobot v2.1 dataset dirs into one: renumber "
                "episode/global indices, reassign task_index per range, drop "
                "flagged trailing episodes, regenerate all meta files "
                "(info/tasks/episodes/episodes_stats). Task specs assign "
                "task strings to episode ranges in source order: "
                "'<count>:<text>' per source slice, 'rest:<text>' for the "
                "remainder — separate multiple specs with ';'. ~1.5 s per "
                "episode (image stats), CPU only. Final step of a DAgger "
                "data build, before validation/upload.",
    fields=[
        _f("src", "Source parent dir", "text", required=True,
           help="directory containing the source dataset dirs"),
        _f("out", "Output dataset dir", "text", required=True),
        _f("tasks", "Task specs (';'-separated)", "text", required=True,
           help="e.g. 50:go through the gate on the left…; rest:go through the gate on the right…"),
        _f("drop_last_pattern", "Drop-last pattern", "text",
           default="*_bad_last"),
        _f("fps", "FPS metadata", "number", default=10),
        _f("overwrite", "Overwrite output", "checkbox"),
    ],
    build=_build_combine_lerobot,
))


# ------------------------------------------------------- services & viz

def _build_ns_viewer(args: dict) -> Built:
    _require(args, "scene")
    port = int(args.get("port") or 7007)
    argv = ["-m", "falsify.cli.preview_scene_nsviewer",
            "--scene", args["scene"], "--port", str(port)]
    return Built(argv, url=f"port:{port}",
                 label=f"ns-viewer {Path(args['scene']).stem} :{port}")


def _build_paint_mask(args: dict) -> Built:
    _require(args, "scene")
    port = int(args.get("port") or 8050)
    argv = ["-m", "falsify.cli.paint_gaussian_mask",
            "--scene", args["scene"], "--port", str(port), "--host", "0.0.0.0"]
    return Built(argv, url=f"port:{port}",
                 label=f"paint-mask {Path(args['scene']).stem} :{port}")


def _build_recovery_dashboard(args: dict) -> Built:
    _require(args, "policy_id", "scenes")
    scenes = args["scenes"] if isinstance(args["scenes"], list) \
        else str(args["scenes"]).split()
    out = f"runs/recovery_collection/_gui_dashboard_{args['policy_id']}.html"
    argv = ["scripts/recovery/live_recovery_dashboard.py",
            "--policy-id", args["policy_id"], "--scenes", *scenes,
            "--out", out,
            "--refresh-seconds", str(int(args.get("refresh_seconds") or 30))]
    return Built(argv, out_dir=out,
                 label=f"recovery dashboard {args['policy_id']}")


def _build_inspect_scene(args: dict) -> Built:
    _require(args, "scene")
    name = f"inspect_{Path(args['scene']).stem}-{_ts()}.html"
    out = f"gui/data/cache/inspect/{name}"
    argv = ["-m", "falsify.cli.inspect_scene_plotly",
            "--scene", args["scene"], "--out", out]
    if args.get("course"):
        argv += ["--course", args["course"]]
    return Built(argv, out_dir=out,
                 label=f"inspect {Path(args['scene']).stem}")


def _build_author_mask(args: dict) -> Built:
    _require(args, "scene")
    name = f"mask_{Path(args['scene']).stem}-{_ts()}.html"
    out = f"gui/data/cache/inspect/{name}"
    argv = ["-m", "falsify.cli.author_gaussian_mask",
            "--scene", args["scene"], "--out", out]
    if args.get("edit_name"):
        argv += ["--edit-name", args["edit_name"]]
    return Built(argv, out_dir=out,
                 label=f"mask viz {Path(args['scene']).stem}")


register(JobType(
    name="ns_viewer",
    label="ns-viewer (live gsplat)",
    kind="service", gpu=True,
    description="Live photoreal gsplat viewer (nerfstudio viser) with the "
                "scene's scene_edits pre-applied — fly around the actual "
                "splat the policy sees, inspect gate placement and edit "
                "results. ~30 s startup (gsplat CUDA JIT + load); if it dies "
                "on the first frame, the CUDA extension cache is broken — "
                "use the mask painter (CPU) instead and see the "
                "falsify-debug-render playbook.",
    fields=[
        _f("scene", "Scene", "select", source="scenes", required=True),
        _f("port", "Port", "number", default=7007),
    ],
    build=_build_ns_viewer,
))

register(JobType(
    name="paint_mask",
    label="Gaussian mask painter (Dash)",
    kind="service", gpu=False,
    description="Interactive Dash app for authoring scene-edit masks: paint "
                "include/exclude AABBs over the scene's gaussian means with "
                "sliders and get YAML-ready bounds back. Reads means "
                "directly (CPU path), so it works even when the gsplat CUDA "
                "JIT is broken. Use it to tune rigid_transform_aabb edits "
                "before committing them to a scene YAML.",
    fields=[
        _f("scene", "Scene", "select", source="scenes", required=True),
        _f("port", "Port", "number", default=8050),
    ],
    build=_build_paint_mask,
))

register(JobType(
    name="recovery_dashboard",
    label="Live recovery dashboard",
    kind="service", gpu=False,
    description="Self-refreshing HTML dashboard that polls in-flight "
                "recovery-collection runs for one policy: per-scene trial "
                "counters, harvest rate, and outcome breakdown, updating "
                "every N seconds. Start it alongside a 'Collect recovery "
                "trajectories' job to watch the harvest without tailing "
                "logs.",
    fields=[
        _f("policy_id", "Policy id (dir under runs/recovery_collection/)",
           "text", required=True),
        _f("scenes", "Scene keys (space-separated)", "text", required=True),
        _f("refresh_seconds", "Refresh (s)", "number", default=30),
    ],
    build=_build_recovery_dashboard,
))

register(JobType(
    name="inspect_scene",
    label="Scene inspector HTML",
    kind="job", gpu=False,
    description="Interactive Plotly HTML of a scene's landmarks — gate AABB "
                "wireframe, table, plane-cut posts, implied start position — "
                "with an optional course overlay (waypoints + planned "
                "spline). The standard tool for sanity-checking waypoints "
                "against scene geometry before planning a trajectory. <1 s, "
                "CPU only; result embeds inline below.",
    fields=[
        _f("scene", "Scene", "select", source="scenes", required=True),
        _f("course", "Course overlay (optional)", "select", source="courses"),
    ],
    build=_build_inspect_scene,
))

register(JobType(
    name="author_mask",
    label="Gaussian mask classification HTML",
    kind="job", gpu=False,
    description="Read-only check of a scene_edit mask: classifies every "
                "gaussian mean against the edit's include/exclude AABBs and "
                "renders the colored 3-D point cloud (cyan=moved, "
                "red=stranded, orange=exclude-only, gray=outside). Use after "
                "editing AABBs in a scene YAML to confirm exactly which "
                "gaussians a rigid_transform_aabb will move. ~5–30 s "
                "(gsplat checkpoint load), CPU path; result embeds inline.",
    fields=[
        _f("scene", "Scene", "select", source="scenes", required=True),
        _f("edit_name", "Edit name (default: first edit)", "text"),
    ],
    build=_build_author_mask,
))

register(JobType(
    name="export_training_data",
    label="Export training data (NPZ → parquet)",
    kind="job", gpu=True,
    description="Render trajectory NPZ(s) against a scene and emit LeRobot "
                "v2.1 training parquet(s) — embedded PNG camera frames + "
                "state + action columns, schema-identical to what DroneVLA's "
                "pipeline ingests. Source can be a single NPZ (planned or "
                "recovery trajectory), a VLA run dir (replays its flown "
                "path), or a whole directory of NPZs (renderer loaded once, "
                "one parquet per NPZ). ~50 ms per frame on the GPU, so a "
                "typical episode exports in seconds. The embodiment YAML "
                "controls the state/action layout and camera mapping.",
    fields=[
        _f("source_kind", "Source kind", "select",
           options=["trajectory", "run-dir", "trajectories-dir"],
           default="trajectory", required=True),
        _f("source_path", "Source path", "text", required=True,
           help="NPZ file, VLA run dir, or directory of NPZs (repo-relative)"),
        _f("scene", "Scene", "select", source="scenes", required=True),
        _f("frame", "Drone frame", "select", source="frames",
           default="configs/frames/carl_dual.yaml", required=True),
        _f("embodiment", "Embodiment", "select", source="embodiments",
           required=True),
        _f("out", "Out dir", "text", required=True,
           help="e.g. runs/datasets/my_export"),
        _f("episode_index", "Episode index base", "number"),
        _f("index_offset", "Global index offset", "number"),
        _f("task_index", "Task index", "number"),
        _f("chunk_steps", "Chunk steps (run-dir source)", "number"),
        _f("hz", "FPS override", "number"),
        _f("prompt_name", "Prompt override (registry)", "select",
           source="prompts"),
    ],
    build=_build_export_training_data,
    progress=progress_mod.export_training_data,
))
