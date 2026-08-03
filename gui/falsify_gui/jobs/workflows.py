"""Hardcoded workflow presets — multi-job pipelines for the sequences we run
constantly. A preset expands a small form into a fully-concrete ordered list
of job steps (every path computed up front via conventions.py); the manager
submits the list as one chained group through the regular GPU queue, so no
DAG machinery is needed — the GPU serializes everything anyway.

Step dict: {"type": <job type>, "args": {...}, "always": bool}
- steps run strictly in order, each launching when its predecessor finishes
- "always": launch this step even if the predecessor FAILED (sweep cells are
  independent; a DAgger build halts instead). A killed job always halts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .. import conventions as conv
from ..services import configs_enum


@dataclass
class WorkflowPreset:
    name: str
    label: str
    description: str
    fields: list[dict]
    expand: Callable[[dict], tuple[str, list[dict]]]   # -> (group_label, steps)

    def schema(self) -> dict:
        return {"name": self.name, "label": self.label,
                "description": self.description, "fields": self.fields}


def _f(name: str, label: str, kind: str, **kw) -> dict:
    return {"name": name, "label": label, "kind": kind, **kw}


def _require(args: dict, *names: str) -> None:
    missing = [n for n in names if not args.get(n)]
    if missing:
        raise ValueError(f"missing required fields: {missing}")


# ----------------------------------------------------------------- eval sweep

def _expand_eval_sweep(args: dict) -> tuple[str, list[dict]]:
    _require(args, "policies", "scenarios")
    policies = args["policies"]
    scenarios = args["scenarios"]
    tag = (args.get("tag") or "sweep").replace(" ", "_")
    sweep_id = f"gui-sweep-{conv.ts()}-{tag}"

    steps: list[dict] = []
    # bundles first, only where missing (fast, idempotent)
    bundles = {b["scenario"]: b for b in configs_enum.get_bundles()}
    for scenario in scenarios:
        b = bundles.get(conv.stem(scenario))
        if b is None or not b["exists"]:
            steps.append({"type": "generate_eval_bundles",
                          "args": {"scenario": scenario}, "always": False})

    # one campaign per (policy × scenario) cell; cells are independent, so
    # each runs even if an earlier cell failed
    grid_specs: list[str] = []
    for policy in policies:
        for scenario in scenarios:
            out = conv.sweep_campaign_dir(policy, scenario, sweep_id)
            cell_args = {
                "scenario": scenario, "policy_config": policy,
                "frame": args.get("frame") or conv.DEFAULT_FRAME,
                "no_rtc": True,                      # eval determinism rule
                "skip_flythrough": bool(args.get("skip_flythrough", True)),
                "out": out,
            }
            if args.get("scenes"):
                cell_args["scenes"] = str(args["scenes"]).split()
            if args.get("trials"):
                cell_args["trials"] = args["trials"]
            steps.append({"type": "eval_campaign", "args": cell_args,
                          "always": True})
            grid_specs.append(
                f"{conv.policy_id(policy)}·{conv.stem(scenario)}:{out}")

    steps.append({"type": "campaign_grid",
                  "args": {"campaigns": grid_specs,
                           "out": conv.sweep_report_path(sweep_id),
                           "title": sweep_id},
                  "always": True})
    return sweep_id, steps


# --------------------------------------------------------------- DAgger build

def _expand_dagger_build(args: dict) -> tuple[str, list[dict]]:
    _require(args, "policy_config", "scenes", "dataset_name")
    policy = args["policy_config"]
    scenes = args["scenes"]
    name = args["dataset_name"].replace(" ", "_")
    build_id = f"gui-dagger-{conv.ts()}-{name}"
    frame = args.get("frame") or conv.DEFAULT_FRAME
    embodiment = args.get("embodiment") or conv.DEFAULT_EMBODIMENT
    n_rec = int(args.get("n_recoveries") or 50)

    steps: list[dict] = []
    task_specs: list[str] = []
    for i, scene in enumerate(scenes):
        safety = conv.safety_for_scene(scene)
        recovery = conv.recovery_for_scene(scene)
        prompt = conv.prompt_for_scene(scene)
        if not (safety and recovery and prompt):
            raise ValueError(
                f"scene {conv.stem(scene)} has no matching "
                f"safety/recovery/prompt by stem convention "
                f"(safety={safety}, recovery={recovery}, prompt={prompt})")
        collect_out = conv.collection_dir(policy, scene, build_id)
        render_out = conv.dagger_render_dir(build_id, i, scene)
        steps.append({"type": "recovery_collect", "args": {
            "policy_config": policy, "scene": scene,
            "safety": safety, "recovery": recovery, "frame": frame,
            "prompt_name": prompt,
            "n_recoveries": n_rec,
            "max_trials": int(args.get("max_trials") or 500),
            "perturbation_recipe": args.get("perturbation_recipe")
                or "configs/eval_suite/gate_perturbed_large.yaml",
            "no_rtc": True,
            "out": collect_out,
        }, "always": False})
        steps.append({"type": "render_recoveries", "args": {
            "recovery_run_dir": collect_out,
            "scene": scene, "frame": frame, "embodiment": embodiment,
            "out": render_out, "task_index": i,
        }, "always": False})
        # combine assigns task strings to episode ranges in sorted-source
        # order; the staging dirs are index-prefixed so this lines up. The
        # last scene uses 'rest' so a short harvest can't misalign labels.
        task = configs_enum.prompt_task(prompt)
        if i < len(scenes) - 1:
            task_specs.append(f"$nparquets({render_out}):{task}")
        else:
            task_specs.append(f"rest:{task}")

    steps.append({"type": "combine_lerobot", "args": {
        "src": conv.dagger_staging_dir(build_id),
        "out": conv.dataset_dir(name),
        "tasks": ";".join(task_specs),
        "overwrite": bool(args.get("overwrite", False)),
    }, "always": False})
    return build_id, steps


WORKFLOWS: dict[str, WorkflowPreset] = {}


def register(wf: WorkflowPreset) -> None:
    WORKFLOWS[wf.name] = wf


register(WorkflowPreset(
    name="eval_sweep",
    label="Eval sweep (policies × scenarios)",
    description="Queue one eval campaign per (policy × scenario) cell — "
                "bundles auto-generated where missing — then emit a faceted "
                "comparison grid HTML. Cells are independent: one failed "
                "campaign doesn't stop the rest. Stack it and walk away; "
                "the GPU queue serializes everything and survives GUI "
                "restarts.",
    fields=[
        _f("policies", "Policies", "multiselect", source="policies",
           required=True),
        _f("scenarios", "Scenarios", "multiselect", source="eval_suite",
           required=True),
        _f("scenes", "Scene-key filter (space-separated, optional)", "text"),
        _f("trials", "Trial indices (optional, e.g. “0 1”)", "text"),
        _f("frame", "Drone frame", "select", source="frames",
           default="configs/frames/carl_dual.yaml"),
        _f("skip_flythrough", "Skip flythrough MP4s", "checkbox",
           default=True),
        _f("tag", "Sweep tag", "text", help="human label in the sweep id"),
    ],
    expand=_expand_eval_sweep,
))

register(WorkflowPreset(
    name="dagger_build",
    label="DAgger data build (collect → render → combine)",
    description="For each scene: collect N recovery trajectories under "
                "perturbation, render them perturbation-aware into a staged "
                "per-scene dataset, then combine everything into one LeRobot "
                "dataset under data/atomic_datasets/<name>. Safety/recovery/"
                "prompt are matched per scene by stem convention; task "
                "strings come from the prompt registry; episode counts are "
                "resolved from the actual harvest at combine time. Halts on "
                "the first failed step (unlike a sweep).",
    fields=[
        _f("policy_config", "Policy", "select", source="policies",
           required=True),
        _f("scenes", "Scenes", "multiselect", source="scenes", required=True,
           help="each needs safety/<stem>.yaml, recovery/<stem>_mpc.yaml, "
                "and a registry prompt named <stem>"),
        _f("dataset_name", "Output dataset name", "text", required=True,
           help="→ data/atomic_datasets/<name>"),
        _f("n_recoveries", "Recoveries per scene", "number", default=50),
        _f("max_trials", "Max trials per scene", "number", default=500),
        _f("perturbation_recipe", "Perturbation recipe", "select",
           source="eval_suite",
           default="configs/eval_suite/gate_perturbed_large.yaml"),
        _f("embodiment", "Embodiment", "select", source="embodiments",
           default="configs/embodiments/carl_dual_mocap.yaml"),
        _f("frame", "Drone frame", "select", source="frames",
           default="configs/frames/carl_dual.yaml"),
        _f("overwrite", "Overwrite output dataset", "checkbox"),
    ],
    expand=_expand_dagger_build,
))
