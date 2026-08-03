"""Shared naming + config-matching conventions.

This is THE module where "we always do it this way" lives — job builders and
workflow presets must use these helpers instead of re-deriving paths or
config matches, so a convention change is a one-line edit here.
"""
from __future__ import annotations

import time
from pathlib import Path

from .services import configs_enum


def ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def stem(path: str) -> str:
    return Path(path).stem


# ---------------------------------------------------------- config matching
# Scene-stem conventions: configs/safety/<scene>.yaml,
# configs/recovery/<scene>_mpc.yaml, prompt registry key == scene stem.

def _match(family: str, name: str) -> str | None:
    for entry in configs_enum.get_configs()[family]:
        if entry["name"] == name:
            return entry["path"]
    return None


def safety_for_scene(scene_path: str) -> str | None:
    return _match("safety", stem(scene_path))


def recovery_for_scene(scene_path: str) -> str | None:
    return _match("recovery", stem(scene_path) + "_mpc")


def prompt_for_scene(scene_path: str) -> str | None:
    name = stem(scene_path)
    return name if name in configs_enum.get_configs()["prompts"] else None


DEFAULT_FRAME = "configs/frames/carl_dual.yaml"
DEFAULT_EMBODIMENT = "configs/embodiments/carl_dual_mocap.yaml"


# ------------------------------------------------------------- output paths
# Mirror the existing on-disk conventions (tools/run_eval_sweep.sh and the
# recovery/dataset scripts) so GUI-produced artifacts land where everything
# else already looks for them.

def policy_id(policy_config_path: str) -> str:
    return stem(policy_config_path)


def sweep_campaign_dir(policy_config: str, scenario: str, sweep_id: str) -> str:
    """Per-cell campaign dir inside a sweep, rooted per-policy like the
    shell sweep: runs/eval_campaigns/<policy>/<sweep_id>/run-<scenario>/"""
    return (f"runs/eval_campaigns/{policy_id(policy_config)}/"
            f"{sweep_id}/run-{stem(scenario)}")


def sweep_report_path(sweep_id: str) -> str:
    return f"runs/eval_campaigns/_sweep_reports/{sweep_id}.html"


def collection_dir(policy_config: str, scene: str, build_id: str) -> str:
    return (f"runs/recovery_collection/{policy_id(policy_config)}/"
            f"{stem(scene)}/{build_id}")


def dagger_staging_dir(build_id: str) -> str:
    """Parent dir whose children are the per-scene rendered datasets;
    combine_lerobot takes this as --src. Children are index-prefixed so
    combine's sorted source order matches the scene order."""
    return f"data/atomic_datasets/_staging/{build_id}"


def dagger_render_dir(build_id: str, scene_index: int, scene: str) -> str:
    return f"{dagger_staging_dir(build_id)}/{scene_index:02d}_{stem(scene)}"


def dataset_dir(name: str) -> str:
    return f"data/atomic_datasets/{name}"
