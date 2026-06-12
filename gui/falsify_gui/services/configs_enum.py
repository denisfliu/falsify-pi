"""Enumerate the configs/ families that populate GUI form selects.

Everything here is a cheap filesystem read; a short TTL cache keeps repeated
form renders from re-parsing YAML on every request.
"""
from __future__ import annotations

import time
from pathlib import Path

import yaml

from ..paths import CONFIGS_DIR, RUNS_DIR

# family name -> directory of *.yaml configs (value = repo-relative path)
_YAML_FAMILIES = {
    "scenes": "scenes",
    "safety": "safety",
    "recovery": "recovery",
    "courses": "courses",
    "eval_suite": "eval_suite",
    "embodiments": "embodiments",
    "perturbations": "perturbations",
    "frames": "frames",
}

_TTL_S = 2.0
_cache: dict[str, tuple[float, object]] = {}


def _cached(key: str, fn):
    now = time.monotonic()
    hit = _cache.get(key)
    if hit and now - hit[0] < _TTL_S:
        return hit[1]
    val = fn()
    _cache[key] = (now, val)
    return val


def _rel(p: Path) -> str:
    return str(p.relative_to(CONFIGS_DIR.parent))


def _list_yaml(subdir: str) -> list[dict]:
    d = CONFIGS_DIR / subdir
    out = []
    for p in sorted(d.glob("*.yaml")):
        out.append({"name": p.stem, "path": _rel(p)})
    return out


def _list_policies() -> list[dict]:
    out = []
    for p in sorted((CONFIGS_DIR / "policies").glob("*.yaml")):
        out.append({"name": p.stem, "path": _rel(p), "backend": "mock"})
    gw = CONFIGS_DIR / "policies" / "pi_gateway"
    for p in sorted(gw.glob("*.yaml")):
        entry = {"name": p.stem, "path": _rel(p), "backend": "pi_gateway"}
        try:
            cfg = yaml.safe_load(p.read_text()) or {}
            entry.update({
                "gateway_url": cfg.get("gateway_url"),
                "bridge_admin_url": cfg.get("bridge_admin_url"),
                "bridge_policy_id": cfg.get("bridge_policy_id"),
                "execute_chunk_size": cfg.get("execute_chunk_size"),
                "traceability": cfg.get("traceability") or {},
            })
        except yaml.YAMLError as e:
            entry["parse_error"] = str(e)
        out.append(entry)
    return out


def _list_prompts() -> dict[str, str]:
    """name -> task string, merged across all registry YAMLs.

    Prompts are the ONLY strings allowed in prompt fields — they must exist
    verbatim in the policy's training tasks, so the GUI never offers free
    text here.
    """
    prompts: dict[str, str] = {}
    for p in sorted((CONFIGS_DIR / "prompts").glob("*.yaml")):
        try:
            doc = yaml.safe_load(p.read_text()) or {}
        except yaml.YAMLError:
            continue
        for name, spec in (doc.get("prompts") or {}).items():
            task = spec.get("task") if isinstance(spec, dict) else None
            if task:
                prompts[name] = task
    return prompts


def _list_mpc_frames() -> list[dict]:
    d = CONFIGS_DIR / "frames" / "figs"
    return [{"name": p.stem, "path": _rel(p)} for p in sorted(d.glob("*.json"))]


def get_configs() -> dict:
    def build():
        out = {fam: _list_yaml(sub) for fam, sub in _YAML_FAMILIES.items()}
        out["policies"] = _list_policies()
        out["prompts"] = _list_prompts()
        out["mpc_frames"] = _list_mpc_frames()
        return out
    return _cached("configs", build)


def get_bundles() -> list[dict]:
    """Per eval-suite scenario: does runs/eval_bundles/<name>/ exist, and how
    many trial cards per scene_key? Drives the campaign launch precondition
    and the progress denominator."""
    def build():
        bundles_root = RUNS_DIR / "eval_bundles"
        out = []
        for scn in _list_yaml("eval_suite"):
            bdir = bundles_root / scn["name"]
            cards = {}
            if bdir.is_dir():
                for scene_dir in sorted(bdir.iterdir()):
                    if scene_dir.is_dir():
                        n = len(list(scene_dir.glob("trial_*.json")))
                        if n:
                            cards[scene_dir.name] = n
            out.append({
                "scenario": scn["name"],
                "scenario_path": scn["path"],
                "bundle_dir": str(bdir.relative_to(RUNS_DIR.parent)),
                "exists": bool(cards),
                "cards_per_scene": cards,
                "n_cards_total": sum(cards.values()),
            })
        return out
    return _cached("bundles", build)


def prompt_task(name: str) -> str:
    prompts = get_configs()["prompts"]
    if name not in prompts:
        raise KeyError(f"prompt {name!r} not in registry")
    return prompts[name]
