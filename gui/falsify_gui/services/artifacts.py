"""Read-only browsing of the runs/ artifact tree.

All entry points take repo-relative paths and resolve through
paths.resolve_runs_path, which rejects anything escaping runs/.
Listing is shallow/lazy — runs/ is large.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from ..paths import RUNS_DIR, resolve_runs_path, runs_rel

_KINDS = {".html": "html", ".png": "png", ".jpg": "image", ".mp4": "video",
          ".npz": "npz", ".json": "json", ".ply": "ply", ".parquet": "parquet",
          ".log": "log", ".yaml": "yaml", ".gif": "image"}

_TTL_S = 5.0
_cache: dict[str, tuple[float, object]] = {}


def _cached(key: str, fn):
    now = time.monotonic()
    hit = _cache.get(key)
    if hit and now - hit[0] < _TTL_S:
        return hit[1]
    val = fn()
    _cache[key] = (now, val)
    return val


def _read_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def list_tree(rel: str) -> list[dict]:
    root = resolve_runs_path(rel)
    if not root.is_dir():
        raise FileNotFoundError(rel)
    out = []
    for p in sorted(root.iterdir(), key=lambda x: (not x.is_dir(), x.name)):
        st = p.stat()
        out.append({
            "name": p.name,
            "path": runs_rel(p),
            "is_dir": p.is_dir(),
            "kind": "dir" if p.is_dir() else _KINDS.get(p.suffix.lower(), "file"),
            "size": 0 if p.is_dir() else st.st_size,
            "mtime": st.st_mtime,
        })
    return out


def campaign_index() -> list[dict]:
    """Every campaign run dir (identified by campaign_summary.json) under
    runs/eval_campaigns/, with rollup fields. Bounded-depth globs keep the
    walk away from per-trial subtrees."""
    def build():
        root = RUNS_DIR / "eval_campaigns"
        if not root.is_dir():
            return []
        found: list[dict] = []
        for depth in range(2, 6):
            for summary in root.glob("/".join(["*"] * depth) + "/campaign_summary.json"):
                run_dir = summary.parent
                doc = _read_json(summary) or {}
                rel_parts = run_dir.relative_to(root).parts
                found.append({
                    "path": runs_rel(run_dir),
                    "policy_id": rel_parts[0],
                    "name": run_dir.name,
                    "scenario": doc.get("scenario"),
                    "n_trials_total": doc.get("n_trials_total"),
                    "n_succeeded": doc.get("n_succeeded"),
                    "by_outcome": doc.get("by_outcome"),
                    "mtime": summary.stat().st_mtime,
                })
        found.sort(key=lambda d: d["mtime"], reverse=True)
        return found
    return _cached("campaigns", build)


def recovery_index() -> list[dict]:
    def build():
        root = RUNS_DIR / "recovery_collection"
        if not root.is_dir():
            return []
        found: list[dict] = []
        for depth in range(2, 5):
            for manifest in root.glob("/".join(["*"] * depth) + "/collection_manifest.json"):
                run_dir = manifest.parent
                doc = _read_json(manifest) or {}
                n_npz = len(list((run_dir / "recoveries").glob("recovery_*.npz")))
                found.append({
                    "path": runs_rel(run_dir),
                    "policy_id": doc.get("policy_id"),
                    "scene_key": doc.get("scene_key"),
                    "name": run_dir.name,
                    "n_recoveries": n_npz,
                    "mtime": manifest.stat().st_mtime,
                })
        found.sort(key=lambda d: d["mtime"], reverse=True)
        return found
    return _cached("recoveries", build)


def campaign_detail(rel: str) -> dict:
    run_dir = resolve_runs_path(rel)
    summary = _read_json(run_dir / "campaign_summary.json") or {}
    # the trials list can be huge; the per-trial scan below replaces it
    summary.pop("trials", None)
    scenes: dict[str, list[dict]] = {}
    for scene_dir in sorted(run_dir.iterdir()):
        if not scene_dir.is_dir() or scene_dir.name == "viz":
            continue
        trials = []
        for trial_dir in sorted(scene_dir.glob("trial_*")):
            es = _read_json(trial_dir / "episode_summary.json")
            if es is None:
                continue
            trials.append({
                "path": runs_rel(trial_dir),
                "name": trial_dir.name,
                "outcome": es.get("posthoc_outcome")
                           or (es.get("failure") or {}).get("type"),
                "transited": es.get("transited"),
                "recovery_fired": (es.get("recovery") or {}).get("fired"),
            })
        if trials:
            scenes[scene_dir.name] = trials
    viz = [runs_rel(p) for p in sorted((run_dir / "viz").glob("*.html"))] \
        if (run_dir / "viz").is_dir() else []
    return {
        "path": rel,
        "summary": summary,
        "run_manifest": _read_json(run_dir / "run_manifest.json"),
        "policy_manifest": _read_json(run_dir / "policy_manifest.json"),
        "scenes": scenes,
        "viz_html": viz,
        "log": runs_rel(run_dir / "campaign.log")
               if (run_dir / "campaign.log").exists() else None,
    }


def trial_detail(rel: str) -> dict:
    trial_dir = resolve_runs_path(rel)
    queries = []
    vla_io = trial_dir / "vla_io"
    if vla_io.is_dir():
        for q in sorted(vla_io.iterdir()):
            if not q.is_dir():
                continue
            queries.append({
                "name": q.name,
                "images": [runs_rel(p) for p in sorted(q.glob("*.png"))],
                "data": runs_rel(q / "data.json") if (q / "data.json").exists() else None,
            })
    mp4s = [runs_rel(p) for p in sorted(trial_dir.glob("*.mp4"))]
    htmls = [runs_rel(p) for p in sorted(trial_dir.glob("*.html"))]
    npzs = [runs_rel(p) for p in sorted(trial_dir.glob("*.npz"))]
    return {
        "path": rel,
        "episode_summary": _read_json(trial_dir / "episode_summary.json"),
        "trial_card": _read_json(trial_dir / "trial_card.json"),
        "vla_io_queries": queries,
        "mp4s": mp4s,
        "htmls": htmls,
        "npzs": npzs,
    }
