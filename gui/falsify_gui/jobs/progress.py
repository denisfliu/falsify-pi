"""Filesystem/log-based progress readers for long-running job types.

Each reader returns a dict: {"mode": "fraction"|"none", "done", "total",
"detail"?, "out_dir"?}. "out_dir" is included when the reader discovered an
auto-generated output directory from the job's log (the manager persists it).
All reads are cheap polls — no subprocess communication.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..paths import REPO_ROOT
from ..services import configs_enum
from .models import Job

_OUT_LINE = re.compile(r"^\[(?:campaign|collect)\] out=(.+)$", re.M)


def _log_head(job: Job, n: int = 32768) -> str:
    try:
        with open(job.log_path, "rb") as f:
            return f.read(n).decode(errors="replace")
    except OSError:
        return ""


def _discover_out_dir(job: Job) -> str | None:
    if job.out_dir:
        return job.out_dir
    m = _OUT_LINE.search(_log_head(job))
    if not m:
        return None
    p = Path(m.group(1).strip())
    if p.is_absolute():
        try:
            p = p.relative_to(REPO_ROOT)
        except ValueError:
            pass
    return str(p)


def eval_campaign(job: Job) -> dict:
    out = {"mode": "fraction", "done": 0, "total": 0}
    out_dir = _discover_out_dir(job)
    if out_dir and out_dir != job.out_dir:
        out["out_dir"] = out_dir

    scenario = Path(job.form_args.get("scenario", "")).stem
    bundle = next((b for b in configs_enum.get_bundles()
                   if b["scenario"] == scenario), None)
    if bundle:
        scenes = job.form_args.get("scenes") or list(bundle["cards_per_scene"])
        trials = job.form_args.get("trials")
        if trials:
            out["total"] = len(str(trials).split()) * len(scenes)
        else:
            out["total"] = sum(bundle["cards_per_scene"].get(s, 0) for s in scenes)

    if out_dir:
        root = REPO_ROOT / out_dir
        out["done"] = len(list(root.glob("*/trial_*/episode_summary.json")))
        if (root / "campaign_summary.json").exists():
            out["detail"] = "campaign_summary written"
    return out


def recovery_collect(job: Job) -> dict:
    out = {"mode": "fraction", "done": 0,
           "total": int(job.form_args.get("n_recoveries") or 50)}
    out_dir = _discover_out_dir(job)
    if out_dir and out_dir != job.out_dir:
        out["out_dir"] = out_dir
    if out_dir:
        root = REPO_ROOT / out_dir
        out["done"] = len(list((root / "recoveries").glob("recovery_*.npz")))
        n_trials = len(list(root.glob("*/trial_*/episode_summary.json")))
        if n_trials:
            out["detail"] = f"{n_trials} trials attempted"
    return out


def export_training_data(job: Job) -> dict:
    if not job.out_dir:
        return {"mode": "none"}
    root = REPO_ROOT / job.out_dir
    n = len(list(root.rglob("*.parquet"))) if root.exists() else 0
    return {"mode": "none", "detail": f"{n} parquet(s) written"}
