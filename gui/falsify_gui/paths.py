"""Canonical paths + safe path resolution for the GUI server."""
from __future__ import annotations

from pathlib import Path

GUI_ROOT = Path(__file__).resolve().parent.parent          # gui/
REPO_ROOT = GUI_ROOT.parent                                 # falsify repo root
RUNS_DIR = REPO_ROOT / "runs"
CONFIGS_DIR = REPO_ROOT / "configs"
FALSIFY_PY = REPO_ROOT / ".venv" / "bin" / "python"

DATA_DIR = GUI_ROOT / "data"
JOBS_DIR = DATA_DIR / "jobs"
CACHE_DIR = DATA_DIR / "cache"
PLOTS_CACHE = CACHE_DIR / "plots"
INSPECT_CACHE = CACHE_DIR / "inspect"
DB_PATH = DATA_DIR / "jobs.db"

STATIC_DIR = Path(__file__).resolve().parent / "static"


def ensure_dirs() -> None:
    for d in (DATA_DIR, JOBS_DIR, PLOTS_CACHE, INSPECT_CACHE):
        d.mkdir(parents=True, exist_ok=True)


class PathOutsideRoot(Exception):
    pass


def resolve_runs_path(rel: str) -> Path:
    """Resolve a client-supplied path (relative to the repo root, e.g.
    'runs/eval_campaigns/...') and require it to land inside runs/."""
    p = (REPO_ROOT / rel).resolve()
    if not p.is_relative_to(RUNS_DIR.resolve()):
        raise PathOutsideRoot(rel)
    return p


def runs_rel(p: Path) -> str:
    """Repo-root-relative string for a path under runs/ (for client use)."""
    return str(p.resolve().relative_to(REPO_ROOT.resolve()))
