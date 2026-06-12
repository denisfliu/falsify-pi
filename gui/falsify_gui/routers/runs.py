from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse

from ..paths import PathOutsideRoot
from ..services import artifacts, npz_plot

router = APIRouter()


def _guard(fn, *args):
    try:
        return fn(*args)
    except PathOutsideRoot:
        raise HTTPException(403, "path escapes runs/")
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@router.get("/runs/campaigns")
def campaigns() -> list[dict]:
    return artifacts.campaign_index()


@router.get("/runs/recovery")
def recovery() -> list[dict]:
    return artifacts.recovery_index()


@router.get("/runs/campaign")
def campaign(path: str) -> dict:
    return _guard(artifacts.campaign_detail, path)


@router.get("/runs/trial")
def trial(path: str) -> dict:
    return _guard(artifacts.trial_detail, path)


@router.get("/runs/tree")
def tree(path: str = "runs") -> list[dict]:
    return _guard(artifacts.list_tree, path)


@router.get("/plot/npz")
def plot_npz(path: str):
    name = _guard(npz_plot.plot_npz, path)
    return RedirectResponse(f"/gui-cache/plots/{name}")
