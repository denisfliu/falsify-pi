from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import RedirectResponse

from ..services import datasets

router = APIRouter()


def _guard(fn, *args):
    try:
        return fn(*args)
    except datasets.PathOutsideData:
        raise HTTPException(403, "path escapes data/")
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except KeyError as e:
        raise HTTPException(404, f"unknown column {e}")


@router.get("/datasets")
def index() -> list[dict]:
    return datasets.dataset_index()


@router.get("/datasets/episodes")
def episodes(path: str) -> list[dict]:
    return _guard(datasets.episode_list, path)


@router.get("/datasets/episode")
def episode(path: str, index: int) -> dict:
    return _guard(datasets.episode_detail, path, index)


@router.get("/datasets/frame.png")
def frame(path: str, index: int, frame: int, camera: str):
    blob = _guard(datasets.frame_png, path, index, frame, camera)
    return Response(content=blob, media_type="image/png",
                    headers={"Cache-Control": "max-age=3600"})


@router.get("/datasets/plot")
def plot(path: str, index: int):
    name = _guard(datasets.episode_plot, path, index)
    return RedirectResponse(f"/gui-cache/plots/{name}")
