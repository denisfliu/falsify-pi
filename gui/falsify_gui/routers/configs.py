from __future__ import annotations

from fastapi import APIRouter

from ..services import configs_enum

router = APIRouter()


@router.get("/configs")
def configs() -> dict:
    return configs_enum.get_configs()


@router.get("/configs/bundles")
def bundles() -> list[dict]:
    return configs_enum.get_bundles()
