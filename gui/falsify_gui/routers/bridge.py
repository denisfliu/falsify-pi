from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from ..services import bridge

router = APIRouter()


@router.get("/bridge/policies")
def policies(admin_url: str | None = None) -> dict:
    return bridge.list_policies(admin_url)


class SwitchBody(BaseModel):
    policy_id: str
    admin_url: str | None = None


@router.post("/bridge/switch")
def switch(body: SwitchBody, request: Request) -> dict:
    # warn-only: switching mid-rollout corrupts whatever is using the bridge
    gpu_busy = request.app.state.jobs.gpu_busy()
    res = bridge.switch_policy(body.policy_id, body.admin_url)
    if gpu_busy:
        res["warning"] = f"a GPU job was running during the switch: {gpu_busy}"
    return res
