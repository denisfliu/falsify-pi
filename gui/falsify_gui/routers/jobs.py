from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..jobs.definitions import JOB_TYPES
from ..jobs.manager import GpuBusyError, JobManager

router = APIRouter()


def _mgr(request: Request) -> JobManager:
    return request.app.state.jobs


def _job_dict(mgr: JobManager, job) -> dict:
    d = job.to_dict()
    d["progress"] = mgr.progress(job) if job.status == "running" else {"mode": "none"}
    return d


class LaunchBody(BaseModel):
    type: str
    args: dict = {}
    override: bool = False
    queue: bool = False          # GPU busy → enqueue instead of 409
    chain: list = []             # [{"type", "args"}] launched on success


@router.get("/jobs/types")
def job_types() -> list[dict]:
    return [jt.schema() for jt in JOB_TYPES.values()]


@router.post("/jobs", status_code=201)
def launch(body: LaunchBody, request: Request) -> dict:
    mgr = _mgr(request)
    try:
        job = mgr.launch(body.type, body.args, override=body.override,
                         queue=body.queue, chain=body.chain)
    except GpuBusyError as e:
        raise HTTPException(409, detail={
            "error": "gpu job already running or queued",
            "running_job_id": e.running_job_id,
            "hint": "pass queue=true to enqueue, or override=true to run anyway"})
    except (KeyError, ValueError) as e:
        raise HTTPException(422, detail=str(e))
    return _job_dict(mgr, job)


@router.get("/jobs")
def list_jobs(request: Request, status: str | None = None,
              type: str | None = None, limit: int = 200) -> list[dict]:
    mgr = _mgr(request)
    return [_job_dict(mgr, j) for j in mgr.list(status=status, type_=type, limit=limit)]


@router.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request) -> dict:
    mgr = _mgr(request)
    job = mgr.get(job_id)
    if job is None:
        raise HTTPException(404)
    return _job_dict(mgr, job)


@router.post("/jobs/{job_id}/kill")
def kill_job(job_id: str, request: Request) -> dict:
    mgr = _mgr(request)
    try:
        job = mgr.kill(job_id)
    except KeyError:
        raise HTTPException(404)
    return {"ok": True, "status": job.status}


@router.get("/jobs/{job_id}/logs")
async def job_logs(job_id: str, request: Request, tail_kb: int = 64):
    """SSE stream of the job's log file. Replays the last tail_kb, then
    follows until the job leaves `running` and EOF is reached."""
    mgr = _mgr(request)
    job = mgr.get(job_id)
    if job is None:
        raise HTTPException(404)
    log_path = Path(job.log_path)

    async def gen():
        f = None
        buf = b""
        try:
            while True:
                if f is None and log_path.exists():
                    f = open(log_path, "rb")
                    size = log_path.stat().st_size
                    if size > tail_kb * 1024:
                        f.seek(size - tail_kb * 1024)
                        f.readline()  # drop the partial first line
                got = False
                if f is not None:
                    chunk = f.read()
                    if chunk:
                        got = True
                        buf += chunk
                        *lines, buf = buf.split(b"\n")
                        for line in lines:
                            yield f"data: {line.decode(errors='replace')}\n\n"
                if await request.is_disconnected():
                    break
                j = mgr.get(job_id)
                if j is not None and j.status not in ("running", "queued") and not got:
                    if buf:
                        yield f"data: {buf.decode(errors='replace')}\n\n"
                    yield ("event: end\ndata: "
                           + json.dumps({"status": j.status, "exit_code": j.exit_code})
                           + "\n\n")
                    break
                await asyncio.sleep(0.5)
        finally:
            if f is not None:
                f.close()

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})
