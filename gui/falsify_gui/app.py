from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from . import paths
from .jobs.manager import JobManager


def create_app() -> FastAPI:
    paths.ensure_dirs()
    manager = JobManager()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        manager.adopt_orphans()
        reaper = asyncio.create_task(manager.reaper_loop())
        yield
        reaper.cancel()

    app = FastAPI(title="falsify-gui", lifespan=lifespan)
    app.state.jobs = manager

    from .routers import bridge, configs, datasets, jobs, runs
    app.include_router(configs.router, prefix="/api")
    app.include_router(jobs.router, prefix="/api")
    app.include_router(runs.router, prefix="/api")
    app.include_router(bridge.router, prefix="/api")
    app.include_router(datasets.router, prefix="/api")

    @app.get("/api/health")
    def health() -> dict:
        import os
        return {
            "ok": True,
            "repo_root": str(paths.REPO_ROOT),
            "falsify_py_exists": paths.FALSIFY_PY.exists(),
            "runs_dir_exists": paths.RUNS_DIR.exists(),
            # pi_gateway jobs + the bridge proxy need these in the server env
            "pi_api_key_set": bool(os.environ.get("PI_API_KEY")),
            "bridge_keys_set": bool(os.environ.get("PI_BRIDGE_API_KEYS")),
        }

    if paths.RUNS_DIR.exists():
        app.mount("/files", StaticFiles(directory=paths.RUNS_DIR), name="files")
    app.mount("/gui-cache", StaticFiles(directory=paths.CACHE_DIR), name="gui-cache")
    app.mount("/", StaticFiles(directory=paths.STATIC_DIR, html=True), name="static")
    return app
