"""Subprocess construction for falsify jobs.

Every job runs the repo's CLIs with the falsify venv's python, under
`tools/env.sh` (gcc-11 pins for CUDA JIT, acados paths, PYTHONPATH).
Sourcing the script — rather than replicating its exports here — means
fixes to env.sh are inherited automatically.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .paths import FALSIFY_PY, REPO_ROOT

_WRAPPER = 'source tools/env.sh && exec "$@"'


def build_argv(script_args: list[str]) -> list[str]:
    return ["bash", "-c", _WRAPPER, "--", str(FALSIFY_PY), *script_args]


def spawn(script_args: list[str], log_path: Path,
          extra_env: dict[str, str] | None = None) -> subprocess.Popen:
    """Start a job in its own process group with stdout+stderr appended to
    log_path. The file handle is duplicated into the child, so the parent
    closes its copy immediately."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if extra_env:
        env.update(extra_env)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as logf:
        proc = subprocess.Popen(
            build_argv(script_args),
            cwd=REPO_ROOT,
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    return proc
