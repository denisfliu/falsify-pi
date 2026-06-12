"""Process lifecycle for GUI-launched falsify jobs.

Design rules (see plan):
- stdout+stderr go to a per-job log file, never a pipe — tailing is file
  reading, jobs survive GUI restarts, no pipe-buffer deadlock.
- every job runs in its own process group (start_new_session) so kill
  signals the whole tree.
- pid + /proc starttime are persisted so a restarted GUI can re-adopt
  still-running jobs without confusing a reused pid for ours.
"""
from __future__ import annotations

import asyncio
import os
import signal
import time
import uuid
from pathlib import Path

from .. import envsetup
from ..paths import JOBS_DIR, REPO_ROOT
from .definitions import JOB_TYPES, JobType
from .models import Job
from .store import JobStore


class GpuBusyError(Exception):
    def __init__(self, running_job_id: str):
        self.running_job_id = running_job_id
        super().__init__(f"a GPU job is already running: {running_job_id}")


def proc_starttime(pid: int) -> int | None:
    """Field 22 of /proc/<pid>/stat (clock ticks since boot). comm (field 2)
    may contain spaces/parens, so parse after the last ')'."""
    try:
        stat = open(f"/proc/{pid}/stat").read()
        return int(stat.rsplit(")", 1)[1].split()[19])
    except (FileNotFoundError, ProcessLookupError, IndexError, ValueError):
        return None


def proc_alive(job: Job) -> bool:
    if not job.pid:
        return False
    st = proc_starttime(job.pid)
    return st is not None and st == job.pid_starttime


class JobManager:
    def __init__(self) -> None:
        self.store = JobStore()
        self._procs: dict[str, object] = {}      # job_id -> Popen (this process only)
        self._kill_requested: dict[str, float] = {}   # job_id -> request time
        self._kill_grace_s = 10.0
        # lingering-process reaping: job_id -> time its done_marker was first
        # seen in the log; after the grace the group is SIGINTed and the job
        # is recorded as succeeded (the script's work was already complete)
        self._done_seen: dict[str, float] = {}
        self._lingering: set[str] = set()
        self._linger_grace_s = 60.0

    # ------------------------------------------------------------- launch
    def launch(self, type_name: str, form_args: dict, override: bool = False,
               queue: bool = False, chain: list | None = None) -> Job:
        """Start a job now, or — for GPU jobs while another GPU job is
        running/queued — either 409 (default) or enqueue (queue=True).
        `chain` entries launch on success with "$out_dir" substituted."""
        jt = JOB_TYPES.get(type_name)
        if jt is None:
            raise KeyError(f"unknown job type {type_name!r}")
        built = jt.build(form_args)
        if jt.kind == "service" and built.url and built.url.startswith("port:"):
            for j in self.store.list(status="running"):
                if j.kind == "service" and j.url == built.url:
                    raise ValueError(
                        f"{built.url} already used by running service {j.id}")
        gpu_blocked = (jt.gpu and not override
                       and (self.gpu_busy() is not None or self._gpu_queued()))
        if gpu_blocked and not queue:
            raise GpuBusyError(self.gpu_busy() or self._gpu_queued() or "?")
        job_id = f"job-{time.strftime('%Y%m%d-%H%M%S')}-{type_name}-{uuid.uuid4().hex[:4]}"
        job = Job(
            id=job_id, type=type_name, kind=jt.kind, form_args=form_args,
            argv=envsetup.build_argv(built.script_args), status="queued",
            pid=None, pid_starttime=None,
            created_at=time.time(), ended_at=None, exit_code=None,
            log_path=str(JOBS_DIR / job_id / "job.log"),
            out_dir=built.out_dir, url=built.url, label=built.label,
            chain=chain or [],
        )
        self.store.insert(job)
        if not gpu_blocked:
            self._start(job)
        return job

    def _start(self, job: Job) -> None:
        # argv = [bash, -c, wrapper, --, python, *script_args]
        script_args = job.argv[5:]
        proc = envsetup.spawn(script_args, Path(job.log_path))
        job.pid = proc.pid
        job.pid_starttime = proc_starttime(proc.pid)
        job.status = "running"
        self.store.update(job)
        self._procs[job.id] = proc

    def _gpu_queued(self) -> str | None:
        for j in self.store.list(status="queued"):
            jt = JOB_TYPES.get(j.type)
            if jt is not None and jt.gpu:
                return j.id
        return None

    def _advance_queue(self) -> None:
        queued = self.store.list(status="queued")
        if not queued:
            return
        queued.sort(key=lambda j: j.created_at)
        gpu_free = self.gpu_busy() is None
        for job in queued:
            jt = JOB_TYPES.get(job.type)
            if jt is None:
                continue
            if jt.gpu:
                if gpu_free:
                    print(f"[gui] starting queued job {job.id}")
                    self._start(job)
                    gpu_free = False
            else:
                # non-GPU jobs never wait (only chain children land here)
                self._start(job)

    # --------------------------------------------------------------- kill
    def kill(self, job_id: str) -> Job:
        """SIGINT the job's process group; the reaper escalates to SIGKILL
        after the grace period if it's still alive."""
        job = self.store.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status == "queued":
            # cancel without signals — nothing is running yet
            job.status = "killed"
            job.ended_at = time.time()
            self.store.update(job)
            return job
        if job.status != "running" or not job.pid:
            return job
        self._kill_requested.setdefault(job_id, time.time())
        try:
            os.killpg(job.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        return job

    # ------------------------------------------------------------- reaper
    async def reaper_loop(self, interval_s: float = 2.0) -> None:
        while True:
            try:
                self._reap_once()
            except Exception as e:  # never let the reaper die
                print(f"[gui] reaper error: {e!r}")
            await asyncio.sleep(interval_s)

    def _reap_once(self) -> None:
        now = time.time()
        for job_id, requested_at in list(self._kill_requested.items()):
            if now - requested_at < self._kill_grace_s:
                continue
            # SIGKILL the whole group even if the main process already
            # exited — a grandchild may have survived the SIGINT.
            j = self.store.get(job_id)
            if j and j.pid:
                try:
                    os.killpg(j.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self._kill_requested.pop(job_id, None)
        for job in self.store.list(status="running"):
            proc = self._procs.get(job.id)
            if proc is not None:
                code = proc.poll()
                if code is not None:
                    self._finish(job, exit_code=code)
                    self._procs.pop(job.id, None)
                    continue
            elif not proc_alive(job):
                # adopted job (launched by a previous GUI process) that ended
                self._finish(job, exit_code=None)
                continue
            # still running (own or adopted) — watch for a printed
            # done_marker with a lingering process
            self._check_lingering(job, now)
        self._advance_queue()

    def _check_lingering(self, job: Job, now: float) -> None:
        """Scripts that leave non-daemon threads behind print their final
        log line and then never exit. Once the job type's done_marker shows
        up in the log tail and the grace passes, SIGINT the group; _finish
        records the job as succeeded."""
        jt = JOB_TYPES.get(job.type)
        if jt is None or jt.done_marker is None or job.id in self._kill_requested:
            return
        if job.id not in self._done_seen:
            try:
                with open(job.log_path, "rb") as f:
                    f.seek(max(0, os.fstat(f.fileno()).st_size - 8192))
                    tail = f.read().decode(errors="replace")
            except OSError:
                return
            if jt.done_marker not in tail:
                return
            self._done_seen[job.id] = now
            return
        if now - self._done_seen[job.id] < self._linger_grace_s:
            return
        print(f"[gui] {job.id}: done_marker seen, process lingering — reaping")
        self._lingering.add(job.id)
        self._kill_requested.setdefault(job.id, now)   # arm SIGKILL escalation
        try:
            os.killpg(job.pid, signal.SIGINT)
        except (ProcessLookupError, TypeError):
            pass

    def _finish(self, job: Job, exit_code: int | None) -> None:
        if exit_code is None:
            # give the progress reader a chance to discover an auto-generated
            # out dir from the log before status inference relies on it
            self.progress(job)
        job.ended_at = time.time()
        job.exit_code = exit_code
        if job.id in self._lingering:
            # the script's work completed (done_marker printed) — we only
            # killed leftover non-daemon threads, so this is a success
            job.status = "succeeded"
            self._lingering.discard(job.id)
            self._done_seen.pop(job.id, None)
        elif job.id in self._kill_requested:
            # leave the entry in place: the reaper still fires the group
            # SIGKILL after the grace period to catch surviving grandchildren
            job.status = "killed"
        elif exit_code is not None:
            job.status = "succeeded" if exit_code == 0 else "failed"
        else:
            # exit code unknown (adopted) — infer from expected outputs
            job.status = self._infer_terminal(job) or "orphaned"
        self.store.update(job)
        if job.status == "succeeded" and job.chain:
            self._launch_chain(job)

    def _launch_chain(self, parent: Job) -> None:
        entry, rest = parent.chain[0], parent.chain[1:]
        args = {}
        for k, v in (entry.get("args") or {}).items():
            if isinstance(v, str) and "$out_dir" in v:
                v = v.replace("$out_dir", parent.out_dir or "")
            args[k] = v
        try:
            child = self.launch(entry["type"], args, queue=True, chain=rest)
            print(f"[gui] chained {child.id} after {parent.id}")
        except Exception as e:
            print(f"[gui] chain after {parent.id} failed to launch: {e!r}")

    def _infer_terminal(self, job: Job) -> str | None:
        jt = JOB_TYPES.get(job.type)
        if jt is not None and jt.finalize is not None:
            try:
                return jt.finalize(job)
            except Exception:
                return None
        # default heuristic: only trust a *file* output — directories exist
        # from the moment a job starts, so they prove nothing
        if job.out_dir:
            out = REPO_ROOT / job.out_dir
            if out.is_file():
                return "succeeded"
        return None

    # ----------------------------------------------------------- adoption
    def adopt_orphans(self) -> None:
        """On startup: jobs recorded running either get re-adopted (process
        still alive, starttime matches) or finalized."""
        for job in self.store.list(status="running"):
            if proc_alive(job):
                print(f"[gui] re-adopted running job {job.id} (pid {job.pid})")
            else:
                self._finish(job, exit_code=None)
                print(f"[gui] finalized stale job {job.id} -> {job.status}")

    # ------------------------------------------------------------ queries
    def get(self, job_id: str) -> Job | None:
        return self.store.get(job_id)

    def list(self, **kw) -> list[Job]:
        return self.store.list(**kw)

    def progress(self, job: Job) -> dict:
        jt = JOB_TYPES.get(job.type)
        if jt is None or jt.progress is None:
            return {"mode": "none"}
        try:
            res = jt.progress(job)
        except Exception as e:
            return {"mode": "none", "error": repr(e)}
        # progress readers report auto-generated out dirs discovered from
        # the job's log; persist so links/finalize keep working
        if res.get("out_dir") and res["out_dir"] != job.out_dir:
            job.out_dir = res["out_dir"]
            self.store.update(job)
        return res

    def gpu_busy(self) -> str | None:
        for j in self.store.list(status="running"):
            jt = JOB_TYPES.get(j.type)
            if jt is not None and jt.gpu:
                return j.id
        return None
