from __future__ import annotations

import sqlite3
import threading

from ..paths import DB_PATH
from .models import Job

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    kind TEXT NOT NULL,
    form_args TEXT NOT NULL,
    argv TEXT NOT NULL,
    status TEXT NOT NULL,
    pid INTEGER,
    pid_starttime INTEGER,
    created_at REAL NOT NULL,
    ended_at REAL,
    exit_code INTEGER,
    log_path TEXT NOT NULL,
    out_dir TEXT,
    url TEXT,
    label TEXT,
    chain TEXT,
    group_id TEXT,
    group_label TEXT
);
"""

_COLS = ("id, type, kind, form_args, argv, status, pid, pid_starttime, "
         "created_at, ended_at, exit_code, log_path, out_dir, url, label, "
         "chain, group_id, group_label")


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._db = sqlite3.connect(DB_PATH, check_same_thread=False)
        with self._lock:
            self._db.execute(_SCHEMA)
            cols = {r[1] for r in self._db.execute("PRAGMA table_info(jobs)")}
            for col in ("chain", "group_id", "group_label"):
                if col not in cols:   # migrate older databases
                    self._db.execute(f"ALTER TABLE jobs ADD COLUMN {col} TEXT")
            self._db.commit()

    def insert(self, job: Job) -> None:
        with self._lock:
            self._db.execute(
                f"INSERT INTO jobs ({_COLS}) VALUES ({','.join('?' * 18)})",
                job.to_row())
            self._db.commit()

    def update(self, job: Job) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE jobs SET status=?, pid=?, pid_starttime=?, ended_at=?, "
                "exit_code=?, out_dir=?, url=? WHERE id=?",
                (job.status, job.pid, job.pid_starttime, job.ended_at,
                 job.exit_code, job.out_dir, job.url, job.id))
            self._db.commit()

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            row = self._db.execute(
                f"SELECT {_COLS} FROM jobs WHERE id=?", (job_id,)).fetchone()
        return Job.from_row(row) if row else None

    def list(self, status: str | None = None, type_: str | None = None,
             limit: int = 200) -> list[Job]:
        q = f"SELECT {_COLS} FROM jobs"
        clauses, params = [], []
        if status:
            clauses.append("status=?")
            params.append(status)
        if type_:
            clauses.append("type=?")
            params.append(type_)
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        q += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._db.execute(q, params).fetchall()
        return [Job.from_row(r) for r in rows]
