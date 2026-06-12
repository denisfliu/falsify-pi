from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

STATUSES = ("queued", "running", "succeeded", "failed", "killed", "orphaned")


@dataclass
class Job:
    id: str
    type: str
    kind: str                      # "job" | "service"
    form_args: dict
    argv: list[str]
    status: str
    pid: int | None
    pid_starttime: int | None      # /proc/<pid>/stat field 22; guards pid reuse
    created_at: float
    ended_at: float | None
    exit_code: int | None
    log_path: str
    out_dir: str | None            # primary artifact dir/file, repo-relative when possible
    url: str | None                # services: user-facing URL
    label: str = ""
    # on-success chain: [{"type": ..., "args": {...}}, ...]; "$out_dir" in an
    # arg value is replaced with this job's out_dir when the child launches
    chain: list = field(default_factory=list)

    def to_row(self) -> tuple:
        return (self.id, self.type, self.kind, json.dumps(self.form_args),
                json.dumps(self.argv), self.status, self.pid, self.pid_starttime,
                self.created_at, self.ended_at, self.exit_code, self.log_path,
                self.out_dir, self.url, self.label, json.dumps(self.chain))

    @classmethod
    def from_row(cls, row: tuple) -> "Job":
        return cls(id=row[0], type=row[1], kind=row[2],
                   form_args=json.loads(row[3]), argv=json.loads(row[4]),
                   status=row[5], pid=row[6], pid_starttime=row[7],
                   created_at=row[8], ended_at=row[9], exit_code=row[10],
                   log_path=row[11], out_dir=row[12], url=row[13],
                   label=row[14] or "",
                   chain=json.loads(row[15]) if row[15] else [])

    def to_dict(self) -> dict:
        return asdict(self)
