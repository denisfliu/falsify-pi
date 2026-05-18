"""Build the `meta/custom_metadata.csv` the pi-data-sharing validator expects.

The validator (``/home/dfliu/code/dataset_validation/pi-data-sharing``) needs
a sidecar CSV alongside the standard LeRobot ``meta/`` files. This script
generates it from what we already know about the dataset:

- ``episode_index`` and per-episode task come from ``meta/episodes.jsonl``.
- ``start_timestamp`` is the parquet file's mtime — the dataset doesn't
  carry an absolute wall-clock time, and the validator wants UTC seconds in
  [year 2000, 2100] so mtime is the cleanest stable choice.
- ``station_id`` is derived from the task: left-gate tasks → ``left_gate``,
  right-gate → ``right_gate``. This is the "scene the episode was collected
  in" knob and matters when the validator (or downstream tooling) wants to
  group episodes by site.
- ``operator_id`` / ``robot_id`` are constants for our setup — both
  overridable via CLI flags if a future dataset has a different collector.

Usage::

    python tools/build_validator_metadata.py \
        --dataset-path data/gate_scenes_real_combined

Re-running overwrites the CSV idempotently.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def task_to_station(task: str) -> str:
    """Map the human-readable task string to a stable station_id.

    Order matters: "center" must be checked before "left"/"right" because
    a center-gate task may mention which side the drone approached from
    (e.g. "...from the left...") and we want station_id to reflect the
    *scene* the episode was rendered against, not the approach direction.
    """
    t = task.lower()
    if "center" in t:
        return "center_gate"
    if "left" in t:
        return "left_gate"
    if "right" in t:
        return "right_gate"
    return "unknown_gate"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--dataset-path", required=True, type=Path,
        help="LeRobot dataset root (must contain meta/episodes.jsonl)",
    )
    ap.add_argument("--operator-id", default="denis")
    ap.add_argument("--robot-id", default="carl")
    ap.add_argument(
        "--dataset-tag", default=None,
        help="Prefix for episode_id (default: dataset directory name)",
    )
    ap.add_argument(
        "--success-default", choices=("true", "false"), default="true",
        help="success column value when we have no per-episode signal",
    )
    args = ap.parse_args()

    root: Path = args.dataset_path.resolve()
    episodes_path = root / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        ap.error(f"missing {episodes_path}")
    out_path = root / "meta" / "custom_metadata.csv"
    dataset_tag = args.dataset_tag or root.name

    success_default = args.success_default == "true"

    rows: list[dict] = []
    with episodes_path.open() as f:
        for line in f:
            ep = json.loads(line)
            idx = int(ep["episode_index"])
            tasks = ep.get("tasks", [])
            task = tasks[0] if tasks else ""
            parquet = root / "data" / "chunk-000" / f"episode_{idx:06d}.parquet"
            if not parquet.exists():
                raise FileNotFoundError(parquet)
            start_ts = int(os.path.getmtime(parquet))
            rows.append({
                "episode_index": idx,
                "operator_id": args.operator_id,
                "is_eval_episode": False,
                "episode_id": f"{dataset_tag}_ep_{idx:04d}",
                "start_timestamp": start_ts,
                "checkpoint_path": "",
                "success": success_default,
                "station_id": task_to_station(task),
                "robot_id": args.robot_id,
            })

    fields = [
        "episode_index", "operator_id", "is_eval_episode", "episode_id",
        "start_timestamp", "checkpoint_path", "success", "station_id",
        "robot_id",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"wrote {out_path} ({len(rows)} episodes)")
    print(f"  stations: {sorted({r['station_id'] for r in rows})}")
    print(f"  timestamp range: {min(r['start_timestamp'] for r in rows)} "
          f"to {max(r['start_timestamp'] for r in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
