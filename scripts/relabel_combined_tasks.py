"""Rewrite task strings + dedupe task_index across a set of combined datasets.

For each dataset:
  - Build a new deduped meta/tasks.jsonl per a hard-coded episode-range → task
    schedule (matches the alphabetical-merge order combine_lerobot used).
  - Rewrite each parquet under data/chunk-000/ with the new task_index column.
  - Rewrite meta/episodes.jsonl with the new task strings.
  - Rewrite the task_index stats block in meta/episodes_stats.jsonl
    (other columns left unchanged).

Run from repo root:
    .venv/bin/python scripts/relabel_combined_tasks.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent

LEFT_GATE = "go through the gate on the left and hover over the stuffed animal"
RIGHT_GATE = "go through the gate on the right and hover over the stuffed animal"
CENTER_LEFT = "go through the center gate from the left and hover over the stuffed animal"
CENTER_RIGHT = "go through the center gate from the right and hover over the stuffed animal"

# (episode_range, task_string) schedule keyed by a stable suffix-stripped
# dataset name. The ranges follow the alphabetical merge order combine_lerobot
# used. Same schedule applies to the no_3pov and with_3pov variants of each
# dataset; only the parent directory differs.
SCHEDULES_BY_KIND = {
    "gate_scenes_center": [
        ((0, 50), CENTER_LEFT),
        ((50, 100), CENTER_RIGHT),
    ],
    "gate_scenes_real_center": [
        ((0, 50), LEFT_GATE),
        ((50, 100), RIGHT_GATE),
        ((100, 150), CENTER_LEFT),
        ((150, 200), CENTER_RIGHT),
    ],
    "gate_scenes_real_synth": [
        ((0, 50), LEFT_GATE),
        ((50, 100), RIGHT_GATE),
        ((100, 150), LEFT_GATE),
        ((150, 200), RIGHT_GATE),
    ],
    "gate_scenes_all": [
        # gate_scenes_real_combined (50 L, 50 R), then center_from_left,
        # center_from_right, synth_left_gate, synth_right_gate.
        ((0, 50), LEFT_GATE),
        ((50, 100), RIGHT_GATE),
        ((100, 150), CENTER_LEFT),
        ((150, 200), CENTER_RIGHT),
        ((200, 250), LEFT_GATE),
        ((250, 300), RIGHT_GATE),
    ],
}

# (parent_dir, dataset_name) pairs. The dataset_name is what lives under the
# parent dir on disk; the schedule key is derived by stripping the _no_3pov
# suffix when present.
DATASETS: list[tuple[str, str]] = [
    ("data/no_3pov", "gate_scenes_center_no_3pov"),
    ("data/no_3pov", "gate_scenes_real_center_no_3pov"),
    ("data/no_3pov", "gate_scenes_real_synth_no_3pov"),
    ("data/no_3pov", "gate_scenes_all_no_3pov"),
    ("data/with_3pov", "gate_scenes_center"),
    ("data/with_3pov", "gate_scenes_real_center"),
    ("data/with_3pov", "gate_scenes_real_synth"),
    ("data/with_3pov", "gate_scenes_all"),
]


def schedule_to_task_table(schedule):
    """Return (dedup_tasks_list, episode_index → task_index map)."""
    # First-appearance ordering.
    seen: list[str] = []
    for (_lo, _hi), text in schedule:
        if text not in seen:
            seen.append(text)
    task_index = {t: i for i, t in enumerate(seen)}
    ep_to_idx: dict[int, int] = {}
    for (lo, hi), text in schedule:
        for ep in range(lo, hi):
            ep_to_idx[ep] = task_index[text]
    return seen, ep_to_idx


def rewrite_parquet(path: Path, new_task_index: int) -> int:
    """Replace the task_index column with a constant int64 = new_task_index."""
    t = pq.read_table(path)
    n = t.num_rows
    col_idx = t.schema.get_field_index("task_index")
    field = t.schema.field("task_index")
    arr = pa.array(np.full(n, new_task_index, dtype=np.int64))
    new_t = t.set_column(col_idx, field, arr)
    # Write to temp then move so a crash mid-write doesn't leave a half-baked
    # parquet in place.
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(new_t, tmp)
    tmp.replace(path)
    return n


def relabel_dataset(parent: str, name: str, schedule):
    ds_dir = REPO / parent / name
    meta = ds_dir / "meta"
    chunks = ds_dir / "data" / "chunk-000"
    assert meta.exists() and chunks.exists(), f"unexpected layout: {ds_dir}"

    tasks_list, ep_to_idx = schedule_to_task_table(schedule)
    print(f"  → dedup tasks ({len(tasks_list)}):")
    for i, t in enumerate(tasks_list):
        print(f"      {i}: {t}")

    # Episodes (we need lengths from existing episodes.jsonl).
    eps_in: list[dict] = []
    with (meta / "episodes.jsonl").open() as f:
        for line in f:
            eps_in.append(json.loads(line))
    eps_in.sort(key=lambda r: r["episode_index"])

    # tasks.jsonl
    (meta / "tasks.jsonl").write_text(
        "\n".join(json.dumps({"task_index": i, "task": t})
                  for i, t in enumerate(tasks_list)) + "\n"
    )

    # Parquets + episodes.jsonl
    new_eps: list[dict] = []
    total_frames = 0
    for r in eps_in:
        ep = int(r["episode_index"])
        if ep not in ep_to_idx:
            raise KeyError(f"{name}: episode {ep} not covered by schedule")
        new_idx = ep_to_idx[ep]
        new_task = tasks_list[new_idx]
        pq_path = chunks / f"episode_{ep:06d}.parquet"
        n = rewrite_parquet(pq_path, new_idx)
        total_frames += n
        new_eps.append({
            "episode_index": ep,
            "tasks": [new_task],
            "length": int(r.get("length", n)),
        })

    with (meta / "episodes.jsonl").open("w") as f:
        for r in new_eps:
            f.write(json.dumps(r) + "\n")

    # episodes_stats.jsonl — keep all other columns' stats; rewrite only
    # task_index per-episode.
    stats: list[dict] = []
    with (meta / "episodes_stats.jsonl").open() as f:
        for line in f:
            stats.append(json.loads(line))
    stats.sort(key=lambda r: r["episode_index"])
    for r in stats:
        ep = int(r["episode_index"])
        new_idx = ep_to_idx[ep]
        # Get current count from existing block (cheap).
        cnt = r["stats"]["task_index"]["count"]
        r["stats"]["task_index"] = {
            "min": [float(new_idx)],
            "max": [float(new_idx)],
            "mean": [float(new_idx)],
            "std": [0.0],
            "count": cnt,
        }
    with (meta / "episodes_stats.jsonl").open("w") as f:
        for r in stats:
            f.write(json.dumps(r) + "\n")

    # info.json — bump total_tasks.
    info_path = meta / "info.json"
    info = json.loads(info_path.read_text())
    info["total_tasks"] = len(tasks_list)
    info_path.write_text(json.dumps(info, indent=4) + "\n")

    print(f"  ✓ rewrote {len(new_eps)} parquets · {total_frames} frames · "
          f"{len(tasks_list)} task(s)")


def main():
    for parent, name in DATASETS:
        kind = name.removesuffix("_no_3pov")
        schedule = SCHEDULES_BY_KIND[kind]
        print(f"=== {parent}/{name} ===")
        relabel_dataset(parent, name, schedule)
        print()
    print("[done]")


if __name__ == "__main__":
    main()
