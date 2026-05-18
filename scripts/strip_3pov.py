"""Create `*_no_3pov` variants of the combined datasets.

For each input dataset under data/, write a sibling directory with `_no_3pov`
appended where:
  - every parquet under data/chunk-000/ has the `3pov_1` column dropped
  - meta/info.json has `3pov_1` removed from `features`
  - meta/episodes_stats.jsonl has `3pov_1` removed from each per-episode stats dict
  - meta/episodes.jsonl, meta/tasks.jsonl, meta/custom_metadata.csv are copied verbatim

Run:
    .venv/bin/python scripts/strip_3pov.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"

INPUTS = [
    "gate_scenes_real_synth",
    "gate_scenes_real_center",
    "gate_scenes_center",
    "gate_scenes_all",
]

DROP = "3pov_1"


def strip_parquet(src: Path, dst: Path) -> None:
    table = pq.read_table(src)
    if DROP in table.column_names:
        table = table.drop([DROP])
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst)


def strip_info(src: Path, dst: Path) -> None:
    info = json.loads(src.read_text())
    info.get("features", {}).pop(DROP, None)
    dst.write_text(json.dumps(info, indent=4))


def strip_stats(src: Path, dst: Path) -> None:
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            obj = json.loads(line)
            obj.get("stats", {}).pop(DROP, None)
            fout.write(json.dumps(obj) + "\n")


def convert(name: str) -> None:
    src_root = DATA / name
    dst_root = DATA / f"{name}_no_3pov"
    print(f"=== {src_root.name} → {dst_root.name} ===")

    if dst_root.exists():
        shutil.rmtree(dst_root)
    (dst_root / "data" / "chunk-000").mkdir(parents=True)
    (dst_root / "meta").mkdir(parents=True)

    src_parquets = sorted((src_root / "data" / "chunk-000").glob("*.parquet"))
    for p in src_parquets:
        strip_parquet(p, dst_root / "data" / "chunk-000" / p.name)
    print(f"  rewrote {len(src_parquets)} parquets")

    strip_info(src_root / "meta" / "info.json", dst_root / "meta" / "info.json")
    strip_stats(
        src_root / "meta" / "episodes_stats.jsonl",
        dst_root / "meta" / "episodes_stats.jsonl",
    )
    for fname in ("episodes.jsonl", "tasks.jsonl", "custom_metadata.csv"):
        s = src_root / "meta" / fname
        if s.exists():
            shutil.copy2(s, dst_root / "meta" / fname)
    print(f"  meta: info.json, episodes_stats.jsonl rewritten; episodes/tasks/custom copied")


def main() -> None:
    for name in INPUTS:
        convert(name)
    print()
    print("=== summary ===")
    for name in INPUTS:
        out = DATA / f"{name}_no_3pov"
        n = len(list((out / "data" / "chunk-000").glob("*.parquet")))
        print(f"  {out}  {n} parquets")


if __name__ == "__main__":
    main()
