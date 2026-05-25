"""Drop the `3pov_1` column from a LeRobot v2.1 dataset.

Reads one (or many) v2.1 datasets and writes the stripped variant(s):

  - every parquet under data/chunk-000/ has the `3pov_1` column dropped
  - meta/info.json has `3pov_1` removed from `features`
  - meta/episodes_stats.jsonl has `3pov_1` removed from each per-episode stats dict
  - meta/episodes.jsonl, meta/tasks.jsonl, meta/custom_metadata.csv are copied verbatim

CLI shape mirrors ``scripts/dataset/convert_no_3pov_to_v3.py`` so the three-stage
pipeline composes cleanly:

  python -m falsify.cli.combine_lerobot --src <atomic-parent> --out data/my_combined ...
  python scripts/dataset/strip_3pov.py            --dataset data/my_combined        --out data/my_combined_no_3pov
  python scripts/dataset/convert_no_3pov_to_v3.py --dataset data/my_combined_no_3pov --out data/my_combined_no_3pov_v3

Batch mode (``--src-root``) processes every immediate child directory.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq

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


def strip_dataset(src_root: Path, dst_root: Path) -> None:
    print(f"=== {src_root} → {dst_root} ===")
    if dst_root.exists():
        shutil.rmtree(dst_root)
    (dst_root / "data" / "chunk-000").mkdir(parents=True)
    (dst_root / "meta").mkdir(parents=True)

    src_parquets = sorted((src_root / "data" / "chunk-000").glob("*.parquet"))
    for p in src_parquets:
        strip_parquet(p, dst_root / "data" / "chunk-000" / p.name)
    print(f"  rewrote {len(src_parquets)} parquets")

    strip_info(src_root / "meta" / "info.json", dst_root / "meta" / "info.json")
    stats_src = src_root / "meta" / "episodes_stats.jsonl"
    if stats_src.exists():
        strip_stats(stats_src, dst_root / "meta" / "episodes_stats.jsonl")
        stats_msg = "episodes_stats.jsonl rewritten"
    else:
        stats_msg = "episodes_stats.jsonl absent (skipped)"
    for fname in ("episodes.jsonl", "tasks.jsonl", "custom_metadata.csv"):
        s = src_root / "meta" / fname
        if s.exists():
            shutil.copy2(s, dst_root / "meta" / fname)
    print(f"  meta: info.json rewritten, {stats_msg}, episodes/tasks/custom copied")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog=(
            "Pass --dataset to convert one v2.1 bundle, or --src-root to "
            "convert every immediate child directory under it."
        ),
    )
    ap.add_argument("--dataset", type=Path, default=None,
                    help="Single v2.1 dataset directory to strip. "
                         "Mutually exclusive with --src-root.")
    ap.add_argument("--src-root", type=Path, default=None,
                    help="Parent directory of v2.1 datasets to strip in batch.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output path. With --dataset: the destination "
                         "stripped dataset dir (default: sibling with "
                         "`_no_3pov` appended). With --src-root: the parent "
                         "directory holding the stripped datasets (default: "
                         "sibling parent with `_no_3pov` appended).")
    args = ap.parse_args()

    if (args.dataset is None) == (args.src_root is None):
        raise SystemExit("exactly one of --dataset / --src-root is required")

    if args.dataset is not None:
        src = args.dataset.resolve()
        out = (args.out or src.parent / f"{src.name}_no_3pov").resolve()
        strip_dataset(src, out)
        print(f"[done] → {out}")
        return

    src_root = args.src_root.resolve()
    out_root = (args.out or src_root.parent / f"{src_root.name}_no_3pov").resolve()
    datasets = sorted(d for d in src_root.iterdir() if d.is_dir())
    print(f"stripping {len(datasets)} dataset(s) under {src_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    for ds in datasets:
        strip_dataset(ds, out_root / ds.name)
        print()
    print(f"[done] all stripped datasets under {out_root}")


if __name__ == "__main__":
    main()
