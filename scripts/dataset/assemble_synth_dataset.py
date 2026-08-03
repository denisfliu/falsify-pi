"""Assemble ``export_training_data`` per-episode dirs into a LeRobot dataset.

The exporter writes one ``episode_NNNNNN/episode_NNNNNN.parquet`` per
trajectory, but ``combine_lerobot`` (which regenerates the four ``meta/``
files) consumes the LeRobot v2.1 ``data/chunk-000/`` layout. This bridges
the two: it reshapes the per-episode dirs into a single v2.1 source
dataset (copying the ``features`` block from a reference dataset's
``info.json``, since the RGB embodiment schema is identical), then runs
``combine_lerobot`` to emit a finished dataset with regenerated meta.

Single-task by default (all episodes get ``--prompt``); pass through any
``--task "COUNT:TEXT"`` specs for multi-task ranges.

Example::

    PYTHONPATH=src python scripts/dataset/assemble_synth_dataset.py \
        --staging-dir runs/datasets_staging/synth_center_from_left \
        --out data/atomic_datasets/synth_center_from_left \
        --prompt "go through the center gate from the left and hover over the stuffed animal"
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_REF_INFO = (
    REPO_ROOT / "data/atomic_datasets/gate_scenes_real_combined_rgb"
    / "meta/info.json"
)


def _episode_parquets(staging: Path) -> list[Path]:
    """Per-episode export parquets, sorted by episode dir name."""
    out = []
    for d in sorted(staging.glob("episode_*")):
        pqs = sorted(d.glob("episode_*.parquet"))
        if pqs:
            out.append(pqs[0])
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--staging-dir", required=True, type=Path,
                   help="Exporter output: episode_NNNNNN/episode_NNNNNN.parquet")
    p.add_argument("--out", required=True, type=Path,
                   help="Final dataset directory to write.")
    p.add_argument("--prompt", type=str, default=None,
                   help="Single task text applied to every episode "
                        "(shorthand for --task 'rest:<prompt>').")
    p.add_argument("--task", action="append", default=[], metavar="COUNT:TEXT",
                   help="Explicit combine_lerobot task spec; repeatable. "
                        "Overrides --prompt when given.")
    p.add_argument("--reference-info", type=Path, default=_DEFAULT_REF_INFO,
                   help="Dataset info.json to copy the features block from "
                        "(schema-identical RGB dataset).")
    p.add_argument("--work-dir", type=Path, default=None,
                   help="Scratch dir for the intermediate v2.1 source "
                        "(default: <out>_v21_src alongside --out).")
    args = p.parse_args(argv)

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from falsify.cli.combine_lerobot import main as combine_main

    parquets = _episode_parquets(args.staging_dir)
    if not parquets:
        raise SystemExit(f"no episode_*/episode_*.parquet under {args.staging_dir}")

    if args.task:
        tasks = args.task
    elif args.prompt:
        tasks = [f"rest:{args.prompt}"]
    else:
        raise SystemExit("pass --prompt or at least one --task")

    # ---- 1) reshape per-episode dirs → one v2.1 source dataset ----------
    work = args.work_dir or args.out.parent / f"{args.out.name}_v21_src"
    src_parent = work                      # combine --src iterates subdirs
    src_ds = src_parent / "src"            # the single source dataset dir
    if src_parent.exists():
        shutil.rmtree(src_parent)
    chunk = src_ds / "data" / "chunk-000"
    chunk.mkdir(parents=True)
    (src_ds / "meta").mkdir(parents=True)
    for i, pq in enumerate(parquets):
        shutil.copy2(pq, chunk / f"episode_{i:06d}.parquet")

    # combine only reads the `features` block from the source info.json;
    # everything else it regenerates. Copy it from the reference dataset.
    ref = json.loads(args.reference_info.read_text())
    (src_ds / "meta" / "info.json").write_text(
        json.dumps({"features": ref.get("features", {})}, indent=4)
    )
    print(f"[assemble] reshaped {len(parquets)} episode(s) → {src_ds}")

    # ---- 2) combine → finished dataset with regenerated meta -----------
    combine_argv = [
        "--src", str(src_parent),
        "--out", str(args.out),
    ]
    for t in tasks:
        combine_argv += ["--task", t]
    rc = combine_main(combine_argv)
    if rc == 0:
        print(f"[assemble] wrote dataset → {args.out}")
        shutil.rmtree(src_parent, ignore_errors=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
