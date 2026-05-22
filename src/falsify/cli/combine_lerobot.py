"""Combine multiple LeRobot v2.1 dataset directories into one.

Reads each source dataset, drops a configurable last parquet from
"bad" directories, renumbers ``episode_index`` and ``index`` globally,
reassigns ``task_index`` according to ``--task`` ranges, and writes a
fresh LeRobot v2.1 dataset with regenerated meta files (info.json,
tasks.jsonl, episodes.jsonl, episodes_stats.jsonl).

Designed to match the on-disk format of ``~/Downloads/episode_000008.parquet``
and the existing ``data/gate_scenes_real`` bundle.

Source directory layout (per LeRobot v2.1)::

  <src>/<dataset_NN>/
    data/chunk-000/episode_NNNNNN.parquet
    meta/{info.json, tasks.jsonl, episodes.jsonl, episodes_stats.jsonl}

Output is the same layout in ``<out>/``.

Task assignment
---------------
``--task "<count>:<text>"`` can be repeated. The first ``<count>``
episodes (in the global combined order) get the first task, the next
``<count>`` get the second, etc. The last ``--task`` may use the literal
``rest`` as its count to grab whatever's left.

Identical task **texts** across multiple ``--task`` specs collapse to a
single ``task_index`` in the output. The resulting ``tasks.jsonl`` always
has exactly one row per unique text — never two rows with the same string
under different indices.

Bad-last filtering
------------------
``--drop-last-pattern`` is an fnmatch pattern (default ``*_bad_last``).
Any source dataset directory whose basename matches the pattern has its
*last* parquet (by sorted episode_index) dropped.

Example::

    PYTHONPATH=src .venv/bin/python -m falsify.cli.combine_lerobot \\
        --src data/gate_scenes_real/datasets-20260513T185355Z-3-001/datasets \\
        --out data/gate_scenes_real_combined \\
        --drop-last-pattern "*_bad_last" \\
        --task "50:go through the gate on the left and hover over the stuffed animal" \\
        --task "rest:go through the gate on the right and hover over the stuffed animal"
"""

from __future__ import annotations

import argparse
import fnmatch
import io
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------


_TRAIL_NUM_RE = re.compile(r"(\d+)(?:_bad_last)?$")


def _dataset_sort_key(p: Path) -> tuple:
    """Sort directories by their trailing numeric index, then by name as tiebreaker."""
    m = _TRAIL_NUM_RE.search(p.name)
    if m:
        return (int(m.group(1)), p.name)
    return (1 << 30, p.name)


def _discover_datasets(src: Path) -> list[Path]:
    return sorted(
        (p for p in src.iterdir() if p.is_dir() and (p / "data").exists()),
        key=_dataset_sort_key,
    )


def _episode_parquets(ds: Path) -> list[Path]:
    return sorted((ds / "data" / "chunk-000").glob("episode_*.parquet"))


# ---------------------------------------------------------------------------
# Task spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskSpec:
    text: str
    count: Optional[int]   # None = "rest"


def _parse_tasks(specs: Iterable[str]) -> list[TaskSpec]:
    out: list[TaskSpec] = []
    for s in specs:
        if ":" not in s:
            raise SystemExit(
                f"--task must look like '<count>:<text>' (got {s!r}); "
                "use 'rest' as the count to consume all remaining episodes"
            )
        count_s, text = s.split(":", 1)
        count_s = count_s.strip()
        text = text.strip()
        if count_s.lower() == "rest":
            out.append(TaskSpec(text=text, count=None))
        else:
            try:
                cnt = int(count_s)
            except ValueError as e:
                raise SystemExit(f"--task count must be int or 'rest'; got {count_s!r}") from e
            if cnt <= 0:
                raise SystemExit(f"--task count must be > 0; got {cnt}")
            out.append(TaskSpec(text=text, count=cnt))
    if not out:
        raise SystemExit("at least one --task required")
    # Only the last task may be 'rest'.
    rest_indices = [i for i, t in enumerate(out) if t.count is None]
    if len(rest_indices) > 1 or (rest_indices and rest_indices[0] != len(out) - 1):
        raise SystemExit("'rest' may only appear as the LAST --task")
    return out


def _canonical_task_map(tasks: list[TaskSpec]) -> dict[str, int]:
    """Map each unique task text to a canonical index, in first-seen order.

    Repeating ``--task "<count>:<same text>"`` does NOT create a new
    task_index — both ranges land on the same canonical index. This is
    what prevents the historical "doubled task indices" bug where two
    --task specs with identical text wrote two distinct rows into
    tasks.jsonl pointing at the same string.
    """
    out: dict[str, int] = {}
    for t in tasks:
        if t.text not in out:
            out[t.text] = len(out)
    return out


def _task_index_for(
    global_episode: int,
    tasks: list[TaskSpec],
    canonical: dict[str, int],
    total: int,
) -> int:
    cursor = 0
    for t in tasks:
        upper = total if t.count is None else cursor + t.count
        if global_episode < upper:
            return canonical[t.text]
        cursor = upper
    raise ValueError(f"task_index_for: episode {global_episode} fell off the end")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def _decode_png_arr(b: bytes) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(b)))


def _image_stats(table: pa.Table, col: str, sample_n: int = 100) -> dict:
    """LeRobot-style per-channel image stats sampled across the episode.

    Output mirrors ``episodes_stats.jsonl`` from the source bundle: shape
    ``(3, 1, 1)`` for min/max/mean/std and ``(1,)`` for count.
    """
    n = table.num_rows
    if n == 0:
        zeros = np.zeros((3, 1, 1), dtype=np.float64)
        return {
            "min": zeros.tolist(), "max": zeros.tolist(),
            "mean": zeros.tolist(), "std": zeros.tolist(),
            "count": [0],
        }
    indices = np.linspace(0, n - 1, min(sample_n, n)).astype(int)
    pngs = table.column(col).to_pylist()
    sampled = np.stack([
        _decode_png_arr(pngs[int(i)]["bytes"]).astype(np.float32) / 255.0
        for i in indices
    ])  # (S, H, W, 3)
    axes = (0, 1, 2)
    mins = sampled.min(axis=axes).reshape(3, 1, 1)
    maxs = sampled.max(axis=axes).reshape(3, 1, 1)
    means = sampled.mean(axis=axes).reshape(3, 1, 1)
    stds = sampled.std(axis=axes).reshape(3, 1, 1)
    return {
        "min": mins.tolist(), "max": maxs.tolist(),
        "mean": means.tolist(), "std": stds.tolist(),
        "count": [int(len(sampled))],
    }


def _vec_stats(table: pa.Table, col: str) -> dict:
    arr = np.asarray(table.column(col).to_pylist(), dtype=np.float64)  # (N, D)
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "count": [int(arr.shape[0])],
    }


def _scalar_stats(table: pa.Table, col: str) -> dict:
    arr = np.asarray(table.column(col).to_pylist(), dtype=np.float64)
    return {
        "min": [float(arr.min())],
        "max": [float(arr.max())],
        "mean": [float(arr.mean())],
        "std": [float(arr.std())],
        "count": [int(arr.shape[0])],
    }


def _episode_stats(table: pa.Table) -> dict:
    cols = set(table.column_names)
    out = {
        "state":        _vec_stats(table, "state"),
        "actions":      _vec_stats(table, "actions"),
        "timestamp":    _scalar_stats(table, "timestamp"),
        "frame_index":  _scalar_stats(table, "frame_index"),
        "episode_index": _scalar_stats(table, "episode_index"),
        "index":        _scalar_stats(table, "index"),
        "task_index":   _scalar_stats(table, "task_index"),
    }
    for img_col in ("image", "wrist_image", "3pov_1"):
        if img_col in cols:
            out[img_col] = _image_stats(table, img_col)
    return out


# ---------------------------------------------------------------------------
# Precomputed-stats reuse
# ---------------------------------------------------------------------------
#
# Recomputing per-episode image stats from PNG bytes dominates the combine
# wall-clock (~1.5 s/episode for 3-camera atomic episodes, vs ~50 ms for
# the actual parquet copy). Every LeRobot v2.1 source dataset ships
# ``meta/episodes_stats.jsonl`` with byte-identical schema, so when it's
# present we reuse the immutable per-episode entries (images, state,
# actions, timestamp, frame_index) and only re-derive the three fields we
# actively renumber: ``episode_index``, ``index``, and ``task_index``.
# Those are closed-form: constant scalars or arithmetic progressions.


def _scalar_stats_constant(value: float, n: int) -> dict:
    return {
        "min":   [float(value)],
        "max":   [float(value)],
        "mean":  [float(value)],
        "std":   [0.0],
        "count": [int(n)],
    }


def _scalar_stats_arange(start: int, n: int) -> dict:
    """Closed-form stats of the integer range [start, start+n)."""
    if n <= 0:
        return {"min": [0.0], "max": [0.0], "mean": [0.0], "std": [0.0], "count": [0]}
    end = start + n - 1
    mean = start + (n - 1) / 2.0
    # std of {0, 1, ..., n-1} is sqrt((n^2 - 1)/12).
    std = ((n * n - 1) / 12.0) ** 0.5
    return {
        "min":   [float(start)],
        "max":   [float(end)],
        "mean":  [float(mean)],
        "std":   [float(std)],
        "count": [int(n)],
    }


def _load_source_stats(ds_root: Path) -> list[dict] | None:
    """Read meta/episodes_stats.jsonl; return entries indexed by parquet
    position (i.e., sorted by episode_index), or None if the file is
    missing. Entries are returned as raw ``stats`` dicts (the outer
    ``episode_index`` field is dropped — we map by position, not by the
    source's local index, because some bundles renumber that field)."""
    p = ds_root / "meta" / "episodes_stats.jsonl"
    if not p.is_file():
        return None
    entries: list[tuple[int, dict]] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        e = json.loads(line)
        entries.append((int(e["episode_index"]), e["stats"]))
    entries.sort(key=lambda t: t[0])
    return [stats for _, stats in entries]


def _renumber_stats(
    prior: dict,
    *,
    n_rows: int,
    global_episode_index: int,
    global_index_start: int,
    task_index: int,
) -> dict:
    """Reuse `prior` for immutable fields; rewrite the three renumbered ones."""
    out = dict(prior)
    out["episode_index"] = _scalar_stats_constant(global_episode_index, n_rows)
    out["index"]         = _scalar_stats_arange(global_index_start, n_rows)
    out["task_index"]    = _scalar_stats_constant(task_index, n_rows)
    return out


# ---------------------------------------------------------------------------
# Parquet renumber / rewrite
# ---------------------------------------------------------------------------


def _renumber_table(
    src_table: pa.Table,
    *,
    global_episode_index: int,
    global_index_start: int,
    task_index: int,
) -> pa.Table:
    """Return a new Table with episode_index/index/task_index updated.

    All other columns (images, state, actions, timestamp, frame_index)
    are preserved by reference — no decode/re-encode of PNG bytes.
    """
    n = src_table.num_rows
    new_ep = pa.array([global_episode_index] * n, type=pa.int64())
    new_idx = pa.array(
        [global_index_start + i for i in range(n)], type=pa.int64(),
    )
    new_task = pa.array([task_index] * n, type=pa.int64())

    cols = {}
    for name in src_table.column_names:
        if name == "episode_index":
            cols[name] = new_ep
        elif name == "index":
            cols[name] = new_idx
        elif name == "task_index":
            cols[name] = new_task
        else:
            cols[name] = src_table.column(name)
    out = pa.table(cols)
    # Preserve HF metadata block.
    if src_table.schema.metadata:
        out = out.replace_schema_metadata(src_table.schema.metadata)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--src", required=True, type=Path,
                   help="Parent directory containing the source LeRobot datasets.")
    p.add_argument("--out", required=True, type=Path,
                   help="Output dataset directory (will be created / overwritten).")
    p.add_argument("--drop-last-pattern", default="*_bad_last",
                   help="fnmatch pattern; matching dirs drop their LAST parquet.")
    p.add_argument("--task", action="append", required=True, default=[],
                   metavar="COUNT:TEXT",
                   help="Repeatable; assigns the next COUNT episodes the TEXT task. "
                        "Use 'rest' as COUNT for the final catch-all entry.")
    p.add_argument("--chunks-size", type=int, default=1000,
                   help="LeRobot chunks_size (default 1000; affects info.json only "
                        "while episodes still go into chunk-000 for total <= chunks_size).")
    p.add_argument("--codebase-version", default="v2.1")
    p.add_argument("--robot-type", default="panda")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--image-stat-sample", type=int, default=100,
                   help="Frames per episode to sample for image stats (default 100).")
    p.add_argument("--overwrite", action="store_true",
                   help="Wipe --out before writing.")
    args = p.parse_args(argv)

    tasks = _parse_tasks(args.task)
    canonical_tasks = _canonical_task_map(tasks)
    datasets = _discover_datasets(args.src)
    if not datasets:
        raise SystemExit(f"no LeRobot datasets found under {args.src}")

    # Plan kept episodes per dataset.
    plan: list[tuple[Path, list[Path], bool]] = []
    total_kept = 0
    for ds in datasets:
        parquets = _episode_parquets(ds)
        bad = fnmatch.fnmatch(ds.name, args.drop_last_pattern)
        keep = parquets[:-1] if bad and parquets else parquets
        plan.append((ds, keep, bad))
        total_kept += len(keep)

    # Sanity-check task counts.
    spec_total = sum(t.count for t in tasks if t.count is not None)
    if tasks[-1].count is None:
        if spec_total > total_kept:
            raise SystemExit(
                f"sum of --task counts ({spec_total}) exceeds kept episodes "
                f"({total_kept}); 'rest' bucket would be negative"
            )
    else:
        if spec_total != total_kept:
            raise SystemExit(
                f"sum of --task counts ({spec_total}) != kept episodes ({total_kept}); "
                "add a final 'rest' entry or fix counts"
            )

    if args.overwrite and args.out.exists():
        shutil.rmtree(args.out)
    out_data = args.out / "data" / "chunk-000"
    out_meta = args.out / "meta"
    out_data.mkdir(parents=True, exist_ok=True)
    out_meta.mkdir(parents=True, exist_ok=True)

    print(f"[combine] {len(datasets)} source dirs → {total_kept} kept episodes")
    for ds, keep, bad in plan:
        flag = " (bad-last; dropped 1)" if bad else ""
        print(f"  {ds.name}: keeping {len(keep)} parquet(s){flag}")
    print()

    # Stream-write each episode.
    global_index = 0
    episodes_jsonl: list[dict] = []
    stats_jsonl: list[dict] = []
    template = None      # cached info.json features from the first source
    text_by_canonical = {v: k for k, v in canonical_tasks.items()}
    t0 = time.time()

    # Reuse-vs-recompute counters for the closing summary.
    n_reused = 0
    n_recomputed = 0

    global_ep = 0
    for ds, keep, _bad in plan:
        # Try to load this source dataset's precomputed per-episode stats
        # once; index into it by parquet position to match source ordering.
        src_stats = _load_source_stats(ds)
        for pos, src_parquet in enumerate(keep):
            src_table = pq.read_table(src_parquet)
            n = src_table.num_rows
            task_index = _task_index_for(global_ep, tasks, canonical_tasks, total_kept)
            new_table = _renumber_table(
                src_table,
                global_episode_index=global_ep,
                global_index_start=global_index,
                task_index=task_index,
            )

            out_path = out_data / f"episode_{global_ep:06d}.parquet"
            pq.write_table(new_table, out_path)

            # Episode's `tasks` list carries the text matching `task_index`;
            # look it up by canonical index (NOT by --task spec position).
            episodes_jsonl.append({
                "episode_index": global_ep,
                "tasks": [text_by_canonical[task_index]],
                "length": n,
            })

            # Reuse the source's precomputed stats when available; only
            # re-derive the three fields we actually renumber. Falls back
            # to a full recompute if the source lacks episodes_stats.jsonl
            # (or has fewer entries than parquets — unexpected, but safe).
            prior = (
                src_stats[pos]
                if src_stats is not None and pos < len(src_stats)
                else None
            )
            if prior is not None:
                stats = _renumber_stats(
                    prior,
                    n_rows=n,
                    global_episode_index=global_ep,
                    global_index_start=global_index,
                    task_index=task_index,
                )
                n_reused += 1
            else:
                stats = _episode_stats(new_table)
                n_recomputed += 1
            stats_jsonl.append({"episode_index": global_ep, "stats": stats})

            if template is None:
                # Pull the original info.json template once for the features block.
                info_path = ds / "meta" / "info.json"
                if info_path.exists():
                    template = json.loads(info_path.read_text())

            global_index += n
            global_ep += 1

    print(f"[stats] reused precomputed: {n_reused}  recomputed from pixels: {n_recomputed}")

    # Write meta files. tasks.jsonl carries ONE row per unique task text,
    # not one per --task spec (multiple specs may share a text).
    tasks_path = out_meta / "tasks.jsonl"
    with tasks_path.open("w") as f:
        for text, idx in sorted(canonical_tasks.items(), key=lambda kv: kv[1]):
            f.write(json.dumps({"task_index": idx, "task": text}) + "\n")

    episodes_path = out_meta / "episodes.jsonl"
    with episodes_path.open("w") as f:
        for ep in episodes_jsonl:
            f.write(json.dumps(ep) + "\n")

    stats_path = out_meta / "episodes_stats.jsonl"
    with stats_path.open("w") as f:
        for s in stats_jsonl:
            f.write(json.dumps(s) + "\n")

    # info.json. Reuse the source features block (schema-identical across our datasets).
    features = (template or {}).get("features") or {}
    info = {
        "codebase_version": args.codebase_version,
        "robot_type": args.robot_type,
        "total_episodes": total_kept,
        "total_frames": global_index,
        "total_tasks": len(canonical_tasks),
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": args.chunks_size,
        "fps": args.fps,
        "splits": {"train": f"0:{total_kept}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    (out_meta / "info.json").write_text(json.dumps(info, indent=4))

    elapsed = time.time() - t0
    print()
    print(f"[done] {total_kept} episodes, {global_index} frames written in {elapsed:.1f}s")
    print(f"       → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
