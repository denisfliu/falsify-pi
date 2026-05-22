"""Convert each LeRobot v2.1 dataset under data/no_3pov/ to the v3.0 layout
used in ../SousVide/scripts/build_recovery_dataset.py.

Schema changes the converter applies per parquet:

  v2.1 column        →  v3.0 column
  ───────────────       ────────────────────────────
  image              →  observation.images.image
  wrist_image        →  observation.images.wrist_image
  3pov_1             →  observation.images.3pov_1
  state              →  observation.state
  actions            →  action                       ← rename (singular)
  timestamp (f32)    →  timestamp (f64)              ← widened
  frame_index        →  frame_index                  ← unchanged
  episode_index      →  episode_index                ← unchanged
  index              →  index                        ← unchanged
  task_index         →  task_index                   ← unchanged

Filenames go from ``episode_NNNNNN.parquet`` → ``episode-NNNNNN.parquet``
(hyphen, matching SousVide's gate_nav_synthetic / build_recovery_dataset.py).

Meta files are restructured:

  v2.1                          →  v3.0
  ─────────────────────────────    ──────────────────────────────────
  meta/info.json                →  meta/info.json   (new feature schema, v3.0)
  meta/episodes.jsonl           →  (dropped)
  meta/episodes_stats.jsonl     →  meta/stats.json  (action-only summary)
  meta/tasks.jsonl              →  meta/tasks.parquet
  meta/custom_metadata.csv      →  meta/custom_metadata.csv (carried over,
                                                              if present)

Run from repo root:
    .venv/bin/python scripts/convert_no_3pov_to_v3.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO / "data/no_3pov"
OUT_ROOT = REPO / "data/no_3pov_v3"

# v2 → v3 column rename map (others kept as-is).
COL_RENAME = {
    "image": "observation.images.image",
    "wrist_image": "observation.images.wrist_image",
    "3pov_1": "observation.images.3pov_1",
    "state": "observation.state",
    "actions": "action",
}

# Image features keys (v3 names).
IMAGE_KEYS = ("observation.images.image",
              "observation.images.wrist_image",
              "observation.images.3pov_1")


def convert_parquet(src: Path, dst: Path) -> int:
    """Read a v2.1 parquet, rewrite columns/dtypes per v3, write to dst."""
    t = pq.read_table(src)
    n = t.num_rows

    cols: dict[str, pa.Array] = {}
    for src_name in t.column_names:
        dst_name = COL_RENAME.get(src_name, src_name)
        col = t[src_name]
        if dst_name == "timestamp":
            # f32 → f64 (and unwrap from FixedSizeList<1> if present).
            try:
                arr = col.to_numpy(zero_copy_only=False).astype(np.float64)
            except Exception:
                arr = np.asarray([v.as_py() if hasattr(v, "as_py") else v
                                  for v in col]).astype(np.float64)
            cols[dst_name] = pa.array(arr.ravel().tolist(), type=pa.float64())
        else:
            cols[dst_name] = col
    new_table = pa.table(cols)
    pq.write_table(new_table, dst)
    return n


def make_v3_info(image_shape: tuple[int, int, int],
                 image_keys: list[str],
                 state_dim: int,
                 action_dim: int,
                 fps: int,
                 total_episodes: int,
                 total_frames: int,
                 total_tasks: int) -> dict:
    f = {}
    for k in image_keys:
        f[k] = {
            "dtype": "image",
            "shape": list(image_shape),
            "names": ["height", "width", "channel"],
            "fps": int(fps),
        }
    f["observation.state"] = {
        "dtype": "float32", "shape": [int(state_dim)],
        "names": ["observation.state"], "fps": int(fps),
    }
    f["action"] = {
        "dtype": "float32", "shape": [int(action_dim)],
        "names": ["action"], "fps": int(fps),
    }
    f["timestamp"] = {
        "dtype": "float32", "shape": [1], "names": None, "fps": int(fps),
    }
    for col in ("frame_index", "episode_index", "index", "task_index"):
        f[col] = {"dtype": "int64", "shape": [1], "names": None, "fps": int(fps)}
    return {
        "codebase_version": "v3.0",
        "robot_type": "panda",
        "total_episodes": int(total_episodes),
        "total_frames": int(total_frames),
        "total_tasks": int(total_tasks),
        "chunks_size": 1000,
        "fps": int(fps),
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/episode-{episode_index:06d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": f,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
    }


def collect_action_stats(out_chunk: Path, total_frames: int):
    """Streaming min/max/mean over the `action` column (now float32 list)."""
    a_min = None
    a_max = None
    a_sum = None
    n_total = 0
    for pq_path in sorted(out_chunk.glob("episode-*.parquet")):
        t = pq.read_table(pq_path, columns=["action"])
        arr = np.stack(t["action"].to_numpy())   # (N, action_dim)
        if a_min is None:
            a_min = arr.min(axis=0)
            a_max = arr.max(axis=0)
            a_sum = arr.sum(axis=0).astype(np.float64)
        else:
            a_min = np.minimum(a_min, arr.min(axis=0))
            a_max = np.maximum(a_max, arr.max(axis=0))
            a_sum += arr.sum(axis=0)
        n_total += arr.shape[0]
    a_mean = (a_sum / max(n_total, 1)).astype(np.float32)
    return {
        "action": {
            "min": a_min.tolist(), "max": a_max.tolist(),
            "mean": a_mean.tolist(),
        }
    }


def convert_dataset(src_ds: Path, out_ds: Path) -> None:
    print(f"=== {src_ds.name} → {out_ds} ===")
    if out_ds.exists():
        shutil.rmtree(out_ds)
    out_data = out_ds / "data" / "chunk-000"
    out_meta = out_ds / "meta"
    out_data.mkdir(parents=True)
    out_meta.mkdir(parents=True)

    src_chunk = src_ds / "data" / "chunk-000"
    src_parquets = sorted(src_chunk.glob("episode_*.parquet"))

    # Convert each parquet (rename + dtype widen).
    total_frames = 0
    sample_state = sample_action = None
    sample_image_shape = None
    present_image_keys: list[str] = []
    for src in src_parquets:
        ep = int(src.stem.split("_")[-1])
        dst = out_data / f"episode-{ep:06d}.parquet"
        n = convert_parquet(src, dst)
        total_frames += n
        if sample_state is None:
            t = pq.read_table(dst)
            sample_state = len(t["observation.state"][0].as_py())
            sample_action = len(t["action"][0].as_py())
            present_image_keys = [
                k for k in IMAGE_KEYS if k in t.column_names
            ]
            # PNG dims by decoding one frame from the first image column.
            import io
            from PIL import Image
            blob = t[present_image_keys[0]][0].as_py()
            arr = np.asarray(Image.open(io.BytesIO(blob["bytes"])))
            sample_image_shape = tuple(arr.shape)
    n_eps = len(src_parquets)
    print(f"  converted {n_eps} parquets · {total_frames} frames · "
          f"state[{sample_state}] · action[{sample_action}] · "
          f"image {sample_image_shape} · image_keys={[k.split('.')[-1] for k in present_image_keys]}")

    # tasks.parquet from v2.1 tasks.jsonl.
    tasks_jsonl = src_ds / "meta" / "tasks.jsonl"
    tasks: list[dict] = []
    with tasks_jsonl.open() as f:
        for line in f:
            tasks.append(json.loads(line))
    tasks_table = pa.table({
        "task_index": [int(t["task_index"]) for t in tasks],
        "__index_level_0__": [t["task"] for t in tasks],
    })
    pq.write_table(tasks_table, out_meta / "tasks.parquet")
    print(f"  wrote tasks.parquet ({len(tasks)} tasks)")

    # info.json — pull fps from old info, use detected dims.
    old_info = json.loads((src_ds / "meta" / "info.json").read_text())
    fps = int(old_info.get("fps", 10))
    info = make_v3_info(
        image_shape=sample_image_shape, image_keys=present_image_keys,
        state_dim=sample_state, action_dim=sample_action, fps=fps,
        total_episodes=n_eps, total_frames=total_frames,
        total_tasks=len(tasks),
    )
    (out_meta / "info.json").write_text(json.dumps(info, indent=4) + "\n")
    print(f"  wrote info.json (v3.0)")

    # stats.json — action-only, computed from converted parquets.
    stats = collect_action_stats(out_data, total_frames)
    (out_meta / "stats.json").write_text(json.dumps(stats, indent=4) + "\n")
    print(f"  wrote stats.json (action min/max/mean)")

    # Carry over the validator sidecar if present.
    src_csv = src_ds / "meta" / "custom_metadata.csv"
    if src_csv.exists():
        shutil.copy2(src_csv, out_meta / "custom_metadata.csv")
        print(f"  copied custom_metadata.csv")


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Convert one or more LeRobot v2.1 datasets to v3.0 layout.",
        epilog=(
            "Default (no args): converts every dataset under data/no_3pov/ "
            "into data/no_3pov_v3/<name>/. Use --dataset to convert a "
            "single bundle from anywhere (e.g. data/atomic_datasets/<name>) "
            "into --out (or data/no_3pov_v3/<name> by default)."
        ),
    )
    ap.add_argument("--dataset", type=Path, default=None,
                    help="Single v2.1 dataset directory to convert. "
                         "Mutually exclusive with --src-root.")
    ap.add_argument("--src-root", type=Path, default=None,
                    help="Parent directory containing v2.1 datasets to "
                         "convert as a batch (default: data/no_3pov/).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output path. With --dataset: the destination "
                         "v3 dataset dir. With --src-root: the parent "
                         "directory holding the converted v3 datasets "
                         "(default: data/no_3pov_v3/).")
    args = ap.parse_args()

    if args.dataset is not None and args.src_root is not None:
        raise SystemExit("--dataset and --src-root are mutually exclusive")

    if args.dataset is not None:
        src = args.dataset.resolve()
        out = (args.out or (OUT_ROOT / src.name)).resolve()
        print(f"converting single dataset: {src} → {out}")
        convert_dataset(src, out)
        print(f"[done] → {out}")
        return

    src_root = (args.src_root or SRC_ROOT).resolve()
    out_root = (args.out or OUT_ROOT).resolve()
    datasets = sorted(d for d in src_root.iterdir() if d.is_dir())
    print(f"converting {len(datasets)} dataset(s) under {src_root}")
    for ds in datasets:
        out = out_root / ds.name
        convert_dataset(ds, out)
        print()
    print(f"[done] all conversions under {out_root}")


if __name__ == "__main__":
    main()
