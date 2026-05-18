"""Sample 2 parquets from each atomic dataset, dump forward+wrist GIFs and
the task string so the user can eyeball trajectory diversity.

Reads:
    data/atomic_datasets/<dataset>/data/chunk-000/episode_*.parquet
    data/atomic_datasets/<dataset>/meta/episodes.jsonl

Writes:
    runs/inspect/atomic_traj_check/<dataset>/episode_<NN>/
        forward.gif           — animated forward camera (10 fps)
        wrist.gif             — animated downward (wrist) camera
        info.txt              — task string, frame count, state bbox

Pick two episodes per dataset by spacing them across the index range so
they exercise different task buckets (for the real dataset which spans
left+right) or different perturbation samples (for the synths).
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "data/atomic_datasets"
OUT_ROOT = REPO / "runs/inspect/atomic_traj_check"

# Two-episode samples per atomic dataset.
SAMPLES = {
    "gate_scenes_real_combined": [0, 75],    # spans left-gate and right-gate tasks
    "synth_left_gate":           [0, 25],
    "synth_right_gate":          [0, 25],
    "synth_center_from_left":    [0, 25],
    "synth_center_from_right":   [0, 25],
}


def episode_task(meta_dir: Path, episode_index: int) -> str:
    with (meta_dir / "episodes.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            if int(r["episode_index"]) == episode_index:
                tasks = r.get("tasks", [])
                return tasks[0] if tasks else ""
    return ""


def decode_png(blob: dict) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(blob["bytes"])))


def write_gif(frames: list[np.ndarray], path: Path, fps: int = 10):
    if not frames:
        return
    pil_frames = [Image.fromarray(f) for f in frames]
    duration_ms = int(1000 / fps)
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:] if len(pil_frames) > 1 else [],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Sampling 2 episodes per atomic dataset → {OUT_ROOT}")
    print()

    for dataset, eps in SAMPLES.items():
        ds_dir = ROOT / dataset
        meta_dir = ds_dir / "meta"
        chunks = ds_dir / "data" / "chunk-000"
        if not chunks.exists():
            print(f"  [skip] {dataset}: no data/chunk-000")
            continue
        for ep in eps:
            pq_path = chunks / f"episode_{ep:06d}.parquet"
            if not pq_path.exists():
                print(f"  [skip] {dataset} ep {ep}: parquet not found")
                continue

            t = pq.read_table(pq_path)
            n = t.num_rows
            task = episode_task(meta_dir, ep)

            ep_out = OUT_ROOT / dataset / f"episode_{ep:06d}"
            ep_out.mkdir(parents=True, exist_ok=True)

            fwd_blobs = t["image"].to_pylist()
            wrist_blobs = t["wrist_image"].to_pylist()
            fwd_frames = [decode_png(b) for b in fwd_blobs]
            wrist_frames = [decode_png(b) for b in wrist_blobs]
            write_gif(fwd_frames, ep_out / "forward.gif")
            write_gif(wrist_frames, ep_out / "wrist.gif")

            # State bbox so the user can compare trajectories numerically.
            state = np.stack(t["state"].to_numpy())
            xyz = state[:, :3]
            yaw = state[:, 3]
            bbox = (xyz.min(0), xyz.max(0))
            info = ep_out / "info.txt"
            info.write_text(
                f"dataset:       {dataset}\n"
                f"episode_index: {ep}\n"
                f"frames:        {n}\n"
                f"task:          {task}\n"
                f"state bbox x:  [{bbox[0][0]:7.3f}, {bbox[1][0]:7.3f}]\n"
                f"state bbox y:  [{bbox[0][1]:7.3f}, {bbox[1][1]:7.3f}]\n"
                f"state bbox z:  [{bbox[0][2]:7.3f}, {bbox[1][2]:7.3f}]\n"
                f"yaw range:     [{yaw.min():7.3f}, {yaw.max():7.3f}]\n"
                f"state[0]:      {state[0].tolist()}\n"
                f"state[-1]:     {state[-1].tolist()}\n"
            )
            print(f"  {dataset}/episode_{ep:06d}: {n} frames")
            print(f"    task: {task[:70]}{'...' if len(task)>70 else ''}")
            print(f"    bbox x={bbox[0][0]:.2f}..{bbox[1][0]:.2f}  y={bbox[0][1]:.2f}..{bbox[1][1]:.2f}  z={bbox[0][2]:.2f}..{bbox[1][2]:.2f}")

    print()
    print(f"[done] GIFs + info.txt under {OUT_ROOT}")


if __name__ == "__main__":
    main()
