"""Sample center-gate episodes from the relabeled no_3pov datasets and dump
forward+wrist GIFs + the task string so we can confirm the side-specific
labels match the rendered direction.

Reads:
    data/no_3pov/<dataset>/data/chunk-000/episode_*.parquet

Writes:
    runs/inspect/center_relabel_check/<dataset>/episode_<NN>/
        forward.gif
        wrist.gif
        info.txt
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO / "runs/inspect/center_relabel_check"

# Two from each side per dataset that has center episodes.
# center_no_3pov: eps 0,25 → CL · eps 50,75 → CR
# real_center_no_3pov: eps 100,125 → CL · eps 150,175 → CR
# all_no_3pov: eps 100,125 → CL · eps 150,175 → CR (synth-left/right ranges 200-299 don't need re-checking)
SAMPLES = {
    "gate_scenes_center_no_3pov":     [0, 25, 50, 75],
    "gate_scenes_real_center_no_3pov":[100, 125, 150, 175],
    "gate_scenes_all_no_3pov":        [100, 125, 150, 175],
}


def episode_task(meta_dir: Path, ep: int) -> str:
    with (meta_dir / "episodes.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            if int(r["episode_index"]) == ep:
                tasks = r.get("tasks", [])
                return tasks[0] if tasks else ""
    return ""


def decode_png(blob: dict) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(blob["bytes"])))


def write_gif(frames, path: Path, fps: int = 10):
    if not frames:
        return
    pil = [Image.fromarray(f) for f in frames]
    pil[0].save(
        path, save_all=True,
        append_images=pil[1:] if len(pil) > 1 else [],
        duration=int(1000 / fps), loop=0, optimize=False, disposal=2,
    )


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Sampling center-gate episodes → {OUT_ROOT}\n")
    for dataset, eps in SAMPLES.items():
        ds = REPO / "data/no_3pov" / dataset
        meta = ds / "meta"
        chunks = ds / "data/chunk-000"
        for ep in eps:
            pq_path = chunks / f"episode_{ep:06d}.parquet"
            t = pq.read_table(pq_path)
            task_idx = int(t["task_index"][0].as_py())
            task_str = episode_task(meta, ep)
            state = np.stack(t["state"].to_numpy())
            xyz = state[:, :3]

            out = OUT_ROOT / dataset / f"episode_{ep:06d}"
            out.mkdir(parents=True, exist_ok=True)
            fwd = [decode_png(b) for b in t["image"].to_pylist()]
            wrist = [decode_png(b) for b in t["wrist_image"].to_pylist()]
            write_gif(fwd, out / "forward.gif")
            write_gif(wrist, out / "wrist.gif")

            # Determine expected side from y-range — center_from_left tends to
            # stay y > -0.85, center_from_right overshoots to y < -1.0.
            y_min = float(xyz[:, 1].min())
            expected_side = "left" if y_min > -0.95 else "right"
            label_side = "left" if "from the left" in task_str else (
                          "right" if "from the right" in task_str else "?")
            verdict = "✓" if expected_side == label_side else "✗ MISMATCH"

            (out / "info.txt").write_text(
                f"dataset:       {dataset}\n"
                f"episode_index: {ep}\n"
                f"task_index:    {task_idx}\n"
                f"task:          {task_str}\n"
                f"state y range: [{xyz[:, 1].min():.3f}, {xyz[:, 1].max():.3f}]\n"
                f"  → trajectory shape suggests center-gate from the {expected_side}\n"
                f"  → label says                        center-gate from the {label_side}\n"
                f"  → {verdict}\n"
            )
            print(f"  {dataset}/episode_{ep:06d}: task_idx={task_idx}  "
                  f"y∈[{xyz[:,1].min():.2f},{xyz[:,1].max():.2f}]  "
                  f"label={label_side}  shape→{expected_side}  {verdict}")
    print(f"\n[done] GIFs + info.txt under {OUT_ROOT}")


if __name__ == "__main__":
    main()
