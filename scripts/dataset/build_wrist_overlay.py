"""Build the wrist-camera gripper-occlusion RGBA overlay from a real dataset.

Implements the documented recipe from ``configs/embodiments/assets/README.md``:
the drone's own struts/gripper occlude a static region of the downward
("wrist") camera; we estimate that region from real footage and bake it into
an RGBA overlay that ``CameraPostprocess.apply`` composites onto every sim
render so sim and real share the same static occlusion.

Recipe:

1. Sample N frames (``--episodes`` evenly-spaced episodes x ``--frames-per-ep``
   evenly-spaced frames) of ``--column``.
2. Per-pixel median across the sample -> ``med``.
3. Median luminance ``lum = 0.299*R + 0.587*G + 0.114*B`` (the dataset must
   store true RGB — i.e. a new-convention ``*_rgb`` dataset).
4. Hard mask: ``(lum < thr) & (y < H/2)``; drop connected components
   < ``--min-component`` px; ``binary_fill_holes``.
5. Alpha = ``gaussian_filter(mask * 255, sigma)``.
6. RGB layer = ``med`` inside the mask, black outside.

The output overlay's channel order equals the dataset's stored order — author
it against a dataset matching the ``channel_order`` it will be composited
under (RGB for the canonical embodiment).

Usage::

    PYTHONPATH=src python scripts/dataset/build_wrist_overlay.py \\
        --dataset data/atomic_datasets/gate_scenes_real_combined_rgb \\
        --out configs/embodiments/assets/carl_wrist_overlay_pinhole_rgb.png
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from scipy import ndimage


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True,
                   help="LeRobot v2.1 dataset root (true-RGB convention).")
    p.add_argument("--column", default="wrist_image")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--frames-per-ep", type=int, default=30)
    p.add_argument("--thr", type=float, default=60.0,
                   help="Median-luminance threshold for the occlusion mask.")
    p.add_argument("--sigma", type=float, default=1.2,
                   help="Gaussian alpha-edge softness (px).")
    p.add_argument("--min-component", type=int, default=20,
                   help="Drop mask components smaller than this many px.")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    parquets = sorted(args.dataset.glob("data/chunk-*/episode_*.parquet"))
    if not parquets:
        raise SystemExit(f"no episode parquets under {args.dataset}/data/")
    ep_idx = np.unique(np.linspace(0, len(parquets) - 1, args.episodes).astype(int))

    frames = []
    for i in ep_idx:
        structs = pq.read_table(parquets[i], columns=[args.column]).column(
            args.column).to_pylist()
        fr_idx = np.unique(np.linspace(0, len(structs) - 1,
                                       args.frames_per_ep).astype(int))
        for j in fr_idx:
            frames.append(np.array(
                Image.open(io.BytesIO(structs[j]["bytes"])).convert("RGB")))
    stack = np.stack(frames)
    print(f"sampled {len(frames)} frames from {len(ep_idx)} episodes "
          f"({stack.shape[1]}x{stack.shape[2]})")

    med = np.median(stack, axis=0).astype(np.uint8)
    lum = 0.299 * med[..., 0] + 0.587 * med[..., 1] + 0.114 * med[..., 2]
    H, W = lum.shape
    mask = (lum < args.thr) & (np.arange(H)[:, None] < H / 2)

    labels, n = ndimage.label(mask)
    sizes = ndimage.sum_labels(np.ones_like(labels), labels, range(1, n + 1))
    keep = {k + 1 for k, s in enumerate(sizes) if s >= args.min_component}
    mask = np.isin(labels, list(keep))
    mask = ndimage.binary_fill_holes(mask)

    alpha = ndimage.gaussian_filter(mask.astype(np.float32) * 255.0,
                                    sigma=args.sigma)
    alpha = np.clip(alpha, 0, 255).astype(np.uint8)
    rgb = np.where(mask[..., None], med, 0).astype(np.uint8)

    overlay = np.dstack([rgb, alpha])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay, "RGBA").save(args.out)
    coverage = mask.mean() * 100.0
    print(f"mask coverage {coverage:.1f}% of frame "
          f"(components kept: {len(keep)}/{n})")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
