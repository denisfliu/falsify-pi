"""Undistort the real-camera images of a LeRobot v2.1 dataset to pinhole.

The real drone cameras are near-equidistant fisheyes (OpenCV KB4 model;
calibration lives in the drone-frame YAML as ``cameras.<name>.real_distortion``
next to the pinhole ``intrinsics``). The falsify gsplat renderer is a
distortion-free pinhole, so instead of warping every sim render to fisheye we
rectify the real footage once: each image column is remapped through
``cv2.fisheye.initUndistortRectifyMap`` with the new camera matrix P set equal
to the (resolution-scaled) calibrated K. Because the equidistant projection
compresses radially relative to tan-projection, P = K samples strictly inside
the source image — no invalid black corners — at the cost of a modest
effective-FoV shrink at the edges.

The calibration K is given at the native sensor resolution (e.g. 1024x768);
stored dataset images are typically resized (e.g. 256x256, anisotropic). K is
scaled per-axis to the stored resolution; the KB4 coefficients act on
normalized coordinates and are unchanged by resizing.

With ``--to-rgb`` the mapped image columns are also channel-swapped BGR→RGB
during the same pass. This is the canonical converter for the repo-wide
**RGB + pinhole** image convention (2026-06-12): legacy real datasets hold
BGR bytes labeled RGB (cv2 collector) at raw fisheye; new-convention datasets
hold true RGB rectified to the calibrated pinhole K. The default output
suffix is ``_rgb`` when ``--to-rgb`` is set, else ``_undistorted``.

Output is a sibling dataset directory: ``meta/`` is copied verbatim, every
parquet is rewritten with the mapped image columns undistorted and all other
columns byte-identical. Note that any image pixel statistics in
``meta/episodes_stats.jsonl`` are NOT recomputed. Unmapped image columns
(e.g. ``3pov_1``) are copied untouched — including their channel order.

Usage::

    PYTHONPATH=src python scripts/dataset/undistort_fisheye_dataset.py \\
        --dataset data/atomic_datasets/gate_scenes_real_combined \\
        --to-rgb \\
        --column image=forward --column wrist_image=downward
"""

from __future__ import annotations

import argparse
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True,
                   help="LeRobot v2.1 dataset root (contains data/ + meta/).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output dataset root. Default: <dataset>_rgb with "
                        "--to-rgb, else <dataset>_undistorted.")
    p.add_argument("--to-rgb", action="store_true",
                   help="Also swap the mapped columns' bytes BGR→RGB "
                        "(new-convention datasets store true RGB).")
    p.add_argument("--frame", type=Path,
                   default=REPO_ROOT / "configs/frames/carl_dual.yaml",
                   help="Drone-frame YAML carrying intrinsics + real_distortion.")
    p.add_argument("--column", action="append", default=None,
                   metavar="PARQUET_COL=CAMERA_NAME",
                   help="Image column → frame-YAML camera mapping. Repeatable. "
                        "Default: image=forward wrist_image=downward. Columns "
                        "not listed (e.g. 3pov_1) are copied untouched.")
    p.add_argument("--workers", type=int, default=8,
                   help="Threads for per-frame decode/remap/encode.")
    p.add_argument("--overwrite", action="store_true",
                   help="Allow writing into an existing --out directory.")
    return p.parse_args()


def _build_remap(cam_yaml: dict, stored_w: int, stored_h: int):
    """Return (map1, map2) rectifying a stored-resolution fisheye image to
    the pinhole K (scaled to the same stored resolution)."""
    intr = cam_yaml["intrinsics"]
    dist = cam_yaml.get("real_distortion")
    if dist is None:
        raise KeyError("camera has no real_distortion block in the frame YAML")
    if dist["model"] != "opencv_fisheye":
        raise ValueError(f"unsupported distortion model {dist['model']!r}")
    sx = stored_w / intr["width"]
    sy = stored_h / intr["height"]
    K = np.array([
        [intr["fx"] * sx, 0.0, intr["cx"] * sx],
        [0.0, intr["fy"] * sy, intr["cy"] * sy],
        [0.0, 0.0, 1.0],
    ])
    D = np.asarray(dist["coeffs"], dtype=np.float64).reshape(4, 1)
    return cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K, (stored_w, stored_h), cv2.CV_16SC2)


def _undistort_png(png_bytes: bytes, maps, swap_channels: bool) -> bytes:
    img = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    out = cv2.remap(img, maps[0], maps[1], interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT)
    if swap_channels:
        out = out[..., ::-1]
    ok, buf = cv2.imencode(".png", out)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()


def main() -> None:
    args = _parse_args()
    t0 = time.time()
    dataset = args.dataset.resolve()
    suffix = "_rgb" if args.to_rgb else "_undistorted"
    out_root = (args.out or dataset.parent / f"{dataset.name}{suffix}").resolve()
    if out_root.exists() and not args.overwrite:
        raise SystemExit(f"{out_root} exists — pass --overwrite to replace")

    col_map = {}
    for tok in (args.column or ["image=forward", "wrist_image=downward"]):
        col, cam = tok.split("=", 1)
        col_map[col] = cam

    frame_cfg = yaml.safe_load(args.frame.read_text())
    cams = frame_cfg["cameras"]
    for col, cam in col_map.items():
        if cam not in cams:
            raise SystemExit(f"camera {cam!r} (for column {col!r}) not in {args.frame}")

    # meta/ verbatim
    out_root.mkdir(parents=True, exist_ok=True)
    if (out_root / "meta").exists():
        shutil.rmtree(out_root / "meta")
    shutil.copytree(dataset / "meta", out_root / "meta")

    parquets = sorted(dataset.glob("data/chunk-*/episode_*.parquet"))
    if not parquets:
        raise SystemExit(f"no episode parquets under {dataset}/data/")
    print(f"undistorting {len(parquets)} episodes "
          f"({', '.join(f'{c}←{m}' for c, m in col_map.items())}) → {out_root}")

    maps_by_cam: dict[str, tuple] = {}   # cam name → (map1, map2), built lazily
    pool = ThreadPoolExecutor(max_workers=args.workers)
    n_frames = 0
    for ep_i, src in enumerate(parquets):
        table = pq.read_table(src)
        for col, cam in col_map.items():
            structs = table.column(col).to_pylist()
            if cam not in maps_by_cam:
                h, w = cv2.imdecode(
                    np.frombuffer(structs[0]["bytes"], np.uint8),
                    cv2.IMREAD_UNCHANGED).shape[:2]
                maps_by_cam[cam] = _build_remap(cams[cam], w, h)
                print(f"  {cam}: stored {w}x{h}, remap built")
            maps = maps_by_cam[cam]
            new_bytes = list(pool.map(
                lambda s: _undistort_png(s["bytes"], maps, args.to_rgb),
                structs))
            new_structs = [
                {"bytes": b, "path": s.get("path")}
                for b, s in zip(new_bytes, structs)
            ]
            idx = table.schema.get_field_index(col)
            arr = pa.array(new_structs, type=table.schema.field(idx).type)
            table = table.set_column(idx, table.schema.field(idx), arr)
        dst = out_root / src.relative_to(dataset)
        dst.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, dst)
        n_frames += table.num_rows
        if (ep_i + 1) % 10 == 0 or ep_i == len(parquets) - 1:
            print(f"  {ep_i + 1}/{len(parquets)} episodes "
                  f"({n_frames} frames, {time.time() - t0:.0f}s)")
    pool.shutdown()
    print(f"DONE in {time.time() - t0:.1f}s → {out_root}")
    print("note: meta/episodes_stats.jsonl image stats were copied, not recomputed")


if __name__ == "__main__":
    main()
