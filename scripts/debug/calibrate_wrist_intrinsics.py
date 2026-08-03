"""Feature-matching wrist-camera intrinsic calibration.

Given:
- N cached pinhole renders at a wide source FoV (e.g. 130°, 512x512) along
  a real teleop trajectory.
- N real wrist_image frames from the matching poses in the original parquet.

For each frame pair, detect SIFT keypoints in both and match. Each match
gives a correspondence: a 3-D ray (from the sim render — known
intrinsics) that projects to a known real pixel. Stack matches across
frames and fit a pinhole + Brown-Conrady radial distortion model
(parameters ``fx_out``, ``k1``, ``k2``, ``cx_out``, ``cy_out``) by
least-squares on the reprojection error.

This avoids the photometric trap where SSIM converges to a degenerate
uniform-color minimum: SIFT is illumination-invariant, so it finds the
gate corners, table edges, etc. that ARE the same geometric features
between sim and real even when the colors differ.

Output:
- best-fit parameters (printed + JSON next to the grid)
- side-by-side grid (REAL | pinhole-source | initial fit | optimized fit)
- a "matches per frame" diagnostic

Usage::

    source tools/env.sh
    PYTHONPATH=src:external/FiGS/src:external/splatnav \\
      .venv/bin/python scripts/debug/calibrate_wrist_intrinsics.py \\
        --scene configs/scenes/left_gate.yaml \\
        --episode data/atomic_datasets/gate_scenes_real_combined/data/chunk-000/episode_000000.parquet \\
        --source-fov-deg 130 \\
        --source-size 512 \\
        --stride 12 \\
        --out /tmp/wrist_calib.png
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import least_squares


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_SIZE = 256


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--episode", type=Path, required=True)
    p.add_argument("--stride", type=int, default=12)
    p.add_argument("--source-fov-deg", type=float, default=130.0)
    p.add_argument("--source-size", type=int, default=512)
    p.add_argument("--cache-dir", type=Path,
                   default=REPO_ROOT / "runs/intrinsics_cache")
    p.add_argument("--init-fx", type=float, default=90.0,
                   help="Initial fx_out for the real-camera model "
                        "(rough: OUT_SIZE/2/tan(real_fov/2)).")
    p.add_argument("--lowe-ratio", type=float, default=0.6,
                   help="Lowe's ratio for SIFT match acceptance. Lower = stricter.")
    p.add_argument("--min-matches-per-frame", type=int, default=4)
    p.add_argument("--ransac-iters", type=int, default=200,
                   help="RANSAC iterations for inlier selection.")
    p.add_argument("--ransac-thresh", type=float, default=6.0,
                   help="Reprojection inlier threshold (px).")
    p.add_argument("--bounds-fx-min", type=float, default=50.0)
    p.add_argument("--bounds-fx-max", type=float, default=250.0)
    p.add_argument("--out", type=Path, default=Path("/tmp/wrist_calib.png"))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Brown-Conrady forward model (world ray → distorted pixel)
# ---------------------------------------------------------------------------


def project_distorted(
    x_norm: np.ndarray, y_norm: np.ndarray,
    fx: float, cx: float, cy: float, k1: float, k2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Brown-Conrady forward: undistorted normalized → distorted pixel.

    x_d = x * (1 + k1*r² + k2*r^4),  u = fx*x_d + cx.
    Assumes square pixels (fy = fx).
    """
    r2 = x_norm * x_norm + y_norm * y_norm
    f = 1.0 + k1 * r2 + k2 * r2 * r2
    x_d = x_norm * f
    y_d = y_norm * f
    u = fx * x_d + cx
    v = fx * y_d + cy
    return u, v


def warp_with_model(
    source_pinhole: np.ndarray,
    *,
    src_fx: float, src_cx: float,
    out_fx: float, out_cx: float, out_cy: float,
    k1: float, k2: float,
    out_size: int = OUT_SIZE,
    n_iter: int = 5,
) -> np.ndarray:
    """Render a real-camera-model image FROM a pinhole render. Used only for
    visualization; not on the inner-loop cost path."""
    H, W = source_pinhole.shape[:2]
    uu, vv = np.meshgrid(
        np.arange(out_size, dtype=np.float32),
        np.arange(out_size, dtype=np.float32),
    )
    x_d = (uu - out_cx) / out_fx
    y_d = (vv - out_cy) / out_fx

    x_u, y_u = x_d.copy(), y_d.copy()
    for _ in range(n_iter):
        r2 = x_u * x_u + y_u * y_u
        f = 1.0 + k1 * r2 + k2 * r2 * r2
        x_u = x_d / np.maximum(f, 1e-6)
        y_u = y_d / np.maximum(f, 1e-6)

    u_src = src_fx * x_u + src_cx
    v_src = src_fx * y_u + src_cx

    u0 = np.floor(u_src).astype(np.int32)
    v0 = np.floor(v_src).astype(np.int32)
    u1, v1 = u0 + 1, v0 + 1
    ib = (u0 >= 0) & (u1 < W) & (v0 >= 0) & (v1 < H)
    u0 = np.clip(u0, 0, W - 1); u1 = np.clip(u1, 0, W - 1)
    v0 = np.clip(v0, 0, H - 1); v1 = np.clip(v1, 0, H - 1)
    du = (u_src - u0).astype(np.float32)[..., None]
    dv = (v_src - v0).astype(np.float32)[..., None]
    src = source_pinhole.astype(np.float32)
    top = src[v0, u0] * (1 - du) + src[v0, u1] * du
    bot = src[v1, u0] * (1 - du) + src[v1, u1] * du
    out = top * (1 - dv) + bot * dv
    out[~ib] = 0
    return out.astype(np.uint8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    t0 = time.time()

    # 1) Load cached source pinhole renders + matching real frames
    cache_key = (
        f"{args.scene.stem}_ep{args.episode.stem}_str{args.stride}"
        f"_size{args.source_size}_fov{int(args.source_fov_deg)}"
    )
    cache_dir = args.cache_dir / cache_key
    if not cache_dir.exists():
        raise SystemExit(
            f"No cached renders at {cache_dir}. Run "
            f"scripts/debug/render_real_trajectory.py first."
        )

    print(f"[1/5] loading cache {cache_dir.name}")
    table = pq.read_table(args.episode)
    indices = list(range(0, table.num_rows, args.stride))
    pinhole = [np.array(Image.open(cache_dir / f"frame_{i:05d}.png"))
               for i in indices]
    wrist_bytes = table.column("wrist_image").to_pylist()
    real_imgs = [np.array(Image.open(io.BytesIO(wrist_bytes[i]["bytes"])).convert("RGB"))
                 for i in indices]
    print(f"      {len(pinhole)} frames, sim {pinhole[0].shape}, real {real_imgs[0].shape}")

    src_fx = args.source_size / (2.0 * np.tan(np.deg2rad(args.source_fov_deg) / 2.0))
    src_cx = (args.source_size - 1) / 2.0

    # 2) SIFT feature matching across all frames
    print("[2/5] SIFT detection + matching…")
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()
    all_matches: list[tuple[float, float, float, float]] = []  # (u_sim, v_sim, u_real, v_real)
    matches_per_frame: list[int] = []
    # Mask out the gripper region (top half) from the REAL image — the gripper
    # is static and would dominate the matches with self-correspondences that
    # tell us nothing about the cam intrinsics.
    real_gripper_mask = np.zeros((OUT_SIZE, OUT_SIZE), dtype=np.uint8)
    real_gripper_mask[OUT_SIZE // 2:, :] = 255
    for k, (pin, real) in enumerate(zip(pinhole, real_imgs)):
        pin_gray = cv2.cvtColor(pin, cv2.COLOR_RGB2GRAY)
        real_gray = cv2.cvtColor(real, cv2.COLOR_RGB2GRAY)
        kp_s, des_s = sift.detectAndCompute(pin_gray, None)
        kp_r, des_r = sift.detectAndCompute(real_gray, real_gripper_mask)
        if des_s is None or des_r is None or len(des_s) < 4 or len(des_r) < 4:
            matches_per_frame.append(0)
            continue
        knn = bf.knnMatch(des_s, des_r, k=2)
        good = [m for pair in knn if len(pair) == 2
                for m, n in [pair] if m.distance < args.lowe_ratio * n.distance]
        matches_per_frame.append(len(good))
        for m in good:
            u_s, v_s = kp_s[m.queryIdx].pt
            u_r, v_r = kp_r[m.trainIdx].pt
            all_matches.append((float(u_s), float(v_s), float(u_r), float(v_r)))
        if (k + 1) % 5 == 0:
            print(f"      frame {k+1}/{len(pinhole)}: "
                  f"running total = {len(all_matches)} matches")

    print(f"      {len(all_matches)} total raw matches (frames: "
          f"min={min(matches_per_frame)}, "
          f"max={max(matches_per_frame)}, "
          f"median={int(np.median(matches_per_frame))})")
    if len(all_matches) < 50:
        raise SystemExit("Too few matches to fit reliably (<50). "
                         "Try a different source FoV or stride.")

    arr = np.array(all_matches, dtype=np.float32)
    u_sim, v_sim = arr[:, 0], arr[:, 1]
    u_real_obs, v_real_obs = arr[:, 2], arr[:, 3]
    # World direction from sim render (z=1 plane):
    x_norm = (u_sim - src_cx) / src_fx
    y_norm = (v_sim - src_cx) / src_fx

    # Fix principal point at output center — a real wrist cam mounted on a
    # drone that filmed centered scenes wouldn't have a wildly off-center
    # principal point. Removing 2 free parameters helps the fit a lot.
    cx_fixed = cy_fixed = (OUT_SIZE - 1) / 2.0

    def project_with_fixed_pp(x_n: np.ndarray, y_n: np.ndarray,
                              fx: float, k1: float, k2: float) -> tuple[np.ndarray, np.ndarray]:
        return project_distorted(x_n, y_n, fx, cx_fixed, cy_fixed, k1, k2)

    def residual_minset(params: np.ndarray,
                        x_n: np.ndarray, y_n: np.ndarray,
                        u_obs: np.ndarray, v_obs: np.ndarray) -> np.ndarray:
        fx, k1, k2 = params
        up, vp = project_with_fixed_pp(x_n, y_n, fx, k1, k2)
        return np.concatenate([up - u_obs, vp - v_obs])

    # 3) RANSAC: random minimal subsets → fit → count inliers
    print(f"[3/5] RANSAC ({args.ransac_iters} iters, "
          f"inlier thresh = {args.ransac_thresh:.1f} px)…")
    rng = np.random.default_rng(0)
    N = len(arr)
    best_count = 0
    best_params = np.array([args.init_fx, 0.0, 0.0])
    bounds = ([args.bounds_fx_min, -0.5, -0.5],
              [args.bounds_fx_max, +0.5, +0.5])
    for it in range(args.ransac_iters):
        sample = rng.choice(N, size=min(8, N), replace=False)
        try:
            r = least_squares(
                residual_minset, np.array([args.init_fx, 0.0, 0.0]),
                args=(x_norm[sample], y_norm[sample],
                      u_real_obs[sample], v_real_obs[sample]),
                bounds=bounds, max_nfev=50, method="trf",
            )
        except Exception:
            continue
        # Score full set with this hypothesis
        up_all, vp_all = project_with_fixed_pp(x_norm, y_norm,
                                               r.x[0], r.x[1], r.x[2])
        err_all = np.sqrt((up_all - u_real_obs) ** 2 + (vp_all - v_real_obs) ** 2)
        inl_count = int((err_all <= args.ransac_thresh).sum())
        if inl_count > best_count:
            best_count = inl_count
            best_params = r.x.copy()
    print(f"      best RANSAC hypothesis: {best_count}/{N} inliers "
          f"(fx={best_params[0]:.1f}, k1={best_params[1]:+.4f}, "
          f"k2={best_params[2]:+.4f})")
    if best_count < 50:
        print("      ⚠ very few inliers; result may be unreliable")

    # 4) Final fit on the full inlier set with robust loss
    up_all, vp_all = project_with_fixed_pp(x_norm, y_norm,
                                           best_params[0], best_params[1],
                                           best_params[2])
    err_all = np.sqrt((up_all - u_real_obs) ** 2 + (vp_all - v_real_obs) ** 2)
    inliers = err_all <= args.ransac_thresh
    res = least_squares(
        residual_minset, best_params,
        args=(x_norm[inliers], y_norm[inliers],
              u_real_obs[inliers], v_real_obs[inliers]),
        bounds=bounds, loss="huber", max_nfev=500, method="trf",
    )
    fx_b, k1_b, k2_b = res.x
    cx_b = cx_fixed
    cy_b = cy_fixed
    final_err = np.median(np.sqrt(res.fun[:len(res.fun)//2] ** 2
                                  + res.fun[len(res.fun)//2:] ** 2))
    half_fov = np.arctan2(OUT_SIZE / 2.0, fx_b)
    fov_deg = 2.0 * np.degrees(half_fov)
    print(f"      BEST  fx={fx_b:.1f}, k1={k1_b:+.4f}, k2={k2_b:+.4f}, "
          f"FoV≈{fov_deg:.1f}°  (median inlier residual {final_err:.2f} px)")

    # 4) Visualize
    print("[4/5] building side-by-side grid…")
    pin_resized = [np.array(Image.fromarray(p).resize((OUT_SIZE, OUT_SIZE),
                                                      Image.BILINEAR))
                   for p in pinhole]
    best_imgs = [warp_with_model(p, src_fx=src_fx, src_cx=src_cx,
                                 out_fx=fx_b, out_cx=cx_b, out_cy=cy_b,
                                 k1=k1_b, k2=k2_b)
                 for p in pinhole]

    columns = [
        ("REAL wrist", real_imgs),
        (f"pinhole src @ {args.source_fov_deg:.0f}°", pin_resized),
        (f"FIT fx={fx_b:.0f} k1={k1_b:+.3f} k2={k2_b:+.3f} "
         f"(fov={fov_deg:.0f}°)", best_imgs),
    ]
    H = W = OUT_SIZE
    gap = 4; header_h = 56; row_h = H + gap; label_w = 220
    n = len(indices)
    n_cols = len(columns)
    canvas_w = label_w + n_cols * W + n_cols * gap
    canvas_h = header_h + n * row_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except Exception:
        font = font_small = ImageFont.load_default()
    for col_idx, (col_label, _) in enumerate(columns):
        x_col = label_w + gap + col_idx * (W + gap)
        for line_idx, chunk in enumerate(_chunk_label(col_label, 30)):
            draw.text((x_col + 4, 6 + 14 * line_idx),
                      chunk,
                      fill=(180, 220, 180) if col_idx == 0 else (180, 200, 240),
                      font=font_small)
    for k in range(n):
        y = header_h + k * row_h
        for col_idx, (_, frames) in enumerate(columns):
            x_col = label_w + gap + col_idx * (W + gap)
            canvas.paste(Image.fromarray(frames[k]), (x_col, y))
        draw.text((6, y + 4),
                  f"t = {indices[k] * 0.1:5.1f} s  m={matches_per_frame[k]}",
                  fill=(240, 240, 240), font=font)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)

    json.dump(
        {
            "scene": str(args.scene),
            "episode": str(args.episode),
            "source_fov_deg": args.source_fov_deg,
            "source_size": args.source_size,
            "stride": args.stride,
            "n_frames": n,
            "n_matches_raw": int(len(arr)),
            "n_inliers": int(inliers.sum()),
            "matches_per_frame": matches_per_frame,
            "best": {
                "fx_out": float(fx_b), "cx_out": float(cx_b),
                "cy_out": float(cy_b), "k1": float(k1_b), "k2": float(k2_b),
                "implied_fov_deg": float(fov_deg),
                "median_residual_px": float(final_err),
            },
        },
        args.out.with_suffix(".json").open("w"), indent=2,
    )
    print(f"[5/5] DONE in {time.time() - t0:.1f}s → {args.out}")


def _chunk_label(s: str, width: int) -> list[str]:
    out, line = [], ""
    for tok in s.split(" "):
        if len(line) + len(tok) + 1 > width:
            out.append(line.strip()); line = tok
        else:
            line += " " + tok
    if line.strip():
        out.append(line.strip())
    return out[:3]


if __name__ == "__main__":
    main()
