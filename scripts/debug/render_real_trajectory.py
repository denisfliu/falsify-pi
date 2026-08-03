"""Render a real teleop trajectory through our downward-cam pipeline.

Diagnostic for camera-intrinsics tuning. Pipeline:

1. Read a real LeRobot episode (state column = MOCAP) + matching
   wrist_image bytes.
2. Convert MOCAP → NED via the scene's FrameGraph.
3. Render the downward camera at every Nth pose through `GSplatRenderer`
   at a chosen *source pinhole* FoV (wider than the target fisheye so
   no information is lost on the way out).
4. (Optional) Post-warp the wide pinhole render into a fisheye image
   under an equidistant projection model — the parameter we're trying
   to fit.
5. Apply the standard `CameraPostprocess` (resize → BGR → gripper
   overlay).
6. Emit a side-by-side grid of `(REAL wrist | OUR render)` pairs.

The wide pinhole renders are cached to disk so iterating fisheye
intrinsics is a sub-second re-warp, not a 30 s renderer reload.

Usage::

    source tools/env.sh
    PYTHONPATH=src:external/FiGS/src:external/splatnav \\
      .venv/bin/python scripts/debug/render_real_trajectory.py \\
        --scene configs/scenes/left_gate.yaml \\
        --episode data/atomic_datasets/gate_scenes_real_combined/data/chunk-000/episode_000000.parquet \\
        --stride 12 \\
        --source-fov-deg 120 \\
        --fisheye-fx 80 \\
        --out /tmp/left_gate_real_vs_render.png
"""

from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--frame", type=Path,
                   default=REPO_ROOT / "configs/frames/carl_dual.yaml")
    p.add_argument("--embodiment", type=Path,
                   default=REPO_ROOT / "configs/embodiments/carl_dual_mocap.yaml")
    p.add_argument("--episode", type=Path, required=True,
                   help="LeRobot parquet path for one real episode")
    p.add_argument("--stride", type=int, default=12,
                   help="Render every Nth frame")
    p.add_argument("--source-fov-deg", type=str, default="120",
                   help="Source-pinhole horizontal FoV (degrees). "
                        "Comma-separated to sweep multiple FoVs — each "
                        "becomes one column in the grid (each renders "
                        "independently and caches to its own subdir).")
    p.add_argument("--source-size", type=int, default=512,
                   help="Source-pinhole render edge (square).")
    p.add_argument("--fisheye-fx", type=str, default=None,
                   help="Comma-separated equidistant fisheye focal lengths "
                        "(pixels at 256x256 output). Multiple values produce "
                        "a sweep — one column per value in the grid. Pass an "
                        "empty string '' to also include a pure-pinhole (no "
                        "warp) column. If unset, no warp at all.")
    p.add_argument("--fisheye-fy", type=str, default=None,
                   help="Defaults to --fisheye-fx (square pixels).")
    p.add_argument("--cx-shift", type=str, default=None,
                   help="Comma-separated principal-point shifts (px, +right) "
                        "applied to the *output* 256x256 pinhole reproject. "
                        "Each value becomes one extra column. Positive shift "
                        "= more world content visible on the LEFT (cx moves "
                        "right). Negative = more on the right.")
    p.add_argument("--cy-shift", type=str, default=None,
                   help="Comma-separated principal-point shifts in Y (px, "
                        "+down). Positive shift = cy moves DOWN, image "
                        "shows more world content on the TOP. Negative = "
                        "more on the bottom. Output FoV honours the "
                        "decoupled --tilt-output-fov-x-deg / -y-deg knobs.")
    p.add_argument("--tilt-y-deg", type=str, default=None,
                   help="Comma-separated extrinsic-tilt sweep about the "
                        "camera Y axis (degrees, + tilts optical axis "
                        "toward image +X). Models a small downward-cam "
                        "yaw/roll mount offset. Reprojects from the cached "
                        "wide pinhole — no fresh render needed.")
    p.add_argument("--tilt-x-deg", type=str, default=None,
                   help="Same idea but about the camera X axis. + tilts "
                        "optical axis toward image +Y (forward/back).")
    p.add_argument("--tilt-output-fov-x-deg", type=float, default=130.0,
                   help="Output horizontal FoV when --tilt-*-deg is set. "
                        "Default 130°.")
    p.add_argument("--tilt-output-fov-y-deg", type=float, default=110.0,
                   help="Output vertical FoV when --tilt-*-deg is set. "
                        "Default 110°. Decoupled from H so we can model "
                        "non-square angular pixels.")
    p.add_argument("--cache-dir", type=Path,
                   default=REPO_ROOT / "runs/intrinsics_cache",
                   help="Where to cache wide-pinhole renders so fisheye "
                        "iteration skips the renderer reload.")
    p.add_argument("--reset-cache", action="store_true",
                   help="Re-render wide pinholes even if cached.")
    p.add_argument("--out", type=Path, default=Path("/tmp/render_real_grid.png"))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Equidistant fisheye warp (post-pinhole resample)
# ---------------------------------------------------------------------------


def pinhole_reproject_tilted(
    src_rgb: np.ndarray,
    *,
    src_fx: float, src_cx: float,
    out_size: int, out_fov_x_deg: float, out_fov_y_deg: float,
    tilt_x_deg: float, tilt_y_deg: float,
) -> np.ndarray:
    """Reproject a source pinhole render under a tilted output camera.

    Output pinhole has decoupled horizontal/vertical FoVs to model
    non-square angular pixels (real wide-angle drone cams often have
    fx ≠ fy). Models: the output camera has the same body position
    but its optical axis is rotated by (tilt_x_deg, tilt_y_deg) about
    the cam X and Y axes (OpenCV camera convention: x=right, y=down,
    z=forward).

    Composition: ``r_src = R_x @ R_y @ r_out`` where r_out is the unit
    ray in the output camera frame. The src pixel is then the pinhole
    projection of r_src through the (symmetric) source intrinsics.
    """
    out_fx = (out_size / 2.0) / np.tan(np.deg2rad(out_fov_x_deg) / 2.0)
    out_fy = (out_size / 2.0) / np.tan(np.deg2rad(out_fov_y_deg) / 2.0)
    out_cx = out_cy = (out_size - 1) / 2.0
    uu, vv = np.meshgrid(
        np.arange(out_size, dtype=np.float32),
        np.arange(out_size, dtype=np.float32),
    )
    x = (uu - out_cx) / out_fx
    y = (vv - out_cy) / out_fy
    z = np.ones_like(x)

    # R_y(theta): rotates about cam Y, taking +Z toward +X for positive theta.
    ty = np.deg2rad(tilt_y_deg)
    tx = np.deg2rad(tilt_x_deg)
    cy_, sy_ = np.cos(ty), np.sin(ty)
    cx_, sx_ = np.cos(tx), np.sin(tx)
    # R_y
    x1 = cy_ * x + sy_ * z
    z1 = -sy_ * x + cy_ * z
    y1 = y
    # R_x (about output X — tilts +Z toward +Y for positive tx)
    y2 = cx_ * y1 - sx_ * z1
    z2 = sx_ * y1 + cx_ * z1
    x2 = x1

    # Project into source pinhole at z=z2 plane
    eps = 1e-6
    u_src = src_fx * (x2 / np.maximum(z2, eps)) + src_cx
    v_src = src_fx * (y2 / np.maximum(z2, eps)) + src_cx
    behind = z2 < eps

    H, W = src_rgb.shape[:2]
    u0 = np.floor(u_src).astype(np.int32)
    v0 = np.floor(v_src).astype(np.int32)
    u1, v1 = u0 + 1, v0 + 1
    ib = (u0 >= 0) & (u1 < W) & (v0 >= 0) & (v1 < H) & (~behind)
    u0 = np.clip(u0, 0, W - 1); u1 = np.clip(u1, 0, W - 1)
    v0 = np.clip(v0, 0, H - 1); v1 = np.clip(v1, 0, H - 1)
    du = (u_src - u0).astype(np.float32)[..., None]
    dv = (v_src - v0).astype(np.float32)[..., None]
    src = src_rgb.astype(np.float32)
    top = src[v0, u0] * (1 - du) + src[v0, u1] * du
    bot = src[v1, u0] * (1 - du) + src[v1, u1] * du
    out = top * (1 - dv) + bot * dv
    out[~ib] = 0
    return out.astype(np.uint8)


def pinhole_reproject_asymmetric(
    src_rgb: np.ndarray,
    *,
    src_fx: float, src_cx: float,
    out_size: int,
    out_fx: float, out_cx: float, out_cy: float | None = None,
    out_fy: float | None = None,
) -> np.ndarray:
    """Resample a symmetric pinhole render into an output pinhole with
    arbitrary (possibly off-center) principal point.

    For each output pixel (u, v):
        x_n = (u - out_cx) / out_fx       # normalized ray direction at z=1
        y_n = (v - out_cy) / out_fx
        u_src = src_fx * x_n + src_cx     # project through source pinhole
        v_src = src_fx * y_n + src_cx
    Sample bilinear.

    Used to test whether asymmetric image content in the real wrist cam is
    explained by a non-central principal point on a wider source FoV.
    """
    out_cy = (out_size - 1) / 2.0 if out_cy is None else out_cy
    out_fy = out_fx if out_fy is None else out_fy
    uu, vv = np.meshgrid(
        np.arange(out_size, dtype=np.float32),
        np.arange(out_size, dtype=np.float32),
    )
    x_n = (uu - out_cx) / out_fx
    y_n = (vv - out_cy) / out_fy
    u_src = src_fx * x_n + src_cx
    v_src = src_fx * y_n + src_cx

    H, W = src_rgb.shape[:2]
    u0 = np.floor(u_src).astype(np.int32)
    v0 = np.floor(v_src).astype(np.int32)
    u1, v1 = u0 + 1, v0 + 1
    ib = (u0 >= 0) & (u1 < W) & (v0 >= 0) & (v1 < H)
    u0 = np.clip(u0, 0, W - 1); u1 = np.clip(u1, 0, W - 1)
    v0 = np.clip(v0, 0, H - 1); v1 = np.clip(v1, 0, H - 1)
    du = (u_src - u0).astype(np.float32)[..., None]
    dv = (v_src - v0).astype(np.float32)[..., None]
    src = src_rgb.astype(np.float32)
    top = src[v0, u0] * (1 - du) + src[v0, u1] * du
    bot = src[v1, u0] * (1 - du) + src[v1, u1] * du
    out = top * (1 - dv) + bot * dv
    out[~ib] = 0
    return out.astype(np.uint8)


def fisheye_remap_from_pinhole(
    src_rgb: np.ndarray,
    *,
    src_fx: float, src_fy: float, src_cx: float, src_cy: float,
    out_size: int,
    fish_fx: float, fish_fy: float, fish_cx: float | None = None,
    fish_cy: float | None = None,
) -> np.ndarray:
    """Resample a pinhole RGB image into an equidistant-fisheye image.

    Equidistant model: ``r_fish = f_fish * theta`` where ``theta`` is the
    ray's angle from the optical axis. For each output pixel:

    1. Convert (u, v) to image-plane angle theta and azimuth phi.
    2. 3D unit ray:  d = (sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)).
    3. Pinhole projection back to source pixel: u_p = fx * dx/dz + cx,
       v_p = fy * dy/dz + cy. Sample (bilinear via PIL).

    Rays with theta > pi/2 (behind the camera) → black.
    """
    fish_cx = (out_size - 1) / 2.0 if fish_cx is None else fish_cx
    fish_cy = (out_size - 1) / 2.0 if fish_cy is None else fish_cy
    uu, vv = np.meshgrid(
        np.arange(out_size, dtype=np.float32),
        np.arange(out_size, dtype=np.float32),
    )
    xn = (uu - fish_cx) / fish_fx
    yn = (vv - fish_cy) / fish_fy
    r = np.sqrt(xn * xn + yn * yn)
    theta = r  # equidistant
    # Behind-camera mask
    behind = theta >= (np.pi / 2 - 1e-3)

    eps = 1e-9
    cos_phi = np.where(r > eps, xn / np.maximum(r, eps), 0.0)
    sin_phi = np.where(r > eps, yn / np.maximum(r, eps), 0.0)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    dx = sin_t * cos_phi
    dy = sin_t * sin_phi
    dz = np.maximum(cos_t, eps)  # avoid div-by-0 right at theta=pi/2

    u_p = src_fx * (dx / dz) + src_cx
    v_p = src_fy * (dy / dz) + src_cy

    H, W, _ = src_rgb.shape
    oob = (u_p < 0) | (u_p > W - 1) | (v_p < 0) | (v_p > H - 1) | behind

    # Bilinear sample
    u0 = np.clip(np.floor(u_p).astype(np.int32), 0, W - 1)
    v0 = np.clip(np.floor(v_p).astype(np.int32), 0, H - 1)
    u1 = np.clip(u0 + 1, 0, W - 1)
    v1 = np.clip(v0 + 1, 0, H - 1)
    du = np.clip(u_p - u0, 0, 1)[..., None]
    dv = np.clip(v_p - v0, 0, 1)[..., None]

    src = src_rgb.astype(np.float32)
    top = src[v0, u0] * (1 - du) + src[v0, u1] * du
    bot = src[v1, u0] * (1 - du) + src[v1, u1] * du
    sampled = top * (1 - dv) + bot * dv
    sampled[oob] = 0
    return sampled.astype(np.uint8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    t0 = time.time()

    # Imports placed inside main so that running with --reset-cache for an
    # already-cached scene doesn't pay the gsplat-load cost.
    import sys
    sys.path.insert(0, str(REPO_ROOT / "src"))
    sys.path.insert(0, str(REPO_ROOT / "external/FiGS/src"))
    sys.path.insert(0, str(REPO_ROOT / "external/splatnav"))
    from falsify.geometry import Point, yaw_to_quat_xyzw
    from falsify.io import build_frame_graph, load_yaml
    from falsify.policy.camera_postprocess import CameraPostprocess
    from falsify.sim.dynamics_state import DroneState
    from falsify.sim.poses import camera_to_world_pose
    from falsify.training import load_embodiment

    print(f"[1/6] loading parquet {args.episode.name}…")
    table = pq.read_table(args.episode)
    states = np.array(table.column("state").to_pylist(), dtype=np.float64)
    times = np.array(table.column("timestamp").to_pylist(), dtype=np.float64)
    wrist_bytes = [c["bytes"] for c in table.column("wrist_image").to_pylist()]
    print(f"      {len(states)} states, dt={times[1]-times[0]:.3f}s")

    print(f"[2/6] building FrameGraph + scene from {args.scene.name}…")
    scene_cfg = load_yaml(args.scene)
    frame_cfg = load_yaml(args.frame)
    fg = build_frame_graph(scene_cfg, base_path=args.scene.parent)

    # MOCAP → NED. state[i] = [px_m, py_m, pz_m, yaw_m, 0, 0, 0]
    pos_mocap = states[:, :3]
    yaw_mocap = states[:, 3]
    T = fg.transform("mocap", "ned")
    R = T.R if not hasattr(T, "s") else T.s * T.R
    pos_ned = (R @ pos_mocap.T).T + T.t
    yaw_ned = -yaw_mocap

    # Source pinhole sweep — one render-set per FoV.
    src_size = args.source_size
    fov_values = [float(t.strip()) for t in args.source_fov_deg.split(",") if t.strip()]
    indices = list(range(0, len(states), args.stride))
    ned_frame = fg.frame("ned")

    renderer = None  # lazy-load only if any FoV needs fresh rendering
    spec = None
    # Map FoV -> {"intr": dict, "frames": [np.ndarray]}
    per_fov_renders: dict[float, dict] = {}
    for fov_deg in fov_values:
        half_fov = np.deg2rad(fov_deg) / 2.0
        src_fx = src_fy = (src_size / 2.0) / np.tan(half_fov)
        src_cx = src_cy = (src_size - 1) / 2.0
        wide_intr = {
            "width": src_size, "height": src_size,
            "fx": float(src_fx), "fy": float(src_fy),
            "cx": float(src_cx), "cy": float(src_cy),
        }
        cache_key = (
            f"{args.scene.stem}_ep{args.episode.stem}_str{args.stride}"
            f"_size{src_size}_fov{int(fov_deg)}"
        )
        cache_dir = args.cache_dir / cache_key
        cached_files = [cache_dir / f"frame_{i:05d}.png" for i in indices]
        if not args.reset_cache and all(p.exists() for p in cached_files):
            print(f"[3/6] FoV {fov_deg:.0f}°: re-using cache {cache_dir.name}")
            wide_renders = [np.array(Image.open(p)) for p in cached_files]
        else:
            if renderer is None:
                print("[3/6] loading GSplatRenderer (CUDA JIT, may take ~30 s)…")
                from falsify.sensors.camera import make_camera_sensor_from_yaml
                from falsify.sim.renderer import GSplatRenderer
                renderer = GSplatRenderer.from_scene_cfg(
                    scene_cfg, scene_dir=args.scene.parent,
                )
                cam_sensor = make_camera_sensor_from_yaml(
                    "downward", frame_cfg["cameras"]["downward"], fg,
                    renderer=renderer.render, body_to_world=camera_to_world_pose,
                )
                spec = cam_sensor.spec
            print(f"      FoV {fov_deg:.0f}°: rendering {len(indices)} frames…")
            cache_dir.mkdir(parents=True, exist_ok=True)
            wide_renders = []
            for k, i in enumerate(indices):
                ds = DroneState(
                    pos=Point(pos_ned[i], frame=ned_frame),
                    vel=np.zeros(3),
                    quat_xyzw=yaw_to_quat_xyzw(float(yaw_ned[i])),
                    t=float(times[i]),
                )
                cam_pose = camera_to_world_pose(ds, spec.body_from_camera)
                rgb, _ = renderer.render(cam_pose, wide_intr)
                rgb = np.asarray(rgb, dtype=np.uint8)
                wide_renders.append(rgb)
                Image.fromarray(rgb).save(cached_files[k])
                if (k + 1) % 5 == 0:
                    print(f"        {k+1}/{len(indices)}")
        per_fov_renders[fov_deg] = {"intr": wide_intr, "frames": wide_renders}

    print("[4/6] applying fisheye warp + postprocess…")
    emb = load_embodiment(args.embodiment)
    wrist_cam = [c for c in emb.cameras if c.column == "wrist_image"][0]
    pp = CameraPostprocess.from_paths(
        image_size=wrist_cam.image_size,
        channel_order=wrist_cam.channel_order,   # "BGR" for carl_dual_mocap
        overlay_path=str(REPO_ROOT / wrist_cam.gripper_overlay_path),
    )

    # Build one column per (fov, fisheye_fx). Pure pinhole = fisheye_fx None.
    fisheye_tokens: list[float | None]
    if args.fisheye_fx is None:
        fisheye_tokens = [None]
    else:
        fisheye_tokens = []
        for tok in args.fisheye_fx.split(","):
            tok = tok.strip()
            fisheye_tokens.append(None if tok == "" else float(tok))

    # Parse --cx-shift sweep
    cx_shift_values: list[float] = []
    if args.cx_shift is not None:
        for tok in args.cx_shift.split(","):
            tok = tok.strip()
            if tok:
                cx_shift_values.append(float(tok))
    cy_shift_values: list[float] = []
    if args.cy_shift is not None:
        for tok in args.cy_shift.split(","):
            tok = tok.strip()
            if tok:
                cy_shift_values.append(float(tok))
    tilt_y_values: list[float] = []
    if args.tilt_y_deg is not None:
        for tok in args.tilt_y_deg.split(","):
            tok = tok.strip()
            if tok:
                tilt_y_values.append(float(tok))
    tilt_x_values: list[float] = []
    if args.tilt_x_deg is not None:
        for tok in args.tilt_x_deg.split(","):
            tok = tok.strip()
            if tok:
                tilt_x_values.append(float(tok))

    columns: list[tuple[str, float, object, float]] = []
    for fov_deg in fov_values:
        for fx in fisheye_tokens:
            label = (f"pinhole FoV={fov_deg:.0f}°" if fx is None
                     else f"FoV={fov_deg:.0f}° + fish fx={fx:.0f}")
            columns.append((label, fov_deg, fx, 0.0))
        for cx_shift in cx_shift_values:
            label = (f"H={args.tilt_output_fov_x_deg:.0f}° "
                     f"V={args.tilt_output_fov_y_deg:.0f}° "
                     f"cx{cx_shift:+.0f}px")
            columns.append((label, fov_deg, "cx_only", cx_shift))
        for cy_shift in cy_shift_values:
            label = (f"H={args.tilt_output_fov_x_deg:.0f}° "
                     f"V={args.tilt_output_fov_y_deg:.0f}° "
                     f"cy{cy_shift:+.0f}px")
            columns.append((label, fov_deg, "cy_only", cy_shift))
        for ty in tilt_y_values:
            label = (f"tilt_y={ty:+.1f}° "
                     f"(out fov H={args.tilt_output_fov_x_deg:.0f}° "
                     f"V={args.tilt_output_fov_y_deg:.0f}°)")
            columns.append((label, fov_deg, "tilt_y", ty))
        for tx in tilt_x_values:
            label = (f"tilt_x={tx:+.1f}° "
                     f"(out fov H={args.tilt_output_fov_x_deg:.0f}° "
                     f"V={args.tilt_output_fov_y_deg:.0f}°)")
            columns.append((label, fov_deg, "tilt_x", tx))

    OUT = wrist_cam.image_size
    final_out: list[list[np.ndarray]] = []
    for col_label, fov_deg, mode, param in columns:
        intr = per_fov_renders[fov_deg]["intr"]
        col_frames: list[np.ndarray] = []
        for wide in per_fov_renders[fov_deg]["frames"]:
            if mode is None:
                col_frames.append(pp.apply(wide))
            elif mode == "cx_only":
                cx_shift = param
                out_fx = (OUT / 2.0) / np.tan(
                    np.deg2rad(args.tilt_output_fov_x_deg) / 2.0)
                out_fy = (OUT / 2.0) / np.tan(
                    np.deg2rad(args.tilt_output_fov_y_deg) / 2.0)
                out_cx = (OUT - 1) / 2.0 + cx_shift
                reproj = pinhole_reproject_asymmetric(
                    wide, src_fx=intr["fx"], src_cx=intr["cx"],
                    out_size=OUT, out_fx=out_fx, out_fy=out_fy,
                    out_cx=out_cx,
                )
                col_frames.append(pp.apply(reproj))
            elif mode == "cy_only":
                cy_shift = param
                out_fx = (OUT / 2.0) / np.tan(
                    np.deg2rad(args.tilt_output_fov_x_deg) / 2.0)
                out_fy = (OUT / 2.0) / np.tan(
                    np.deg2rad(args.tilt_output_fov_y_deg) / 2.0)
                out_cy = (OUT - 1) / 2.0 + cy_shift
                reproj = pinhole_reproject_asymmetric(
                    wide, src_fx=intr["fx"], src_cx=intr["cx"],
                    out_size=OUT, out_fx=out_fx, out_fy=out_fy,
                    out_cx=(OUT - 1) / 2.0, out_cy=out_cy,
                )
                col_frames.append(pp.apply(reproj))
            elif mode == "tilt_y":
                reproj = pinhole_reproject_tilted(
                    wide, src_fx=intr["fx"], src_cx=intr["cx"],
                    out_size=OUT,
                    out_fov_x_deg=args.tilt_output_fov_x_deg,
                    out_fov_y_deg=args.tilt_output_fov_y_deg,
                    tilt_x_deg=0.0, tilt_y_deg=param,
                )
                col_frames.append(pp.apply(reproj))
            elif mode == "tilt_x":
                reproj = pinhole_reproject_tilted(
                    wide, src_fx=intr["fx"], src_cx=intr["cx"],
                    out_size=OUT,
                    out_fov_x_deg=args.tilt_output_fov_x_deg,
                    out_fov_y_deg=args.tilt_output_fov_y_deg,
                    tilt_x_deg=param, tilt_y_deg=0.0,
                )
                col_frames.append(pp.apply(reproj))
            else:
                # fisheye fx
                warped = fisheye_remap_from_pinhole(
                    wide,
                    src_fx=intr["fx"], src_fy=intr["fy"],
                    src_cx=intr["cx"], src_cy=intr["cy"],
                    out_size=OUT,
                    fish_fx=mode, fish_fy=mode,
                )
                col_frames.append(pp.apply(warped))
        final_out.append(col_frames)

    # The pipeline-wide convention is BGR everywhere (the parquet bytes are
    # BGR; the policy receives BGR; the gripper overlay was authored in
    # BGR). Diagnostics match by leaving the real images in their as-stored
    # form and forcing the postprocess to BGR below.
    real_imgs = [
        np.array(Image.open(io.BytesIO(wrist_bytes[i])).convert("RGB"))
        for i in indices
    ]
    poses_logged = [
        (float(pos_ned[i, 0]), float(pos_ned[i, 1]),
         float(pos_ned[i, 2]), float(yaw_ned[i]))
        for i in indices
    ]

    print("[5/6] composing grid…")
    H = W = 256
    gap = 4
    header_h = 32
    row_h = H + gap
    label_w = 220
    n = len(indices)
    n_cols = 1 + len(columns)  # REAL + each rendered column
    canvas_w = label_w + n_cols * W + n_cols * gap
    canvas_h = header_h + n * row_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font = font_small = ImageFont.load_default()
    # Column headers
    draw.text((label_w + gap + 4, 8), "REAL wrist (parquet)",
              fill=(180, 220, 180), font=font)
    for col_idx, (col_label, _fov, _fx, _cx) in enumerate(columns):
        x_col = label_w + gap + (col_idx + 1) * (W + gap)
        draw.text((x_col + 4, 8), col_label, fill=(180, 200, 240), font=font)
    for k in range(n):
        y = header_h + k * row_h
        canvas.paste(Image.fromarray(real_imgs[k]), (label_w + gap, y))
        for col_idx, _ in enumerate(columns):
            x_col = label_w + gap + (col_idx + 1) * (W + gap)
            canvas.paste(Image.fromarray(final_out[col_idx][k]), (x_col, y))
        px, py, pz, yw = poses_logged[k]
        draw.text((6, y + 4),
                  f"t = {indices[k]*0.1:5.1f} s",
                  fill=(240, 240, 240), font=font)
        draw.text((6, y + 22),
                  f"NED = ({px:+.2f}, {py:+.2f}, {pz:+.2f})",
                  fill=(200, 200, 200), font=font_small)
        draw.text((6, y + 38),
                  f"yaw  = {yw:+.2f} rad",
                  fill=(200, 200, 200), font=font_small)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(f"[6/6] DONE in {time.time()-t0:.1f}s → {args.out}")


if __name__ == "__main__":
    main()
