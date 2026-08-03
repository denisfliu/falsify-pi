"""Side-by-side: undistorted real episode images vs pinhole gsplat renders.

Validates the rectify-real-to-pinhole strategy (see
``scripts/dataset/undistort_fisheye_dataset.py``): after the real images are
KB4-undistorted to the calibrated pinhole K, a distortion-free gsplat render
at the same (resolution-scaled) K from the same pose should line up
geometrically — same gate position, same edge straightness, same scale.

For every Nth frame of one episode the grid shows, per camera:

    REAL raw (fisheye) | REAL undistorted | RENDER pinhole @ calibrated K

Poses come from the episode's state column (MOCAP → NED via the scene's
FrameGraph, yaw_ned = -yaw_mocap), exactly as in render_real_trajectory.py.
Renders go through the standard `CameraPostprocess` (resize → BGR → wrist
gripper overlay) so the columns are channel-order-comparable with the
as-stored (BGR) parquet bytes.

Usage::

    source tools/env.sh
    PYTHONPATH=src .venv/bin/python \\
      scripts/debug/compare_real_undistorted_vs_render.py \\
        --scene configs/scenes/left_gate.yaml \\
        --episode data/atomic_datasets/gate_scenes_real_combined_undistorted/data/chunk-000/episode_000000.parquet \\
        --rows 8 --out /tmp/undistort_vs_render.png
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--episode", type=Path, required=True,
                   help="Parquet from the *undistorted* dataset.")
    p.add_argument("--orig-episode", type=Path, default=None,
                   help="Matching parquet from the original (fisheye) dataset "
                        "for the RAW column. Default: replace '_undistorted' "
                        "in --episode's path.")
    p.add_argument("--frame", type=Path,
                   default=REPO_ROOT / "configs/frames/carl_dual.yaml")
    p.add_argument("--embodiment", type=Path,
                   default=REPO_ROOT / "configs/embodiments/carl_dual_mocap.yaml")
    p.add_argument("--rows", type=int, default=8,
                   help="How many evenly-spaced frames to show.")
    p.add_argument("--out", type=Path, default=Path("/tmp/undistort_vs_render.png"))
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    t0 = time.time()
    sys.path.insert(0, str(REPO_ROOT / "src"))
    sys.path.insert(0, str(REPO_ROOT / "external/FiGS/src"))
    sys.path.insert(0, str(REPO_ROOT / "external/splatnav"))
    from falsify.geometry import Point, yaw_to_quat_xyzw
    from falsify.io import build_frame_graph, load_yaml
    from falsify.policy.camera_postprocess import CameraPostprocess
    from falsify.sensors.camera import make_camera_sensor_from_yaml
    from falsify.sim.dynamics_state import DroneState
    from falsify.sim.poses import camera_to_world_pose
    from falsify.sim.renderer import GSplatRenderer
    from falsify.training import load_embodiment

    orig_episode = args.orig_episode
    if orig_episode is None:
        orig_episode = Path(str(args.episode).replace("_undistorted", ""))
        if orig_episode == args.episode:
            raise SystemExit("--episode has no '_undistorted' in its path; "
                             "pass --orig-episode explicitly")

    print(f"[1/4] loading parquets…")
    table = pq.read_table(args.episode)
    orig_table = pq.read_table(orig_episode)
    states = np.array(table.column("state").to_pylist(), dtype=np.float64)
    n = len(states)
    indices = np.unique(np.linspace(0, n - 1, args.rows).astype(int)).tolist()

    # column → camera name, from the embodiment (render-sourced cameras only)
    emb = load_embodiment(args.embodiment)
    render_cams = [c for c in emb.cameras if c.source == "render"]

    print(f"[2/4] scene + FrameGraph from {args.scene.name}…")
    scene_cfg = load_yaml(args.scene)
    frame_cfg = load_yaml(args.frame)
    fg = build_frame_graph(scene_cfg, base_path=args.scene.parent)
    pos_mocap = states[:, :3]
    yaw_mocap = states[:, 3]
    T = fg.transform("mocap", "ned")
    R = T.R if not hasattr(T, "s") else T.s * T.R
    pos_ned = (R @ pos_mocap.T).T + T.t
    yaw_ned = -yaw_mocap
    ned_frame = fg.frame("ned")

    print("[3/4] loading GSplatRenderer (CUDA JIT, may take ~30 s)…")
    renderer = GSplatRenderer.from_scene_cfg(scene_cfg, scene_dir=args.scene.parent)

    cols: list[tuple[str, list[np.ndarray]]] = []
    for cam in render_cams:
        cam_yaml = frame_cfg["cameras"][cam.camera_name]
        sensor = make_camera_sensor_from_yaml(
            cam.camera_name, cam_yaml, fg,
            renderer=renderer.render, body_to_world=camera_to_world_pose,
        )
        # Scale calibrated K to the embodiment's square output resolution —
        # same per-axis scaling the dataset images underwent (1024x768 →
        # 256x256 plain resize), so render and rectified real share K.
        native = cam_yaml["intrinsics"]
        sx = cam.image_size / native["width"]
        sy = cam.image_size / native["height"]
        intr = {
            "width": cam.image_size, "height": cam.image_size,
            "fx": native["fx"] * sx, "fy": native["fy"] * sy,
            "cx": native["cx"] * sx, "cy": native["cy"] * sy,
        }
        pp = CameraPostprocess.from_paths(
            image_size=cam.image_size,
            channel_order=cam.channel_order,
            overlay_path=(REPO_ROOT / cam.gripper_overlay_path
                          if cam.gripper_overlay_path else None),
        )
        raw_imgs, und_imgs, ren_imgs = [], [], []
        for i in indices:
            for src_table, acc in ((orig_table, raw_imgs), (table, und_imgs)):
                b = src_table.column(cam.column).to_pylist()[i]["bytes"]
                acc.append(np.array(Image.open(io.BytesIO(b)).convert("RGB")))
            ds = DroneState(
                pos=Point(pos_ned[i], frame=ned_frame),
                vel=np.zeros(3),
                quat_xyzw=yaw_to_quat_xyzw(float(yaw_ned[i])),
                t=float(i),
            )
            cam_pose = camera_to_world_pose(ds, sensor.spec.body_from_camera)
            rgb, _ = renderer.render(cam_pose, intr)
            ren_imgs.append(pp.apply(np.asarray(rgb, dtype=np.uint8)))
        cols.append((f"{cam.camera_name} REAL raw", raw_imgs))
        cols.append((f"{cam.camera_name} REAL undist", und_imgs))
        cols.append((f"{cam.camera_name} RENDER pinhole", ren_imgs))
        print(f"      {cam.camera_name}: {len(indices)} frames rendered")

    print("[4/4] composing grid…")
    H = W = 256
    gap, header_h, label_w = 4, 32, 150
    canvas_w = label_w + len(cols) * (W + gap) + gap
    canvas_h = header_h + len(indices) * (H + gap)
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        font_small = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font = font_small = ImageFont.load_default()
    for c, (label, _) in enumerate(cols):
        draw.text((label_w + gap + c * (W + gap) + 4, 8), label,
                  fill=(180, 200, 240), font=font)
    for r, i in enumerate(indices):
        y = header_h + r * (H + gap)
        for c, (_, imgs) in enumerate(cols):
            canvas.paste(Image.fromarray(imgs[r]),
                         (label_w + gap + c * (W + gap), y))
        draw.text((6, y + 4), f"frame {i}", fill=(240, 240, 240), font=font)
        draw.text((6, y + 22),
                  f"NED ({pos_ned[i, 0]:+.2f}, {pos_ned[i, 1]:+.2f}, "
                  f"{pos_ned[i, 2]:+.2f})",
                  fill=(200, 200, 200), font=font_small)
        draw.text((6, y + 38), f"yaw {yaw_ned[i]:+.2f}",
                  fill=(200, 200, 200), font=font_small)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(f"DONE in {time.time() - t0:.1f}s → {args.out}")


if __name__ == "__main__":
    main()
