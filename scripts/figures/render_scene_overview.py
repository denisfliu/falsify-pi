"""Render an overhead + start-position-front view of a scene through the
GSplatRenderer. Used to visually confirm scene_edits (including
DuplicateAABB) are applied correctly to the actual gsplat — not just the
PLY clouds the inspector shows.

Usage::

    PYTHONPATH=src python scripts/figures/render_scene_overview.py \\
        --scene configs/scenes/left_and_center.yaml \\
        --out runs/inspect/left_and_center_render

Saves ``<out>/overhead.png`` and ``<out>/front_from_start.png``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from falsify.geometry import Point, Pose, assert_frame
from falsify.io import build_frame_graph, load_yaml


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _look_at_ned(cam_pos_ned: np.ndarray, target_ned: np.ndarray,
                 world_up_ned: np.ndarray) -> np.ndarray:
    """Build a 4x4 camera-to-world matrix in NED for an OpenCV-style camera
    (x = right, y = down, z = forward-into-image). ``world_up_ned`` only
    needs to be non-parallel to the look-at vector."""
    cam_pos = np.asarray(cam_pos_ned, dtype=np.float64)
    target = np.asarray(target_ned, dtype=np.float64)
    up = np.asarray(world_up_ned, dtype=np.float64)
    forward = target - cam_pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    image_down = np.cross(forward, right)            # cam +y (down) in world
    R = np.column_stack([right, image_down, forward])
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = cam_pos
    return M


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path,
                    help="Output directory; PNGs land here.")
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--fov-deg", type=float, default=110.0)
    args = ap.parse_args()

    scene_yaml = (args.scene if args.scene.is_absolute()
                  else (REPO_ROOT / args.scene).resolve())
    scene_cfg = load_yaml(scene_yaml)
    scene_dir = scene_yaml.parent
    fg = build_frame_graph(scene_cfg, base_path=scene_dir)

    # Build the renderer (loads gsplat, applies scene_edits).
    from falsify.sim.renderer import GSplatRenderer
    renderer = GSplatRenderer.from_scene_cfg(scene_cfg, scene_dir=scene_dir)
    print(f"[render] scene={scene_yaml.name} "
          f"gsplat n_gaussians={renderer._impl.pipeline.model.means.shape[0]}")

    # Intrinsics — square-ish FOV.
    fov = np.deg2rad(args.fov_deg)
    fx = args.width / (2 * np.tan(fov / 2))
    fy = fx
    intrinsics = {
        "width": args.width, "height": args.height,
        "fx": fx, "fy": fy,
        "cx": args.width / 2.0, "cy": args.height / 2.0,
    }

    # Viewpoint definitions in MOCAP (human-readable), converted to NED.
    start_mocap = np.asarray(scene_cfg["start_position_mocap"], dtype=np.float64)

    def _to_ned(p_mocap):
        pt = Point.of(*p_mocap, fg.frame("mocap"))
        return fg.convert(pt, to="ned").xyz

    # Overhead at 3 m MOCAP-z so we stay roughly inside the trained-scene
    # volume (gates top out near 2 m). Image "up" is MOCAP +x (= NED +x)
    # so the scene's forward direction reads top-of-frame.
    cam_overhead_ned = _to_ned(np.array([1.5, 0.0, 3.0]))
    look_overhead_ned = _to_ned(np.array([1.5, 0.0, 0.0]))
    up_overhead = np.array([1.0, 0.0, 0.0])

    cam_front_ned = _to_ned(start_mocap)
    look_front_ned = _to_ned(start_mocap + np.array([3.0, 0.0, 0.0]))
    # World-up for any side / front view = MOCAP +z (up); in NED that's -z.
    up_world = np.array([0.0, 0.0, -1.0])

    # Behind-and-slightly-above the start, looking forward at gate height.
    # Both the original gate and the duplicate at the center should land in
    # frame with the wider default FOV.
    cam_overview_ned = _to_ned(np.array([-0.5, 0.2, 1.8]))
    look_overview_ned = _to_ned(np.array([2.0, 0.2, 1.5]))

    args.out.mkdir(parents=True, exist_ok=True)

    import imageio.v2 as imageio
    for name, cam_pos, look_at, up in [
        ("overhead",          cam_overhead_ned, look_overhead_ned, up_overhead),
        ("front_from_start",  cam_front_ned,    look_front_ned,    up_world),
        ("overview_behind",   cam_overview_ned, look_overview_ned, up_world),
    ]:
        M = _look_at_ned(cam_pos, look_at, up)
        pose = Pose.from_matrix(M, fg.frame("ned"))
        assert_frame(pose, "ned")
        rgb, _ = renderer.render(pose, intrinsics)
        out_path = args.out / f"{name}.png"
        imageio.imwrite(str(out_path), rgb)
        print(f"[render] {name}: wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
