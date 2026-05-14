"""Render the forward camera at the drone's initial pose and dump diagnostics.

Run::

    CC=gcc-11 CXX=g++-11 \\
    PYTHONPATH=src:external/FiGS/src:external/splatnav \\
    .venv/bin/python scripts/debug_render_at_pose.py \\
        --scene configs/scenes/left_gate.yaml \\
        --frame configs/frames/carl_dual.yaml \\
        --out runs/diag

What it prints / writes:

- Tw2g matrix that FiGS computed for the scene (NED → NS via perm5 + dataparser).
- The drone's initial NED state and the resulting camera-to-world matrix in NED.
- The NS-frame camera origin (= Tw2g @ T_cam_to_world.translation).
- The NS-frame direction the camera lens looks at (OpenGL: -col3 of NS rotation).
- ``debug_init_pose.png`` — the forward-camera render at the initial pose.

If the render is gray, compare the NS camera position against the gate's NS
bounding box (printed at the end) — if the camera is outside the box, the
splat is offscreen and the issue is the pose chain, not the model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--frame", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image
    from falsify.geometry import Point
    from falsify.io import build_frame_graph, load_yaml
    from falsify.orchestrator.orchestrator import build_initial_state
    from falsify.sensors.camera import make_camera_sensor_from_yaml
    from falsify.sim.poses import camera_to_world_pose
    from falsify.sim.renderer import GSplatRenderer

    scene_cfg = load_yaml(args.scene)
    frame_cfg = load_yaml(args.frame)
    scene_dir = args.scene.parent

    def _resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (scene_dir / path).resolve()

    fg = build_frame_graph(scene_cfg, base_path=scene_dir)
    gsplat_config = _resolve(scene_cfg["gsplat_config_yml"])
    data_cwd = _resolve(scene_cfg["gsplat_data_cwd"]) if "gsplat_data_cwd" in scene_cfg else None
    print(f"[scene] gsplat = {gsplat_config}")
    print(f"[scene] cwd    = {data_cwd}")

    from falsify.sim.scene_edits import load_scene_edits
    renderer = GSplatRenderer(
        gsplat_config, world_frame="ned", data_cwd=data_cwd, frame_graph=fg,
        scene_edits=load_scene_edits(scene_cfg),
    )
    Tw2g = np.asarray(renderer._impl.Tw2g)
    print("\nFiGS Tw2g (NED → NS):")
    np.set_printoptions(precision=4, suppress=True)
    print(Tw2g)

    # Build initial state and camera pose.
    init_state = build_initial_state(scene_cfg, fg)
    print(f"\nInitial state pos NED  = {init_state.pos.xyz}")
    print(f"Initial state quat xyzw= {init_state.quat_xyzw}")

    fwd_yaml = frame_cfg["cameras"]["forward"]
    fwd = make_camera_sensor_from_yaml(
        "forward", fwd_yaml, fg,
        renderer=renderer.render,
        body_to_world=camera_to_world_pose,
    )
    cam_pose_ned = camera_to_world_pose(init_state, fwd.spec.body_from_camera)
    T_c2w_ned = cam_pose_ned.as_matrix()
    print("\nT_cam_to_world (NED):")
    print(T_c2w_ned)

    # Apply Tw2g to get NS-frame camera-to-world.
    T_c2g = Tw2g @ T_c2w_ned
    cam_pos_ns = T_c2g[:3, 3]
    # OpenGL convention: lens looks at -z_cam. World-frame lens direction is -col3 of R_c2g.
    lens_dir_ns = -T_c2g[:3, 2]
    lens_dir_ns_norm = lens_dir_ns / max(1e-12, np.linalg.norm(lens_dir_ns))
    print(f"\nNS-frame camera origin = {cam_pos_ns}")
    print(f"NS-frame lens direction (OpenGL -z_cam, unit) = {lens_dir_ns_norm}")

    # Gate NS bounding box from objects_summary.json (if present).
    summary_path = scene_dir / "../../data/gate_scenes_export/objects_final/objects_summary.json"
    summary_path = summary_path.resolve()
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        scene_key = scene_cfg.get("scene_key", "left_gate")
        gate_key = "left_gate" if scene_key == "left_gate" else "right_gate"
        obj = summary["objects"].get(gate_key)
        if obj is not None:
            # AABB is in joint mocap → convert to NS via FrameGraph.
            from falsify.geometry import PointCloud
            corners_m = np.array([
                obj["aabb_min"],
                obj["aabb_max"],
            ])
            pc_mocap = PointCloud(points=corners_m, frame=fg.frame("mocap"))
            pc_ns = fg.convert(pc_mocap, to="ns")
            print(f"\nGate AABB in NS: min={pc_ns.points[0]}  max={pc_ns.points[1]}")
            extent = pc_ns.points[1] - pc_ns.points[0]
            center = 0.5 * (pc_ns.points[0] + pc_ns.points[1])
            print(f"  center={center}  extent={extent}")
            # Distance from camera to gate center, and dot product with lens dir.
            cam_to_gate = center - cam_pos_ns
            dist = np.linalg.norm(cam_to_gate)
            cos_ang = float(np.dot(cam_to_gate / dist, lens_dir_ns_norm)) if dist > 0 else 0.0
            ang_deg = np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))
            print(f"\nGate is {dist:.3f} NS-units from camera "
                  f"(~{dist/Tw2g[0,0] if Tw2g[0,0]!=0 else float('nan'):.2f} m in real world).")
            print(f"Lens-to-gate-center angle = {ang_deg:.1f}°  "
                  f"(0° = perfect aim, 90° = perpendicular, 180° = backward).")

    # Render and save.
    print("\nRendering at initial pose…")
    rgb, _ = renderer.render(cam_pose_ned, fwd.spec.intrinsics)
    img = Image.fromarray(rgb)
    out_path = out_dir / "debug_init_pose.png"
    img.save(out_path)
    print(f"Saved {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
