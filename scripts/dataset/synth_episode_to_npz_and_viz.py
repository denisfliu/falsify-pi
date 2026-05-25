"""Pick a random episode from a synth atomic dataset, emit a MOCAP-frame
trajectory NPZ, and render an HTML visualization overlaid on the
center_gate scene point cloud.

Note: this NPZ is intentionally **not** the canonical falsify Trajectory
NPZ (which is NED-only by contract — see
``src/falsify/training/trajectory.py``). Keys here are
``positions_mocap`` / ``yaws_mocap`` so it cannot be silently fed to
``falsify-export-parquet`` (which would otherwise double-transform).

Usage:

    PYTHONPATH=src python scripts/dataset/synth_episode_to_npz_and_viz.py \\
        --dataset data/atomic_datasets/synth_center_from_left \\
        --scene configs/scenes/center_gate.yaml \\
        --out-npz runs/synth_random_episode.npz \\
        --out-html runs/synth_random_episode.html
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _gate_aabb_wireframe(aabb_min, aabb_max):
    p = np.array([
        [aabb_min[0], aabb_min[1], aabb_min[2]],
        [aabb_max[0], aabb_min[1], aabb_min[2]],
        [aabb_max[0], aabb_max[1], aabb_min[2]],
        [aabb_min[0], aabb_max[1], aabb_min[2]],
        [aabb_min[0], aabb_min[1], aabb_max[2]],
        [aabb_max[0], aabb_min[1], aabb_max[2]],
        [aabb_max[0], aabb_max[1], aabb_max[2]],
        [aabb_min[0], aabb_max[1], aabb_max[2]],
    ])
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    xs, ys, zs = [], [], []
    for a, b in edges:
        xs.extend([p[a, 0], p[b, 0], None])
        ys.extend([p[a, 1], p[b, 1], None])
        zs.extend([p[a, 2], p[b, 2], None])
    return xs, ys, zs


def _build_scene_traces(scene_yaml: Path, fg, max_cloud_points: int = 6000):
    from falsify.geometry import PointCloud
    from falsify.io import load_yaml
    from falsify.sim.scene_edits import apply_edits_to_scene_object, load_scene_edits
    from falsify.visualization import read_ply, subsample
    import plotly.graph_objects as go

    scene_cfg = load_yaml(scene_yaml)
    scene_dir = scene_yaml.parent
    edits = load_scene_edits(scene_cfg)
    traces = []

    region = scene_cfg.get("gate_region") or {}
    if region:
        xs, ys, zs = _gate_aabb_wireframe(
            np.asarray(region["aabb_min"], dtype=np.float64),
            np.asarray(region["aabb_max"], dtype=np.float64),
        )
        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(color="rgba(20,80,120,0.7)", width=3),
            name="gate AABB", legendgroup="scene", hoverinfo="skip",
        ))

    for entry in scene_cfg.get("scene_objects") or []:
        ply_path = scene_dir / entry["ply"]
        if not ply_path.is_file():
            continue
        cloud = read_ply(ply_path, fg.frame(entry["frame"]))
        cloud = subsample(cloud, max_cloud_points)
        if edits:
            new_pts = apply_edits_to_scene_object(entry["name"], cloud.points, edits, fg)
            new_colors = cloud.colors
            if new_colors is not None and new_pts.shape[0] != cloud.points.shape[0]:
                reps = new_pts.shape[0] // cloud.points.shape[0]
                new_colors = np.tile(cloud.colors, (reps, 1))
            cloud = PointCloud(points=new_pts, frame=cloud.frame, colors=new_colors)
        pts = np.asarray(cloud.points, dtype=np.float64)
        colour = entry.get("color", (0.5, 0.5, 0.5))
        rgb = f"rgb({int(255*colour[0])},{int(255*colour[1])},{int(255*colour[2])})"
        traces.append(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
            marker=dict(size=1.2, color=rgb, opacity=0.5),
            name=entry["name"], legendgroup="scene", hoverinfo="skip",
        ))

    goal = scene_cfg.get("goal_position_mocap")
    if goal:
        g = np.asarray(goal, dtype=np.float64)
        traces.append(go.Scatter3d(
            x=[g[0]], y=[g[1]], z=[g[2]], mode="markers+text",
            marker=dict(size=6, color="rgba(50,200,50,0.95)", symbol="diamond"),
            text=["goal"], textposition="top center",
            name="goal", legendgroup="scene",
        ))
    return traces


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path,
                    default=Path("data/atomic_datasets/synth_center_from_left"))
    ap.add_argument("--scene", type=Path,
                    default=Path("configs/scenes/center_gate.yaml"))
    ap.add_argument("--out-npz", type=Path, required=True)
    ap.add_argument("--out-html", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=None,
                    help="If given, pick a deterministic episode.")
    ap.add_argument("--episode", type=int, default=None,
                    help="Explicit episode index, overrides --seed.")
    args = ap.parse_args(argv)

    import pyarrow.parquet as pq
    import plotly.graph_objects as go
    import plotly.io as pio
    from falsify.io import build_frame_graph, load_yaml

    # Pick episode.
    parquets = sorted((args.dataset / "data" / "chunk-000").glob("episode_*.parquet"))
    if not parquets:
        raise SystemExit(f"no parquets found under {args.dataset}/data/chunk-000")
    if args.episode is not None:
        ep_path = args.dataset / "data" / "chunk-000" / f"episode_{args.episode:06d}.parquet"
        if not ep_path.is_file():
            raise SystemExit(f"episode {args.episode:06d} not in dataset")
    else:
        rng = random.Random(args.seed)
        ep_path = rng.choice(parquets)
    print(f"[pick] {ep_path.name}")

    # Read parquet — state[:, :4] = (x_m, y_m, z_m, yaw_m).
    table = pq.read_table(ep_path)
    states = np.stack([np.asarray(s, dtype=np.float64)
                       for s in table.column("state").to_pylist()])
    timestamps = np.asarray(table.column("timestamp").to_pylist(), dtype=np.float64)
    pos_mocap = states[:, :3]
    yaw_mocap = states[:, 3]

    # Need the FrameGraph for the scene viz (and to record the transform
    # provenance), but the trajectory itself stays in mocap.
    scene_cfg = load_yaml(args.scene)
    fg = build_frame_graph(scene_cfg, base_path=args.scene.parent)

    # Read the prompt from tasks.jsonl (single-task dataset).
    tasks_path = args.dataset / "meta" / "tasks.jsonl"
    prompt = ""
    if tasks_path.is_file():
        with tasks_path.open() as f:
            line = f.readline().strip()
        if line:
            prompt = json.loads(line).get("task", "")

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out_npz,
        times=timestamps,
        positions_mocap=pos_mocap.astype(np.float64),
        yaws_mocap=yaw_mocap.astype(np.float64),
        prompt=np.array(prompt),
        source=np.array(f"synth_replay_mocap:{args.dataset.name}/{ep_path.stem}"),
        frame=np.array("mocap"),
    )
    duration_s = float(timestamps[-1] - timestamps[0])
    print(f"[npz] wrote {args.out_npz}  N={len(timestamps)}  duration={duration_s:.2f}s  frame=mocap")

    # Visualization — trajectory in MOCAP overlaid on scene cloud.
    scene_traces = _build_scene_traces(args.scene, fg)
    fig = go.Figure()
    for tr in scene_traces:
        fig.add_trace(tr)
    fig.add_trace(go.Scatter3d(
        x=pos_mocap[:, 0], y=pos_mocap[:, 1], z=pos_mocap[:, 2],
        mode="lines",
        line=dict(color="rgb(220, 30, 30)", width=5),
        name=f"trajectory ({ep_path.stem}) — mocap",
        hovertemplate="frame %{pointNumber}<br>(%{x:.2f},%{y:.2f},%{z:.2f})<extra></extra>",
    ))
    fig.add_trace(go.Scatter3d(
        x=[pos_mocap[0, 0]], y=[pos_mocap[0, 1]], z=[pos_mocap[0, 2]],
        mode="markers", marker=dict(size=7, color="rgb(30, 130, 220)", symbol="circle",
                                     line=dict(width=1, color="black")),
        name="start",
    ))
    fig.add_trace(go.Scatter3d(
        x=[pos_mocap[-1, 0]], y=[pos_mocap[-1, 1]], z=[pos_mocap[-1, 2]],
        mode="markers", marker=dict(size=7, color="rgb(30, 180, 80)", symbol="diamond",
                                     line=dict(width=1, color="black")),
        name="end",
    ))
    fig.update_layout(
        title=f"{args.dataset.name} / {ep_path.stem}  —  "
              f"\"{prompt}\"  ({len(timestamps)} frames @ {1.0/np.median(np.diff(timestamps)):.0f} Hz, mocap)",
        scene=dict(
            xaxis=dict(title="mocap x (m)"),
            yaxis=dict(title="mocap y (m)"),
            zaxis=dict(title="mocap z (m, up)"),
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.0)),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=720,
        legend=dict(itemsizing="constant"),
    )

    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    pio.write_html(fig, str(args.out_html), include_plotlyjs="cdn", full_html=True)
    print(f"[html] wrote {args.out_html}  ({args.out_html.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
