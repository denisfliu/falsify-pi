"""Interactive plotly viewer for one or more rollout trajectories.

Reads each run's `vla_io/query_*/data.json` (per-query drone state in NED)
and overlays the trajectories on top of the scene's `scene_objects` PLY
clouds. Writes a self-contained HTML to `--out`.

Run:

    PYTHONPATH=src python scripts/plot_rollout_trajectories.py \\
        --runs runs/v7rtc_from_left_*/ runs/v7rtc_from_right_*/ \\
        --scene configs/scenes/center_gate.yaml \\
        --out runs/v7rtc_compare.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from falsify.geometry import PointCloud
from falsify.io import build_frame_graph, load_yaml
from falsify.sim.scene_edits import apply_edits_to_scene_object, load_scene_edits
from falsify.visualization import read_ply, subsample


_TRAJ_COLORS = [
    "rgb(50,165,242)",   # blue
    "rgb(217,77,77)",    # red
    "rgb(74,196,109)",   # green
    "rgb(217,147,77)",   # orange
    "rgb(155,89,182)",   # purple
]


def _load_rollout_trajectory_ned(run_dir: Path) -> np.ndarray:
    """Read every `vla_io/query_*/data.json`'s `state_ned_pos`, return (N,3)."""
    vla_io = run_dir / "vla_io"
    qdirs = sorted([d for d in vla_io.iterdir() if d.is_dir() and d.name.startswith("query_")])
    if not qdirs:
        raise FileNotFoundError(f"no query_* dirs under {vla_io}")
    pts = []
    for q in qdirs:
        meta = json.load((q / "data.json").open())
        pts.append(meta["state_ned_pos"])
    return np.asarray(pts, dtype=np.float64)


def _convert_xyz(fg, xyz: np.ndarray, src: str, dst: str) -> np.ndarray:
    """Convert (N,3) points from src frame to dst frame via the FrameGraph."""
    if src == dst:
        return xyz
    src_frame = fg.frame(src)
    cloud = PointCloud(points=xyz.astype(np.float64), frame=src_frame)
    return fg.convert(cloud, to=dst).points


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+", required=True, type=Path,
                    help="One or more `runs/<stamp>/` directories from run_vla_episode.")
    ap.add_argument("--scene", required=True, type=Path,
                    help="Scene YAML — used to build the FrameGraph and locate scene_objects PLYs.")
    ap.add_argument("--out", required=True, type=Path,
                    help="Destination HTML.")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="Optional per-run legend labels (defaults to run directory name).")
    ap.add_argument("--max-scene-points", type=int, default=20000,
                    help="Subsample each scene PLY to keep the HTML small.")
    ap.add_argument("--view-frame", default="ned", choices=("ned", "mocap", "ns"),
                    help="Frame to plot in. Trajectories live in NED; mocap is "
                         "the gate-frame the scene_edits are authored in (and "
                         "the policy's training frame).")
    args = ap.parse_args(argv)

    if args.labels is not None and len(args.labels) != len(args.runs):
        raise SystemExit("--labels count must match --runs count")

    try:
        import plotly.graph_objects as go
    except ImportError:
        raise SystemExit("plotly not installed; `pip install plotly`")

    # ---- scene clouds, edited and converted to the view frame ---------
    scene_cfg = load_yaml(args.scene)
    scene_dir = args.scene.parent
    fg = build_frame_graph(scene_cfg, base_path=scene_dir)
    edits = load_scene_edits(scene_cfg)
    view_frame = args.view_frame

    def _resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (scene_dir / path).resolve()

    fig = go.Figure()

    for entry in scene_cfg.get("scene_objects", []):
        ply_path = _resolve(entry["ply"])
        cloud = read_ply(ply_path, fg.frame(entry["frame"]))
        if cloud.points.shape[0] > args.max_scene_points:
            cloud = subsample(cloud, args.max_scene_points)
        # Apply any scene_edits that name this object — the scene's gsplat
        # Gaussians are moved at load time, and we mirror the same rigid
        # transform on the PLY decoration so the viz matches what the
        # policy actually sees rendered.
        pts_authored = apply_edits_to_scene_object(
            entry["name"], cloud.points, edits, fg,
        )
        # `apply_edits_to_scene_object` returns points still in the edit's
        # authored frame, which is the same as the PLY's stored frame
        # (`entry["frame"]`). Convert to the view frame via the FrameGraph.
        pts_view = _convert_xyz(fg, pts_authored, src=entry["frame"], dst=view_frame)
        color = entry.get("color", (0.5, 0.5, 0.5))
        cstr = f"rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})"
        fig.add_trace(go.Scatter3d(
            x=pts_view[:, 0], y=pts_view[:, 1], z=pts_view[:, 2],
            mode="markers",
            name=f"scene/{entry['name']}",
            marker=dict(size=1.2, color=cstr, opacity=0.55),
        ))

    # ---- per-run trajectories (converted from NED → view_frame) ------
    goal_view = None
    start_view = None
    for i, run_dir in enumerate(args.runs):
        run_dir = run_dir.resolve()
        traj_ned = _load_rollout_trajectory_ned(run_dir)
        traj_view = _convert_xyz(fg, traj_ned, src="ned", dst=view_frame)
        label = args.labels[i] if args.labels else run_dir.name

        if start_view is None:
            start_view = traj_view[0]
        try:
            summary = json.load((run_dir / "episode_summary.json").open())
            if goal_view is None and summary.get("goal_ned") is not None:
                goal_view = _convert_xyz(
                    fg, np.asarray(summary["goal_ned"])[None, :], "ned", view_frame,
                )[0]
            failure = summary.get("failure")
        except FileNotFoundError:
            failure = None

        color = _TRAJ_COLORS[i % len(_TRAJ_COLORS)]
        fig.add_trace(go.Scatter3d(
            x=traj_view[:, 0], y=traj_view[:, 1], z=traj_view[:, 2],
            mode="lines+markers",
            name=label,
            line=dict(color=color, width=4),
            marker=dict(size=2, color=color),
            hovertext=[f"step {k}<br>t≈{k/30:.2f}s" for k in range(traj_view.shape[0])],
        ))
        if failure:
            ff = failure.get("type", "FAIL")
            fig.add_trace(go.Scatter3d(
                x=[traj_view[-1, 0]], y=[traj_view[-1, 1]], z=[traj_view[-1, 2]],
                mode="markers", name=f"{label} : {ff}",
                marker=dict(size=10, symbol="x", color=color),
            ))

        # Recovery trajectory (MPC replan from last_safe → goal), if present.
        rec_npz = run_dir / "recovery_trajectory.npz"
        if rec_npz.is_file():
            rec = np.load(rec_npz)
            rec_ned = rec["positions_ned"]
            rec_view = _convert_xyz(fg, rec_ned, src="ned", dst=view_frame)
            fig.add_trace(go.Scatter3d(
                x=rec_view[:, 0], y=rec_view[:, 1], z=rec_view[:, 2],
                mode="lines+markers",
                name=f"{label} : recovery",
                line=dict(color=color, width=4, dash="dash"),
                marker=dict(size=2, color=color, symbol="diamond"),
                hovertext=[f"recovery step {k}<br>t={float(rec['times'][k]):.2f}s"
                           for k in range(rec_view.shape[0])],
            ))
            # Mark the recovery's starting (= last_safe) point distinctly.
            fig.add_trace(go.Scatter3d(
                x=[rec_view[0, 0]], y=[rec_view[0, 1]], z=[rec_view[0, 2]],
                mode="markers", name=f"{label} : last_safe",
                marker=dict(size=8, color=color, symbol="circle-open"),
            ))

    if start_view is not None:
        fig.add_trace(go.Scatter3d(
            x=[start_view[0]], y=[start_view[1]], z=[start_view[2]],
            mode="markers", name="start",
            marker=dict(size=8, color="rgb(60,200,80)", symbol="diamond"),
        ))
    if goal_view is not None:
        fig.add_trace(go.Scatter3d(
            x=[goal_view[0]], y=[goal_view[1]], z=[goal_view[2]],
            mode="markers", name="goal",
            marker=dict(size=10, color="rgb(255,200,40)", symbol="diamond"),
        ))

    fig.update_layout(
        title=f"falsify rollouts ({view_frame}): {args.scene.name}",
        scene=dict(
            xaxis_title=f"x [{view_frame}]",
            yaxis_title=f"y [{view_frame}]",
            zaxis_title=f"z [{view_frame}]",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(itemsizing="constant"),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(args.out), include_plotlyjs="cdn")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
