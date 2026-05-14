"""Single plotly HTML overlaying every *_check trajectory NPZ on the union of
all three scenes' gate + table point clouds (left, right, center-edited).

Reads:
  runs/trajectories/through_left_gate_check/*.npz
  runs/trajectories/through_right_gate_check/*.npz
  runs/trajectories/through_center_gate_from_left_check/*.npz
  runs/trajectories/through_center_gate_from_right_check/*.npz

Writes:
  runs/inspect/all_check_trajectories.html
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "runs/inspect/all_check_trajectories.html"

GROUPS = [
    # (label, trajectory dir, base color hex by mode)
    ("left_gate",  REPO / "runs/trajectories/through_left_gate_check",
     {"none": "#444", "up": "#1565c0", "down": "#6a1b9a", "left": "#2e7d32", "right": "#c62828"}),
    ("right_gate", REPO / "runs/trajectories/through_right_gate_check",
     {"none": "#444", "up": "#1976d2", "down": "#8e24aa", "left": "#388e3c", "right": "#d32f2f"}),
    ("center_from_left",  REPO / "runs/trajectories/through_center_gate_from_left_check",
     {"none": "#666", "up": "#42a5f5", "down": "#ab47bc", "left": "#66bb6a", "right": "#ef5350"}),
    ("center_from_right", REPO / "runs/trajectories/through_center_gate_from_right_check",
     {"none": "#888", "up": "#64b5f6", "down": "#ba68c8", "left": "#81c784", "right": "#e57373"}),
]
DASH_BY_GROUP = {
    "left_gate":  "solid",
    "right_gate": "solid",
    "center_from_left":  "solid",
    "center_from_right": "dot",
}


def ned_to_mocap(p: np.ndarray) -> np.ndarray:
    out = p.copy()
    out[..., 1] *= -1.0
    out[..., 2] *= -1.0
    return out


def subsample(arr: np.ndarray, n: int) -> np.ndarray:
    if arr.shape[0] <= n:
        return arr
    idx = np.linspace(0, arr.shape[0] - 1, n).astype(int)
    return arr[idx]


def main():
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from falsify.io import build_frame_graph, load_yaml
    from falsify.sim.scene_edits import load_scene_edits, apply_edits_to_scene_object
    from falsify.visualization import read_ply

    traces = []

    # --- Scene clouds for all three scenes ---
    pc_specs = [
        ("configs/scenes/left_gate.yaml",   "gate",  "left_gate",   "rgb(110,140,220)"),
        ("configs/scenes/left_gate.yaml",   "table", "left_table",  "rgb(160,140,110)"),
        ("configs/scenes/right_gate.yaml",  "gate",  "right_gate",  "rgb(110,200,140)"),
        ("configs/scenes/right_gate.yaml",  "table", "right_table", "rgb(150,160,120)"),
    ]
    for scene_path_str, obj_name, display, color in pc_specs:
        scene_path = REPO / scene_path_str
        scene_cfg = load_yaml(scene_path)
        fg = build_frame_graph(scene_cfg, base_path=scene_path.parent)
        mocap = fg.frame("mocap")
        entry = next(e for e in scene_cfg["scene_objects"] if e["name"] == obj_name)
        ply = Path(entry["ply"])
        if not ply.is_absolute():
            ply = (scene_path.parent / ply).resolve()
        cloud = read_ply(ply, mocap)
        pts = subsample(cloud.points, 6000)
        traces.append(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=dict(size=1.8, color=color, opacity=0.5),
            name=display,
        ))

    # Center-edited gate.
    center_path = REPO / "configs/scenes/center_gate.yaml"
    center_cfg = load_yaml(center_path)
    center_fg = build_frame_graph(center_cfg, base_path=center_path.parent)
    mocap = center_fg.frame("mocap")
    edits = load_scene_edits(center_cfg)
    gate_entry = next(e for e in center_cfg["scene_objects"] if e["name"] == "gate")
    gate_ply = Path(gate_entry["ply"])
    if not gate_ply.is_absolute():
        gate_ply = (center_path.parent / gate_ply).resolve()
    cloud = read_ply(gate_ply, mocap)
    moved = apply_edits_to_scene_object("gate", cloud.points, edits, center_fg)
    moved = subsample(moved, 6000)
    traces.append(go.Scatter3d(
        x=moved[:, 0], y=moved[:, 1], z=moved[:, 2],
        mode="markers",
        marker=dict(size=1.8, color="rgb(220,110,140)", opacity=0.55),
        name="center_gate (edited)",
    ))

    # --- Trajectories ---
    total = 0
    for label, npz_dir, mode_colors in GROUPS:
        dash = DASH_BY_GROUP[label]
        for npz in sorted(npz_dir.glob("*.npz")):
            d = np.load(npz, allow_pickle=True)
            pos = ned_to_mocap(d["positions_ned"])
            # File stem like "through_left_gate__left_000"
            mode = npz.stem.rsplit("__", 1)[-1].split("_")[0]
            color = mode_colors.get(mode, "#888")
            traces.append(go.Scatter3d(
                x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
                mode="lines+markers",
                line=dict(color=color, width=3, dash=dash),
                marker=dict(size=2.0, color=color),
                name=f"{label} · {mode}_{npz.stem.rsplit('_', 1)[-1]}",
                hovertemplate=npz.stem + "<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
                legendgroup=label,
                legendgrouptitle_text=label,
            ))
            total += 1

    # Anchors.
    for name, pt, c in [("start", (0, 0, 1.5), "rgb(0,180,0)"),
                         ("hover_goal", (1.525, -0.615, 1.0), "rgb(0,0,0)")]:
        traces.append(go.Scatter3d(
            x=[pt[0]], y=[pt[1]], z=[pt[2]],
            mode="markers+text",
            marker=dict(size=7, color=c, symbol="diamond"),
            text=[name], textposition="top center",
            name=name, showlegend=False,
        ))

    fig = go.Figure(traces)
    fig.update_layout(
        title=f"All check trajectories ({total} total) overlaid"
              "<br><sub>colors = corrective mode (gray=none, blue=up, "
              "purple=down, green=left, red=right) · dash = center-from-right</sub>",
        scene=dict(
            xaxis_title="x_mocap (m)",
            yaxis_title="y_mocap (m)",
            zaxis_title="z_mocap (m)",
            aspectmode="data",
        ),
        legend=dict(orientation="v", x=0.01, y=0.99, groupclick="togglegroup"),
        margin=dict(l=0, r=0, t=70, b=0),
        height=900,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(OUT)
    print(f"[done] wrote {OUT}")
    print(f"        traces: {len(traces)}, trajectories: {total}")


if __name__ == "__main__":
    main()
