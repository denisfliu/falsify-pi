"""Single plotly HTML showing the corrective-maneuver variants for BOTH
from_left and from_right, overlaid on the center_gate scene.

Each course has a `pre_gate` waypoint between approach and gate that is
perturbed to create corrective deviations between approach and gate.

Reads:
  runs/trajectories/through_center_gate_from_left_check/*.npz  (5 NPZs)
  runs/trajectories/through_center_gate_from_right_check/*.npz  (5 NPZs)
  configs/scenes/center_gate.yaml

Writes:
  runs/inspect/center_corrective_trajectories.html
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "runs/inspect/center_corrective_trajectories.html"
SCENE = REPO / "configs/scenes/center_gate.yaml"

SIDES = [
    ("from_left", REPO / "runs/trajectories/through_center_gate_from_left_check",
     {"center": "#444444", "up": "#1565c0", "down": "#6a1b9a",
      "left": "#2e7d32", "right": "#c62828"}),
    ("from_right", REPO / "runs/trajectories/through_center_gate_from_right_check",
     {"center": "#888888", "up": "#42a5f5", "down": "#ab47bc",
      "left": "#66bb6a", "right": "#ef5350"}),
]


def ned_to_mocap(pos_ned: np.ndarray) -> np.ndarray:
    out = pos_ned.copy()
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

    scene_cfg = load_yaml(SCENE)
    fg = build_frame_graph(scene_cfg, base_path=SCENE.parent)
    mocap = fg.frame("mocap")
    edits = load_scene_edits(scene_cfg)

    for entry in scene_cfg.get("scene_objects", []) or []:
        ply = Path(entry["ply"])
        if not ply.is_absolute():
            ply = (SCENE.parent / ply).resolve()
        cloud = read_ply(ply, mocap)
        pts = apply_edits_to_scene_object(entry["name"], cloud.points, edits, fg)
        pts = subsample(pts, 6000)
        r, g, b = [int(255 * c) for c in entry.get("color", (0.5, 0.5, 0.5))]
        traces.append(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=dict(size=1.8, color=f"rgb({r},{g},{b})", opacity=0.5),
            name=entry["name"],
        ))

    # Table top footprint (dashed) and 0.5 floor reference.
    tbl_x = [1.20, 1.85, 1.85, 1.20, 1.20]
    tbl_y = [-0.85, -0.85, 0.40, 0.40, -0.85]
    traces.append(go.Scatter3d(
        x=tbl_x, y=tbl_y, z=[0.6] * 5,
        mode="lines", line=dict(color="rgb(170,140,80)", width=4, dash="dash"),
        name="table footprint (top, z=0.6)",
    ))
    traces.append(go.Scatter3d(
        x=[1.20, 1.85], y=[0.50, 0.50], z=[1.5, 1.5],
        mode="lines", line=dict(color="rgb(0,160,0)", width=4),
        name="y=0.5 approach floor",
    ))

    # Trajectories — one trace per variant, dash style differentiates east/west.
    for side_label, npz_dir, mode_colors in SIDES:
        dash = "solid" if side_label == "from_left" else "dot"
        for npz in sorted(npz_dir.glob("*.npz")):
            d = np.load(npz, allow_pickle=True)
            pos = ned_to_mocap(d["positions_ned"])
            mode = npz.stem.rsplit("__", 1)[-1].split("_")[0]
            color = mode_colors.get(mode, "#888888")
            traces.append(go.Scatter3d(
                x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
                mode="lines+markers",
                line=dict(color=color, width=4, dash=dash),
                marker=dict(size=2.2, color=color),
                name=f"{side_label} · {mode}",
                hovertemplate=npz.stem + "<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
                legendgroup=side_label,
                legendgrouptitle_text=side_label if mode == "center" else None,
            ))

    # Start / hover_goal markers.
    for label, pt, c in [("start", (0, 0, 1.5), "rgb(0,180,0)"),
                          ("hover_goal", (1.525, -0.615, 1.0), "rgb(0,0,0)")]:
        traces.append(go.Scatter3d(
            x=[pt[0]], y=[pt[1]], z=[pt[2]],
            mode="markers+text",
            marker=dict(size=7, color=c, symbol="diamond"),
            text=[label], textposition="top center",
            name=label, showlegend=False,
        ))

    fig = go.Figure(traces)
    fig.update_layout(
        title="center gate — corrective variants of pre_gate (east solid, west dotted)"
              "<br><sub>pre_gate sits midway between approach and gate; "
              "modes center/up/down/left/right × 1 sample, magnitudes 0.1–0.3 m, seed 0</sub>",
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
    print(f"        traces: {len(traces)}")


if __name__ == "__main__":
    main()
