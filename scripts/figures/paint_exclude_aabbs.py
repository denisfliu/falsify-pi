"""Dash app to interactively paint exclude AABBs for the scene pointcloud
renderer.

Workflow
--------
1. Pick a scene, e.g. ``configs/scenes/left_and_center.yaml``. The app
   loads the same MOCAP-frame gaussian-means cloud the renderer uses
   (CPU-only: ``pipeline.model.means`` + scene_edits applied via
   ``apply_edits_to_pipeline``).
2. The 3-D scatter colors **gate-protected** points (above ``--z-cull``
   that are inside one of the scene's gate AABBs — i.e. what the
   ``--z-cull`` filter keeps) in red, and everything else in its
   gaussian DC color. Bad red blobs (stray gaussians that survived the
   z-cull because they happened to fall in the gate AABB) are exactly
   what you want to paint away.
3. Sweep the candidate exclude box with the six sliders. Points that
   would be excluded by the current box are previewed in magenta.
4. "Add to list" stamps the current box into the accumulator. "Save"
   writes the list to ``--out`` (default ``runs/figures/exclude_aabbs.json``).
5. Re-run the renderer with ``--exclude-json <that file>``.

Usage::

    PYTHONPATH=src:external/FiGS/src:external/splatnav \\
        .venv/bin/python scripts/figures/paint_exclude_aabbs.py \\
        --scene configs/scenes/left_and_center.yaml \\
        --out runs/figures/exclude_aabbs_left_and_center.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SH_C0 = 0.28209479177387814


def _resolve(rel: str, base: Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (base / p).resolve()


def _gate_aabbs_mocap(scene_cfg: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    out: list[tuple[np.ndarray, np.ndarray]] = []
    blocks: list[dict] = []
    if isinstance(scene_cfg.get("gate_region"), dict):
        blocks.append(scene_cfg["gate_region"])
    if isinstance(scene_cfg.get("gate_regions"), list):
        blocks.extend(scene_cfg["gate_regions"])
    for b in blocks:
        if b.get("aabb_frame", "mocap") != "mocap":
            continue
        out.append((np.asarray(b["aabb_min"], dtype=np.float64),
                    np.asarray(b["aabb_max"], dtype=np.float64)))
    return out


def _load_cloud(scene_yaml: Path):
    from falsify.io import load_yaml, build_frame_graph
    from falsify.sim.scene_edits import apply_edits_to_pipeline, load_scene_edits
    from nerfstudio.utils.eval_utils import eval_setup

    scene_cfg = load_yaml(scene_yaml)
    scene_dir = scene_yaml.parent
    fg = build_frame_graph(scene_cfg, base_path=scene_dir)

    gsplat_yml = _resolve(scene_cfg["gsplat_config_yml"], scene_dir)
    data_cwd = _resolve(scene_cfg["gsplat_data_cwd"], scene_dir) if "gsplat_data_cwd" in scene_cfg else None

    prev_cwd = os.getcwd()
    if data_cwd is not None:
        os.chdir(data_cwd)
    try:
        _, pipeline, _, _ = eval_setup(gsplat_yml, eval_num_rays_per_chunk=None, test_mode="test")
    finally:
        os.chdir(prev_cwd)

    edits = load_scene_edits(scene_cfg) or []
    if edits:
        apply_edits_to_pipeline(pipeline, edits, fg)

    means_ns = pipeline.model.means.detach().cpu().numpy().astype(np.float64)
    feats_dc = pipeline.model.features_dc.detach().cpu().numpy().astype(np.float32)
    rgb = np.clip(0.5 + SH_C0 * feats_dc, 0.0, 1.0)
    T = fg.transform("ns", "mocap")
    s = getattr(T, "s", 1.0)
    means_mocap = (s * (T.R @ means_ns.T)).T + T.t

    return means_mocap, rgb, scene_cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True,
                    help="Where to save the JSON list of exclude AABBs.")
    ap.add_argument("--crop", default="-1.6,-1.5,-0.1:3.5,1.5,2.3",
                    metavar="MNX,MNY,MNZ:MXX,MXY,MXZ",
                    help="Initial MOCAP AABB crop matching the renderer.")
    ap.add_argument("--z-cull", type=float, default=1.5,
                    help="Same z-cull threshold as the renderer (default 1.5).")
    ap.add_argument("--max-points", type=int, default=120_000)
    ap.add_argument("--gate-only", action="store_true",
                    help="Restrict the visible cloud to points inside any "
                         "gate AABB. Use on center_gate.yaml to paint "
                         "boxes in the gate's own frame; transformations "
                         "to other scenes happen downstream.")
    ap.add_argument("--load-from", type=Path, default=None,
                    help="Pre-populate the painter from an existing painted "
                         "JSON. Use the per-box dropdown to pull a saved box "
                         "back into the sliders for editing.")
    ap.add_argument("--port", type=int, default=8051)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    scene_yaml = (args.scene if args.scene.is_absolute()
                  else (REPO_ROOT / args.scene).resolve())

    print(f"[load] {scene_yaml.relative_to(REPO_ROOT)}")
    means_mocap, rgb, scene_cfg = _load_cloud(scene_yaml)
    print(f"[load] {means_mocap.shape[0]:,} Gaussians")
    full_means_mocap = means_mocap  # keep the un-subsampled cloud for the SAVE step

    # Apply the renderer's crop + z-cull (gate-protected) so the viewer
    # shows exactly the cloud the figure will see.
    lo, hi = args.crop.split(":")
    crop_min = np.array([float(v) for v in lo.split(",")], dtype=np.float64)
    crop_max = np.array([float(v) for v in hi.split(",")], dtype=np.float64)
    crop_mask = ((means_mocap >= crop_min) & (means_mocap <= crop_max)).all(axis=1)

    gate_aabbs = _gate_aabbs_mocap(scene_cfg)
    print(f"[load] {len(gate_aabbs)} gate AABB(s)")
    above = means_mocap[:, 2] > args.z_cull
    protected = np.zeros(means_mocap.shape[0], dtype=bool)
    for mn, mx in gate_aabbs:
        protected |= ((means_mocap >= mn) & (means_mocap <= mx)).all(axis=1)
    keep = crop_mask & ~(above & ~protected)
    if args.gate_only:
        keep &= protected
        print(f"[gate-only] restricting to gate-AABB points only")

    pts = means_mocap[keep]
    cols = rgb[keep]
    is_gate = protected[keep]   # red overlay candidates

    rng = np.random.default_rng(args.seed)
    if pts.shape[0] > args.max_points:
        idx = rng.choice(pts.shape[0], size=args.max_points, replace=False)
        pts = pts[idx]
        cols = cols[idx]
        is_gate = is_gate[idx]
    print(f"[view] showing {pts.shape[0]:,} points "
          f"({int(is_gate.sum()):,} flagged gate-protected)")

    try:
        from dash import Dash, dcc, html, Input, Output, State, no_update
        import plotly.graph_objects as go
    except ImportError as e:
        raise SystemExit("dash + plotly required (in SousVide-shared venv)") from e

    # Color scheme: under --gate-only the "gate-protected" flag is true for
    # every visible point, so the red overlay would tint the entire cloud
    # red — useless. Show DC colors instead. Outside --gate-only we keep
    # red for gate-protected so they stand out against the rest.
    if args.gate_only:
        base_colors = cols
    else:
        base_colors = np.where(
            is_gate[:, None],
            np.array([[1.0, 0.0, 0.0]]),
            cols,
        )
    base_colors_rgb = (base_colors * 255).astype(np.uint8)
    base_color_str = [f"rgb({r},{g},{b})" for r, g, b in base_colors_rgb]

    # Slider bounds = visible cloud's actual extent + a small pad so the
    # initial pose lands near the data, not off in empty crop corners.
    pad = 0.1
    if pts.shape[0] > 0:
        s_lo = pts.min(axis=0) - pad
        s_hi = pts.max(axis=0) + pad
    else:
        s_lo = crop_min - pad
        s_hi = crop_max + pad
    cloud_ctr = 0.5 * (s_lo + s_hi)
    init_half = 0.05  # cm-scale candidate; user widens as needed
    init_lo = cloud_ctr - init_half
    init_hi = cloud_ctr + init_half

    initial_boxes: list[dict] = []
    if args.load_from is not None:
        src = json.loads(args.load_from.read_text())
        if isinstance(src, list):
            initial_boxes = src
        elif isinstance(src, dict):
            initial_boxes = src.get("boxes", [])
        else:
            raise SystemExit(f"--load-from: unrecognized payload in {args.load_from}")
        print(f"[load-from] {len(initial_boxes)} pre-existing box(es)")

    # Default dropdown selection: box #1 (the user's "box 2" 1-indexed) when
    # at least 2 boxes were loaded. Just sets the dropdown — does NOT pull
    # into the sliders; the user clicks "Pull into sliders" themselves.
    default_edit_pick = 1 if len(initial_boxes) > 1 else None

    def _slider(id_, label, lo, hi, init_lo, init_hi):
        return html.Div([
            html.Label(label, style={"fontWeight": "bold"}),
            dcc.RangeSlider(
                id=id_, min=float(lo), max=float(hi),
                value=[float(init_lo), float(init_hi)],
                step=0.01, allowCross=False,
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ], style={"marginBottom": "0.6em"})

    btn_style = {
        "padding": "0.6em 1.2em",
        "marginRight": "0.6em",
        "fontSize": "14px",
        "cursor": "pointer",
    }
    add_style = {**btn_style,
                 "background": "#2b8a3e", "color": "white",
                 "border": "none", "fontWeight": "bold"}
    save_style = {**btn_style,
                  "background": "#1864ab", "color": "white",
                  "border": "none", "fontWeight": "bold"}

    app = Dash(__name__)
    app.layout = html.Div([
        html.H3(f"Exclude-AABB painter — {scene_yaml.name}"),
        html.Div([
            html.B("How to use:"), html.Ul([
                html.Li("Drag the X / Y / Z range sliders — the magenta "
                        "wireframe and magenta points show the CANDIDATE "
                        "exclude box."),
                html.Li("When the magenta points cover the junk you want "
                        "to remove, click the green ADD button. The box "
                        "is stored and its points dim to gray."),
                html.Li("Repeat for each blob you want excluded."),
                html.Li("Click SAVE when done. The renderer reads the "
                        "saved JSON via --exclude-json."),
            ], style={"marginTop": "0.3em"}),
        ], style={"background": "#fffbea", "padding": "0.8em 1em",
                  "borderLeft": "4px solid #f4d35e", "marginBottom": "1em"}),
        html.Div([
            _slider("x", "X range (mocap)", s_lo[0], s_hi[0], init_lo[0], init_hi[0]),
            _slider("y", "Y range (mocap)", s_lo[1], s_hi[1], init_lo[1], init_hi[1]),
            _slider("z", "Z range (mocap)", s_lo[2], s_hi[2], init_lo[2], init_hi[2]),
            html.Pre(id="candidate-readout",
                     style={"background": "#f4f4f4", "padding": "0.5em",
                            "marginTop": "0.5em", "fontSize": "12px"}),
        ], style={"width": "45%", "display": "inline-block",
                  "verticalAlign": "top", "padding": "0 1em"}),
        html.Div([
            dcc.Graph(id="cloud", style={"height": "75vh"}),
        ], style={"width": "54%", "display": "inline-block"}),
        html.Hr(),
        html.Div([
            html.Button("➕ ADD candidate to exclude list",
                        id="add-btn", n_clicks=0, style=add_style),
            html.Button("Undo last", id="undo-btn", n_clicks=0, style=btn_style),
            html.Button("Clear all", id="clear-btn", n_clicks=0, style=btn_style),
            html.Button(f"💾 SAVE → {args.out.name}",
                        id="save-btn", n_clicks=0, style=save_style),
        ]),
        html.Div([
            html.Label("Edit a saved box → ", style={"fontWeight": "bold",
                                                    "marginRight": "0.5em"}),
            dcc.Dropdown(id="edit-pick", options=[],
                         value=default_edit_pick,
                         placeholder="(none yet)",
                         style={"display": "inline-block",
                                "minWidth": "20em",
                                "verticalAlign": "middle"}),
            html.Button("Pull into sliders",
                        id="pull-btn", n_clicks=0,
                        style={**btn_style, "marginLeft": "0.6em"}),
        ], style={"marginTop": "0.8em"}),
        html.Pre(id="status",
                 children=(f"Pre-loaded {len(initial_boxes)} box(es) from "
                           f"{args.load_from.name if args.load_from else 'none'}.\n"
                           + json.dumps(initial_boxes, indent=2)
                           if initial_boxes else ""),
                 style={"marginTop": "1em",
                        "background": "#f4f4f4",
                        "padding": "0.8em",
                        "maxHeight": "30vh",
                        "overflowY": "auto"}),
        dcc.Store(id="boxes", data=initial_boxes),
    ])

    @app.callback(
        Output("cloud", "figure"),
        Output("candidate-readout", "children"),
        Input("x", "value"), Input("y", "value"), Input("z", "value"),
        Input("boxes", "data"),
    )
    def _update_figure(xv, yv, zv, boxes):
        # Highlight points inside the candidate box (magenta) and inside
        # any already-accumulated exclude box (gray dimmed).
        cand_mask = (
            (pts[:, 0] >= xv[0]) & (pts[:, 0] <= xv[1])
            & (pts[:, 1] >= yv[0]) & (pts[:, 1] <= yv[1])
            & (pts[:, 2] >= zv[0]) & (pts[:, 2] <= zv[1])
        )
        accum_mask = np.zeros(pts.shape[0], dtype=bool)
        for b in boxes:
            mn = np.asarray(b["min"]); mx = np.asarray(b["max"])
            accum_mask |= ((pts >= mn) & (pts <= mx)).all(axis=1)

        colors = list(base_color_str)
        for i in np.where(accum_mask)[0]:
            colors[i] = "rgb(80,80,80)"
        for i in np.where(cand_mask)[0]:
            colors[i] = "rgb(255,0,255)"

        fig = go.Figure(data=[go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=dict(size=1.4, color=colors, opacity=0.85),
            hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
            name="cloud", showlegend=False,
        )])

        def _aabb_edges(mn, mx):
            corners = np.array([
                [mn[0] if not (k & 1) else mx[0],
                 mn[1] if not (k & 2) else mx[1],
                 mn[2] if not (k & 4) else mx[2]]
                for k in range(8)
            ])
            ex_, ey_, ez_ = [], [], []
            for (i, j) in [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),
                           (3,7),(4,5),(4,6),(5,7),(6,7)]:
                a_, b_ = corners[i], corners[j]
                ex_ += [a_[0], b_[0], None]
                ey_ += [a_[1], b_[1], None]
                ez_ += [a_[2], b_[2], None]
            return ex_, ey_, ez_, corners

        # Saved boxes — each drawn with a distinct color, labeled by index.
        # Hovering an edge shows the box id so you can spot the offender.
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                   "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]
        for i, b in enumerate(boxes or []):
            color = palette[i % len(palette)]
            ex_, ey_, ez_, _ = _aabb_edges(np.asarray(b["min"]),
                                            np.asarray(b["max"]))
            fig.add_trace(go.Scatter3d(
                x=ex_, y=ey_, z=ez_, mode="lines",
                line=dict(color=color, width=4),
                name=f"#{i}", showlegend=True,
                hovertemplate=f"box #{i}<extra></extra>",
            ))

        # Candidate box wireframe (magenta on top).
        ex_, ey_, ez_, _ = _aabb_edges(np.array([xv[0], yv[0], zv[0]]),
                                        np.array([xv[1], yv[1], zv[1]]))
        fig.add_trace(go.Scatter3d(
            x=ex_, y=ey_, z=ez_, mode="lines",
            line=dict(color="magenta", width=6),
            name="candidate", showlegend=True,
            hoverinfo="skip",
        ))
        fig.update_layout(
            scene=dict(
                aspectmode="data",
                xaxis_title="x (mocap)",
                yaxis_title="y (mocap)",
                zaxis_title="z (mocap)",
            ),
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=True,
            legend=dict(itemsizing="constant"),
        )
        readout = (
            f"Candidate box (magenta):\n"
            f"  min = [{xv[0]:.3f}, {yv[0]:.3f}, {zv[0]:.3f}]\n"
            f"  max = [{xv[1]:.3f}, {yv[1]:.3f}, {zv[1]:.3f}]\n"
            f"  size = [{xv[1]-xv[0]:.3f}, {yv[1]-yv[0]:.3f}, {zv[1]-zv[0]:.3f}]\n"
            f"  enclosing {int(cand_mask.sum()):,} of {pts.shape[0]:,} visible points\n"
            f"\n"
            f"Click ADD to commit this box. {len(boxes)} box(es) saved so far."
        )
        return fig, readout

    @app.callback(
        Output("edit-pick", "options"),
        Input("boxes", "data"),
    )
    def _edit_pick_options(boxes):
        opts = []
        for i, b in enumerate(boxes or []):
            mn = b["min"]; mx = b["max"]
            opts.append({
                "label": (f"#{i}  x[{mn[0]:.2f},{mx[0]:.2f}]  "
                          f"y[{mn[1]:.2f},{mx[1]:.2f}]  "
                          f"z[{mn[2]:.2f},{mx[2]:.2f}]"),
                "value": i,
            })
        return opts

    @app.callback(
        Output("x", "value"), Output("y", "value"), Output("z", "value"),
        Output("boxes", "data", allow_duplicate=True),
        Output("edit-pick", "value"),
        Input("pull-btn", "n_clicks"),
        State("edit-pick", "value"),
        State("boxes", "data"),
        prevent_initial_call=True,
    )
    def _pull_into_sliders(_n, pick_idx, boxes):
        if pick_idx is None or boxes is None or pick_idx >= len(boxes):
            from dash import no_update
            return no_update, no_update, no_update, no_update, no_update
        chosen = boxes[pick_idx]
        new_boxes = [b for i, b in enumerate(boxes) if i != pick_idx]
        mn = chosen["min"]; mx = chosen["max"]
        return [mn[0], mx[0]], [mn[1], mx[1]], [mn[2], mx[2]], new_boxes, None

    @app.callback(
        Output("boxes", "data"),
        Output("status", "children"),
        Input("add-btn", "n_clicks"),
        Input("undo-btn", "n_clicks"),
        Input("clear-btn", "n_clicks"),
        Input("save-btn", "n_clicks"),
        State("x", "value"), State("y", "value"), State("z", "value"),
        State("boxes", "data"),
        prevent_initial_call=True,
    )
    def _edit_boxes(_add, _undo, _clear, _save, xv, yv, zv, boxes):
        from dash import ctx
        boxes = list(boxes or [])
        trig = ctx.triggered_id
        msg = ""
        if trig == "add-btn":
            boxes.append({"min": [xv[0], yv[0], zv[0]],
                          "max": [xv[1], yv[1], zv[1]]})
            msg = f"Added box #{len(boxes)}: {boxes[-1]}"
        elif trig == "undo-btn":
            if boxes:
                popped = boxes.pop()
                msg = f"Undid: {popped}"
            else:
                msg = "Nothing to undo."
        elif trig == "clear-btn":
            boxes = []
            msg = "Cleared all boxes."
        elif trig == "save-btn":
            args.out.parent.mkdir(parents=True, exist_ok=True)
            # Re-evaluate the boxes against the FULL un-subsampled cloud
            # so the saved exclude-point set catches every Gaussian inside
            # a painted box, not just the ones that survived subsampling.
            excluded_mask = np.zeros(full_means_mocap.shape[0], dtype=bool)
            for b in boxes:
                mn = np.asarray(b["min"]); mx = np.asarray(b["max"])
                excluded_mask |= ((full_means_mocap >= mn) & (full_means_mocap <= mx)).all(axis=1)
            excluded_points = full_means_mocap[excluded_mask]
            # Determine the source-gate metadata for downstream transforms.
            src_gate = None
            gates = _gate_aabbs_mocap(scene_cfg)
            blocks: list[dict] = []
            if isinstance(scene_cfg.get("gate_region"), dict):
                blocks.append(scene_cfg["gate_region"])
            if isinstance(scene_cfg.get("gate_regions"), list):
                blocks.extend(scene_cfg["gate_regions"])
            for b in blocks:
                if "anchor" in b and "normal" in b:
                    src_gate = {
                        "name": b.get("name", "gate"),
                        "anchor": list(b["anchor"]),
                        "normal": list(b["normal"]),
                    }
                    break
            payload = {
                "boxes": boxes,
                "exclude_points_mocap": excluded_points.tolist(),
                "source_gate": src_gate,
                "source_scene": scene_yaml.name,
            }
            args.out.write_text(json.dumps(payload, indent=2))
            msg = (f"Saved {len(boxes)} box(es) + "
                   f"{excluded_points.shape[0]:,} exclude point(s) to {args.out}")
        status = msg + "\n\nCurrent boxes:\n" + json.dumps(boxes, indent=2)
        return boxes, status

    print(f"[dash] http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
