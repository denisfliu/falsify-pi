"""Interactive box-based painting of scene-edit AABBs.

The static plotly tool (``falsify.cli.author_gaussian_mask``) shows what's
captured by your current AABBs. This Dash app is the complementary
"build the missing region by hand" tool: three MOCAP range sliders
(x, y, z) carve out a *candidate* AABB shown live in the 3D view, and
two buttons add / subtract the Gaussians inside that box to a painted
set. The bounding box of everything in the painted set is surfaced in
YAML-ready form for paste into ``scene_edits``.

Why sliders + add/subtract instead of clicks or lasso
-----------------------------------------------------
Plotly Scatter3d has no native lasso; ``clickData`` on Scatter3d works
on some browsers / renderer combinations but not others (silent
failure, no error). Range sliders are renderer-agnostic and give pixel-
precise control — useful for a workflow where the goal is a YAML AABB.

Workflow
--------
1. Launch the app against your scene; the gsplat loads once.
2. Set the x / y / z range sliders to cover the cluster you want.
   The yellow wireframe in the 3D view tracks the current candidate
   AABB; the count text shows how many unpainted Gaussians are inside.
3. Click **Add → painted** to fill the box. Painted Gaussians go
   magenta in the 3D view.
4. Re-tune the sliders and click **Add** again to extend coverage,
   or **Subtract** to remove a region from the painted set.
5. The bottom block updates live with the MOCAP AABB of the entire
   painted set, formatted as paste-ready YAML.

Each painted point lives in MOCAP; the surfaced AABB drops straight into
``rigid_transform_aabb.target_aabb_*`` (or, once we land multi-AABB,
into an ``include_aabbs`` list entry).

Example::

    PYTHONPATH=src .venv/bin/python -m falsify.cli.paint_gaussian_mask \\
        --scene configs/scenes/center_gate.yaml \\
        --port 8050
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Heavy load helpers (mirror author_gaussian_mask)
# ---------------------------------------------------------------------------


def _load_means_mocap(scene_cfg: dict, scene_dir: Path):
    """Returns (means_mocap, broad_aabbs, precise_aabbs, exclude_aabbs,
    oriented_includes, oriented_excludes).

    ``broad_aabbs`` is the per-edit main ``target_aabb_min/max`` (one box per
    edit). These boxes are exclude-subject. ``precise_aabbs`` are the
    hand-curated ``include_aabbs`` extras — they override excludes (treated
    as ground truth). Each list element is a ``(min, max, name)`` tuple.
    ``oriented_*`` are ``(_OrientedBox, name)`` tuples; oriented includes
    follow the precise / override rule.
    """
    from falsify.io import build_frame_graph
    from falsify.sim.scene_edits import load_scene_edits

    fg = build_frame_graph(scene_cfg, base_path=scene_dir)
    edits = load_scene_edits(scene_cfg)

    broad_aabbs: list[tuple[np.ndarray, np.ndarray, str]] = []
    precise_aabbs: list[tuple[np.ndarray, np.ndarray, str]] = []
    exclude_aabbs: list[tuple[np.ndarray, np.ndarray, str]] = []
    oriented_includes: list = []
    oriented_excludes: list = []
    for e in edits:
        broad_aabbs.append((
            np.asarray(e.target_aabb_min),
            np.asarray(e.target_aabb_max),
            f"{e.name}:include",
        ))
        for k, b in enumerate(e.include_aabbs):
            precise_aabbs.append((
                np.asarray(b.min), np.asarray(b.max),
                f"{e.name}:include_extra_{k}",
            ))
        for k, b in enumerate(e.exclude_aabbs):
            exclude_aabbs.append((
                np.asarray(b.min), np.asarray(b.max),
                f"{e.name}:exclude_{k}",
            ))
        for k, b in enumerate(e.oriented_include_aabbs):
            oriented_includes.append((b, f"{e.name}:oriented_include_{k}"))
        for k, b in enumerate(e.oriented_exclude_aabbs):
            oriented_excludes.append((b, f"{e.name}:oriented_exclude_{k}"))

    def _resolve(rel: str) -> Path:
        pp = Path(rel)
        return pp if pp.is_absolute() else (scene_dir / pp).resolve()

    from nerfstudio.utils.eval_utils import eval_setup
    gsplat_yml = _resolve(scene_cfg["gsplat_config_yml"])
    data_cwd = _resolve(scene_cfg["gsplat_data_cwd"]) if "gsplat_data_cwd" in scene_cfg else None

    prev_cwd = os.getcwd()
    if data_cwd is not None:
        os.chdir(data_cwd)
    try:
        _, pipeline, _, _ = eval_setup(gsplat_yml, eval_num_rays_per_chunk=None, test_mode="test")
    finally:
        os.chdir(prev_cwd)

    means_ns = pipeline.model.means.detach().cpu().numpy().astype(np.float64)

    # NS → MOCAP
    T = fg.transform("ns", "mocap")
    s = getattr(T, "s", 1.0)
    means_mocap = (s * (T.R @ means_ns.T)).T + T.t
    return (means_mocap, broad_aabbs, precise_aabbs, exclude_aabbs,
            oriented_includes, oriented_excludes)


def _mask_inside(points: np.ndarray, mn: np.ndarray, mx: np.ndarray) -> np.ndarray:
    return ((points >= mn) & (points <= mx)).all(axis=1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scene", required=True, type=Path)
    p.add_argument("--port", type=int, default=8050,
                   help="Dash server port (default 8050).")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--max-points", type=int, default=60000,
                   help="Subsample after the neighborhood crop (default 60000).")
    p.add_argument("--neighborhood", default=None, metavar="[mn]:[mx]",
                   help="Crop visualization to this AABB in MOCAP. Default: union of "
                        "existing AABBs + 0.4 m buffer.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    # Lazy import — Dash + plotly.
    try:
        from dash import Dash, dcc, html, Input, Output, State, Patch, ALL, no_update
        import plotly.graph_objects as go
    except ImportError as e:
        raise SystemExit(
            "dash and plotly required. Both are in the SousVide-shared venv; "
            "if missing, run `uv sync`."
        ) from e

    from falsify.io import load_yaml

    scene_cfg = load_yaml(args.scene)
    scene_dir = args.scene.parent

    (means_mocap, broad_aabbs, precise_aabbs, exclude_aabbs,
     oriented_includes, oriented_excludes) = _load_means_mocap(scene_cfg, scene_dir)
    # Neighborhood + downstream code work on the union of all boxes:
    include_aabbs = broad_aabbs + precise_aabbs
    print(f"[gsplat]  loaded {means_mocap.shape[0]:,} Gaussians")

    # Neighborhood crop.
    if args.neighborhood is not None:
        a, b = args.neighborhood.split(":", 1)
        import json
        nb_min = np.asarray(json.loads(a), dtype=np.float64)
        nb_max = np.asarray(json.loads(b), dtype=np.float64)
    else:
        all_boxes = include_aabbs + exclude_aabbs
        if all_boxes:
            nb_min = np.min([b[0] for b in all_boxes], axis=0) - 0.4
            nb_max = np.max([b[1] for b in all_boxes], axis=0) + 0.4
        else:
            nb_min = means_mocap.min(axis=0)
            nb_max = means_mocap.max(axis=0)

    in_nb = _mask_inside(means_mocap, nb_min, nb_max)
    means_mocap = means_mocap[in_nb]
    print(f"[nb]      {means_mocap.shape[0]:,} Gaussians in neighborhood "
          f"min={nb_min.tolist()}  max={nb_max.tolist()}")

    if means_mocap.shape[0] > args.max_points:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(means_mocap.shape[0], size=args.max_points, replace=False)
        idx.sort()
        means_mocap = means_mocap[idx]
    print(f"[sub]     showing {means_mocap.shape[0]:,} points")

    # Pre-classify against current AABBs.  Mask semantics mirror the applier:
    #   move = (broad ∩ ¬exclude) ∪ precise
    # Precise inclusions (hand-curated) override excludes — they are treated
    # as ground truth.
    in_broad = np.zeros(means_mocap.shape[0], dtype=bool)
    for mn, mx, _ in broad_aabbs:
        in_broad |= _mask_inside(means_mocap, mn, mx)
    in_precise = np.zeros(means_mocap.shape[0], dtype=bool)
    for mn, mx, _ in precise_aabbs:
        in_precise |= _mask_inside(means_mocap, mn, mx)
    for ob, _ in oriented_includes:
        in_precise |= ob.contains(means_mocap)
    in_any_exc = np.zeros(means_mocap.shape[0], dtype=bool)
    for mn, mx, _ in exclude_aabbs:
        in_any_exc |= _mask_inside(means_mocap, mn, mx)
    for ob, _ in oriented_excludes:
        in_any_exc |= ob.contains(means_mocap)

    in_any_inc = in_broad | in_precise
    # Applier-true move mask: broad carved by excludes, then precise unioned in.
    move_mask = (in_broad & ~in_any_exc) | in_precise

    cyan_mask  = move_mask                          # will move
    red_mask   = in_broad & in_any_exc & ~in_precise  # stranded (broad ∩ exclude, no precise rescue)
    orange_mask= ~in_any_inc & in_any_exc           # in exclude only
    gray_mask  = ~in_any_inc & ~in_any_exc          # outside everything

    # Slider extents (MOCAP). Each slider can paint anywhere in the
    # neighborhood; user can also pan outside via direct text entry on
    # the tooltip if they need to.
    extent_min = means_mocap.min(axis=0).tolist()
    extent_max = means_mocap.max(axis=0).tolist()
    pad = 0.05
    sld_x = (float(extent_min[0]) - pad, float(extent_max[0]) + pad)
    sld_y = (float(extent_min[1]) - pad, float(extent_max[1]) + pad)
    sld_z = (float(extent_min[2]) - pad, float(extent_max[2]) + pad)

    # Sensible defaults for the candidate box: a 30 cm-wide slab roughly
    # around the centroid of the gray cluster (i.e. somewhere likely to
    # contain stragglers).
    if gray_mask.any():
        gc = means_mocap[gray_mask].mean(axis=0)
    else:
        gc = means_mocap.mean(axis=0)
    half = 0.15
    init_x = [float(gc[0] - half), float(gc[0] + half)]
    init_y = [float(gc[1] - half), float(gc[1] + half)]
    init_z = [sld_z[0], min(sld_z[1], 0.5)]

    # ---- Dash app -----------------------------------------------------
    app = Dash(__name__)

    # Inject CSS so the dcc.RangeSlider tooltips + value marks are readable
    # against the dark page background. Dash's defaults are tuned for a
    # light theme and end up white-on-white in our layout.
    app.index_string = """<!DOCTYPE html>
<html>
<head>
{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
  body { background-color: black; color: white; }
  /* Slider track + handles (Dash 4 / rc-slider names) */
  .rc-slider-rail { background-color: #333 !important; }
  .rc-slider-track { background-color: #06f !important; }
  .rc-slider-handle { border-color: #fff !important; background-color: #06f !important; }
  /* Tick-mark labels under the sliders */
  .rc-slider-mark-text { color: #ccc !important; }
  .rc-slider-mark-text-active { color: #fff !important; }
  /* The hovering tooltip (only when not using allow_direct_input) */
  .rc-slider-tooltip-inner {
    background-color: #222 !important;
    color: #fff !important;
    border: 1px solid #555 !important;
    border-radius: 3px;
    font-family: monospace !important;
    padding: 4px 6px !important;
  }
  .rc-slider-tooltip-arrow {
    border-top-color: #222 !important;
    border-bottom-color: #222 !important;
  }
  /* The editable input boxes added by allow_direct_input — make text
     readable on dark theme. */
  .rc-slider input,
  .rc-slider input[type="number"],
  input[class*="rc-slider"] {
    background-color: #222 !important;
    color: #fff !important;
    border: 1px solid #555 !important;
    border-radius: 3px !important;
    padding: 2px 4px !important;
    font-family: monospace !important;
    width: 4em !important;
  }
</style>
</head>
<body>
{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>"""

    initial_painted: list[int] = []

    # Cached AABB-wireframe traces for the *existing* edit AABBs so we
    # don't rebuild them every callback.
    def _aabb_trace(mn, mx, name, color, dash="solid", width=4):
        x0, y0, z0 = mn; x1, y1, z1 = mx
        edges = [
            ((x0,y0,z0),(x1,y0,z0)),((x1,y0,z0),(x1,y1,z0)),
            ((x1,y1,z0),(x0,y1,z0)),((x0,y1,z0),(x0,y0,z0)),
            ((x0,y0,z1),(x1,y0,z1)),((x1,y0,z1),(x1,y1,z1)),
            ((x1,y1,z1),(x0,y1,z1)),((x0,y1,z1),(x0,y0,z1)),
            ((x0,y0,z0),(x0,y0,z1)),((x1,y0,z0),(x1,y0,z1)),
            ((x1,y1,z0),(x1,y1,z1)),((x0,y1,z0),(x0,y1,z1)),
        ]
        xs, ys, zs = [], [], []
        for a, b in edges:
            xs += [a[0], b[0], None]; ys += [a[1], b[1], None]; zs += [a[2], b[2], None]
        return go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(color=color, width=width, dash=dash),
            name=name, hoverinfo="skip",
        )

    def _obox_edges_trace(box, name, color, dash="solid", width=4):
        # Reuses the candidate-box corner builder defined below by replicating
        # the rotation math inline (this closure runs before the candidate
        # builder is defined).
        c, s = np.cos(box.yaw), np.sin(box.yaw)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
        h = box.half_extents
        local = np.array([
            [-h[0], -h[1], -h[2]], [+h[0], -h[1], -h[2]],
            [-h[0], +h[1], -h[2]], [+h[0], +h[1], -h[2]],
            [-h[0], -h[1], +h[2]], [+h[0], -h[1], +h[2]],
            [-h[0], +h[1], +h[2]], [+h[0], +h[1], +h[2]],
        ])
        corners = (R @ local.T).T + box.center
        edge_idx = [
            (0, 1), (1, 3), (3, 2), (2, 0),
            (4, 5), (5, 7), (7, 6), (6, 4),
            (0, 4), (1, 5), (3, 7), (2, 6),
        ]
        xs, ys, zs = [], [], []
        for a, b in edge_idx:
            xs += [corners[a, 0], corners[b, 0], None]
            ys += [corners[a, 1], corners[b, 1], None]
            zs += [corners[a, 2], corners[b, 2], None]
        return go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(color=color, width=width, dash=dash),
            name=name, hoverinfo="skip",
        )

    edit_aabb_traces = (
        [_aabb_trace(mn, mx, n, "cyan") for mn, mx, n in include_aabbs]
        + [_obox_edges_trace(b, n, "cyan", dash="dot") for b, n in oriented_includes]
        + [_aabb_trace(mn, mx, n, "orange", dash="dash") for mn, mx, n in exclude_aabbs]
        + [_obox_edges_trace(b, n, "orange", dash="dashdot") for b, n in oriented_excludes]
    )

    def _candidate_obb_corners(box_min, box_max, yaw):
        """8 corners of the candidate box yawed about its xy-centre."""
        mn = np.asarray(box_min, dtype=np.float64)
        mx = np.asarray(box_max, dtype=np.float64)
        centre = 0.5 * (mn + mx)
        half = 0.5 * (mx - mn)
        local = np.array([
            [-half[0], -half[1], -half[2]], [+half[0], -half[1], -half[2]],
            [-half[0], +half[1], -half[2]], [+half[0], +half[1], -half[2]],
            [-half[0], -half[1], +half[2]], [+half[0], -half[1], +half[2]],
            [-half[0], +half[1], +half[2]], [+half[0], +half[1], +half[2]],
        ])
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
        return (R @ local.T).T + centre

    def _candidate_obb_mask(box_min, box_max, yaw, points):
        """Mask of points inside the yawed candidate box."""
        mn = np.asarray(box_min, dtype=np.float64)
        mx = np.asarray(box_max, dtype=np.float64)
        centre = 0.5 * (mn + mx)
        half = 0.5 * (mx - mn)
        c, s = np.cos(-yaw), np.sin(-yaw)
        R_inv = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
        local = (R_inv @ (points - centre).T).T
        return (np.abs(local) <= half).all(axis=1)

    def _candidate_obb_edge_trace(box_min, box_max, yaw, name="candidate",
                                  color="yellow", width=6):
        corners = _candidate_obb_corners(box_min, box_max, yaw)
        edge_idx = [
            (0, 1), (1, 3), (3, 2), (2, 0),
            (4, 5), (5, 7), (7, 6), (6, 4),
            (0, 4), (1, 5), (3, 7), (2, 6),
        ]
        xs, ys, zs = [], [], []
        for a, b in edge_idx:
            xs += [corners[a, 0], corners[b, 0], None]
            ys += [corners[a, 1], corners[b, 1], None]
            zs += [corners[a, 2], corners[b, 2], None]
        return go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(color=color, width=width),
            name=name, hoverinfo="skip",
        )

    def _build_traces(painted: list[int], box_min, box_max, yaw) -> list:
        """Return the ordered trace list for the 3D figure. Used both for
        the initial figure and for partial-update Patches."""
        painted_set = np.zeros(len(means_mocap), dtype=bool)
        if painted:
            painted_set[np.asarray(painted, dtype=int)] = True

        in_box_mask = _candidate_obb_mask(box_min, box_max, yaw, means_mocap)

        def _scatter(mask, name, color, size=2):
            sub_mask = mask & ~painted_set
            sub = means_mocap[sub_mask]
            return go.Scatter3d(
                x=sub[:, 0], y=sub[:, 1], z=sub[:, 2],
                mode="markers",
                marker=dict(size=size, color=color, opacity=0.45),
                name=f"{name} ({sub.shape[0]})",
                hovertemplate=(name + "<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>"),
            )

        traces = [
            _scatter(cyan_mask,   "moved",       "cyan"),
            _scatter(red_mask,    "stranded",    "red"),
            _scatter(orange_mask, "in-exclude",  "orange"),
            _scatter(gray_mask,   "outside",     "lightgray", size=1.5),
        ]
        # Painted set (magenta).
        if painted_set.any():
            sub = means_mocap[painted_set]
            traces.append(go.Scatter3d(
                x=sub[:, 0], y=sub[:, 1], z=sub[:, 2],
                mode="markers",
                marker=dict(size=3, color="magenta", opacity=0.95),
                name=f"painted ({sub.shape[0]})",
                hovertemplate=("painted<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>"),
            ))
        else:
            traces.append(go.Scatter3d(x=[], y=[], z=[], mode="markers",
                                       name="painted (0)", showlegend=True))

        # In-box highlight (yellow).
        in_box_unpainted = in_box_mask & ~painted_set
        if in_box_unpainted.any():
            sub = means_mocap[in_box_unpainted]
            traces.append(go.Scatter3d(
                x=sub[:, 0], y=sub[:, 1], z=sub[:, 2],
                mode="markers",
                marker=dict(size=3.5, color="yellow", opacity=0.9,
                            line=dict(color="black", width=0.5)),
                name=f"in candidate box ({sub.shape[0]})",
                hovertemplate=("in candidate box<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>"),
            ))
        else:
            traces.append(go.Scatter3d(x=[], y=[], z=[], mode="markers",
                                       name="in candidate box (0)", showlegend=True))

        traces.extend(edit_aabb_traces)

        # The candidate (possibly yawed) box wireframe (yellow).
        traces.append(_candidate_obb_edge_trace(box_min, box_max, yaw))
        return traces

    def _build_initial_figure() -> go.Figure:
        fig = go.Figure(data=_build_traces(
            [],
            [init_x[0], init_y[0], init_z[0]],
            [init_x[1], init_y[1], init_z[1]],
            0.0,
        ))
        fig.update_layout(
            title="Slide x/y/z to frame a box (yellow). Add → fill, Subtract → carve.",
            scene=dict(
                xaxis_title="x_mocap", yaxis_title="y_mocap", zaxis_title="z_mocap",
                aspectmode="data", bgcolor="black",
            ),
            paper_bgcolor="black", font=dict(color="white"),
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(itemsizing="constant"),
        )
        return fig

    def _slider(label_id, value, lo, hi):
        # Pass styled-marks (the dict-per-tick form) so each tick label
        # carries its own white `style` — that bypasses any CSS rule
        # specificity issues with Dash 4's class names.
        marks = {
            float(round(v, 2)): {
                "label": f"{v:.1f}",
                "style": {"color": "#ddd", "fontFamily": "monospace"},
            }
            for v in np.linspace(lo, hi, 7)
        }
        return dcc.RangeSlider(
            id=label_id, min=lo, max=hi, step=0.01, value=value, marks=marks,
            allow_direct_input=True,
            tooltip={"placement": "bottom", "always_visible": False,
                     "style": {"backgroundColor": "#222", "color": "#fff",
                               "border": "1px solid #555", "padding": "3px 6px",
                               "fontFamily": "monospace"}},
        )

    app.layout = html.Div(
        style={"backgroundColor": "black", "color": "white",
               "fontFamily": "monospace", "padding": "10px"},
        children=[
            html.H2("Gaussian-mask paint (box → painted)"),
            html.Div([
                "Scene: ", html.Code(str(args.scene)),
                "    Gaussians (subsampled): ", html.Code(f"{means_mocap.shape[0]:,}"),
            ]),
            html.Hr(),
            html.Div([
                html.Span("x range (MOCAP):  "),
                html.Span(id="sld-x-readout", style={"color": "#fff",
                          "fontFamily": "monospace", "fontWeight": "bold"}),
            ]),
            _slider("sld-x", init_x, sld_x[0], sld_x[1]),
            html.Div([
                html.Span("y range (MOCAP):  "),
                html.Span(id="sld-y-readout", style={"color": "#fff",
                          "fontFamily": "monospace", "fontWeight": "bold"}),
            ], style={"marginTop": "16px"}),
            _slider("sld-y", init_y, sld_y[0], sld_y[1]),
            html.Div([
                html.Span("z range (MOCAP):  "),
                html.Span(id="sld-z-readout", style={"color": "#fff",
                          "fontFamily": "monospace", "fontWeight": "bold"}),
            ], style={"marginTop": "16px"}),
            _slider("sld-z", init_z, sld_z[0], sld_z[1]),
            html.Div([
                html.Span("yaw (rad about z, CCW about box xy-centre):  "),
                html.Span(id="sld-yaw-readout", style={"color": "#fff",
                          "fontFamily": "monospace", "fontWeight": "bold"}),
            ], style={"marginTop": "16px"}),
            dcc.Slider(
                id="sld-yaw", min=-np.pi, max=np.pi, step=0.01, value=0.0,
                marks={float(round(v, 2)): {
                    "label": f"{v:.2f}",
                    "style": {"color": "#ddd", "fontFamily": "monospace"},
                } for v in np.linspace(-np.pi, np.pi, 9)},
                allow_direct_input=True,
            ),
            html.Div(style={"display": "flex", "gap": "10px", "marginTop": "14px"}, children=[
                html.Button("Add box → painted", id="btn-add", n_clicks=0,
                            style={"backgroundColor": "#0a5", "color": "white"}),
                html.Button("Subtract box from painted", id="btn-sub", n_clicks=0,
                            style={"backgroundColor": "#a50", "color": "white"}),
                html.Button("Undo", id="btn-undo", n_clicks=0),
                html.Button("Clear painted", id="btn-clear", n_clicks=0),
                html.Div(id="box-count", style={"marginLeft": "20px", "alignSelf": "center"}),
            ]),
            dcc.Graph(id="fig3d", style={"height": "740px"},
                      figure=_build_initial_figure()),
            html.Hr(),
            html.Div("Painted bounding box (paste-ready for scene_edits):"),
            html.Pre(id="aabb-out", style={
                "backgroundColor": "#101010", "padding": "10px",
                "border": "1px solid #333", "whiteSpace": "pre-wrap",
            }),
            dcc.Store(id="painted-store", data=initial_painted),
            dcc.Store(id="painted-history", data=[]),
        ],
    )

    def _box_indices(box_min, box_max, yaw):
        return np.where(
            _candidate_obb_mask(box_min, box_max, yaw, means_mocap)
        )[0]

    @app.callback(
        Output("sld-x-readout", "children"),
        Input("sld-x", "value"),
    )
    def _x_readout(v):
        return f"[{v[0]:.3f}, {v[1]:.3f}]"

    @app.callback(
        Output("sld-y-readout", "children"),
        Input("sld-y", "value"),
    )
    def _y_readout(v):
        return f"[{v[0]:.3f}, {v[1]:.3f}]"

    @app.callback(
        Output("sld-z-readout", "children"),
        Input("sld-z", "value"),
    )
    def _z_readout(v):
        return f"[{v[0]:.3f}, {v[1]:.3f}]"

    @app.callback(
        Output("sld-yaw-readout", "children"),
        Input("sld-yaw", "value"),
    )
    def _yaw_readout(v):
        return f"{v:+.3f} rad   ({np.degrees(v):+6.1f}°)"

    @app.callback(
        Output("box-count", "children"),
        Input("sld-x", "value"),
        Input("sld-y", "value"),
        Input("sld-z", "value"),
        Input("sld-yaw", "value"),
        Input("painted-store", "data"),
    )
    def _box_count(xv, yv, zv, yaw, painted):
        idx = _box_indices([xv[0], yv[0], zv[0]], [xv[1], yv[1], zv[1]], yaw or 0.0)
        n_total = len(idx)
        if painted:
            painted_set = set(painted)
            n_painted = sum(1 for i in idx if int(i) in painted_set)
        else:
            n_painted = 0
        return (f"box contains {n_total:,} Gaussians  "
                f"({n_total - n_painted:,} unpainted, {n_painted:,} already painted)")

    @app.callback(
        Output("painted-store", "data"),
        Output("painted-history", "data"),
        Input("btn-add", "n_clicks"),
        Input("btn-sub", "n_clicks"),
        Input("btn-undo", "n_clicks"),
        Input("btn-clear", "n_clicks"),
        State("sld-x", "value"),
        State("sld-y", "value"),
        State("sld-z", "value"),
        State("sld-yaw", "value"),
        State("painted-store", "data"),
        State("painted-history", "data"),
        prevent_initial_call=True,
    )
    def _update_painted(n_add, n_sub, n_undo, n_clear,
                        xv, yv, zv, yaw, painted, history):
        from dash import ctx
        trigger = ctx.triggered_id
        painted = painted or []
        history = history or []

        if trigger == "btn-clear":
            return [], history + [list(painted)]
        if trigger == "btn-undo":
            if history:
                return history[-1], history[:-1]
            return painted, history

        idx = _box_indices([xv[0], yv[0], zv[0]], [xv[1], yv[1], zv[1]], yaw or 0.0)
        idx_set = set(int(i) for i in idx)
        if trigger == "btn-add":
            new_painted = sorted(set(painted) | idx_set)
        elif trigger == "btn-sub":
            new_painted = sorted(set(painted) - idx_set)
        else:
            return painted, history
        return new_painted, history + [list(painted)]

    @app.callback(
        Output("fig3d", "figure"),
        Input("painted-store", "data"),
        Input("sld-x", "value"),
        Input("sld-y", "value"),
        Input("sld-z", "value"),
        Input("sld-yaw", "value"),
        prevent_initial_call=True,
    )
    def _refresh_3d(painted, xv, yv, zv, yaw):
        # Patch updates only `data` so layout (camera) stays put.
        new_traces = _build_traces(
            painted or [],
            [xv[0], yv[0], zv[0]],
            [xv[1], yv[1], zv[1]],
            yaw or 0.0,
        )
        patch = Patch()
        patch["data"] = new_traces
        return patch

    @app.callback(
        Output("aabb-out", "children"),
        Input("painted-store", "data"),
        Input("sld-x", "value"),
        Input("sld-y", "value"),
        Input("sld-z", "value"),
        Input("sld-yaw", "value"),
    )
    def _show_aabb(painted, xv, yv, zv, yaw):
        yaw = yaw or 0.0
        out_lines = []

        # 1. Painted set summary (axis-aligned AABB of accumulated points).
        if painted:
            pts = means_mocap[np.asarray(painted, dtype=int)]
            mn = pts.min(axis=0); mx = pts.max(axis=0)
            out_lines += [
                f"# {len(painted)} painted Gaussians",
                f"# MOCAP AABB (axis-aligned bracket of painted points):",
                f"min: [{mn[0]:.3f}, {mn[1]:.3f}, {mn[2]:.3f}]",
                f"max: [{mx[0]:.3f}, {mx[1]:.3f}, {mx[2]:.3f}]",
                f"",
                f"# Paste-ready (scene_edits → include_aabbs / exclude_aabbs):",
                f"  - {{ min: [{mn[0]:.3f}, {mn[1]:.3f}, {mn[2]:.3f}], "
                f"max: [{mx[0]:.3f}, {mx[1]:.3f}, {mx[2]:.3f}] }}",
                f"",
            ]
        else:
            out_lines += [
                "(no painted points yet — set sliders to frame a region "
                "and click \"Add box → painted\")",
                "",
            ]

        # 2. Current candidate (oriented) box — paste-ready oriented form,
        # useful when the painted region is best captured as a single
        # yawed strip rather than a loose AABB.
        cmn = [float(xv[0]), float(yv[0]), float(zv[0])]
        cmx = [float(xv[1]), float(yv[1]), float(zv[1])]
        centre = [0.5 * (cmn[i] + cmx[i]) for i in range(3)]
        half = [0.5 * (cmx[i] - cmn[i]) for i in range(3)]
        out_lines += [
            f"# Current candidate box (the yellow wireframe):",
            f"#   centre = [{centre[0]:.3f}, {centre[1]:.3f}, {centre[2]:.3f}]",
            f"#   half_extents = [{half[0]:.3f}, {half[1]:.3f}, {half[2]:.3f}]",
            f"#   yaw = {yaw:+.4f} rad ({np.degrees(yaw):+6.1f}°)",
            f"#",
            f"# Paste-ready (scene_edits → oriented_include_aabbs):",
            f"  - {{ center: [{centre[0]:.3f}, {centre[1]:.3f}, {centre[2]:.3f}], "
            f"half_extents: [{half[0]:.3f}, {half[1]:.3f}, {half[2]:.3f}], "
            f"yaw: {yaw:.4f} }}",
        ]
        return "\n".join(out_lines)

    print(f"[dash]    http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
