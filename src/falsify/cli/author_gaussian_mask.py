"""Interactive plotly tool for authoring scene-edit AABBs.

Loads a scene's gsplat, colors each Gaussian mean by which AABB it falls
in (per ``scene_edits`` entry), and renders a single HTML you can rotate /
hover. Used to dial in the include / exclude AABBs for a
``rigid_transform_aabb`` edit without trial-rendering every time.

Classification per Gaussian (per edit)::

  cyan    →  inside include AND outside all excludes (will be moved)
  red     →  inside include AND inside an exclude  (will be stranded — visible bug)
  orange  →  inside an exclude only                 (e.g. table; correctly left alone)
  gray    →  outside everything (hidden by default; toggle in the legend)

Hover any point to read its MOCAP coordinates plus the classification.

Iterate without editing the scene YAML
--------------------------------------
``--add-include "[xmin,ymin,zmin]:[xmax,ymax,zmax]"`` (repeatable) adds a
candidate inclusion AABB on top of whatever's in the scene's
``scene_edits``. Same for ``--add-exclude``. The candidates are tagged in
the legend so you can see what changed.

When the candidate set looks right, copy the AABBs into the scene YAML
and drop the ``--add-*`` flags.

Example::

    PYTHONPATH=src .venv/bin/python -m falsify.cli.author_gaussian_mask \\
        --scene configs/scenes/center_gate.yaml \\
        --out runs/inspect/mask_center_gate.html

    # Try a wider include AABB that catches the L-foot:
    PYTHONPATH=src .venv/bin/python -m falsify.cli.author_gaussian_mask \\
        --scene configs/scenes/center_gate.yaml \\
        --add-include "[0.36, 0.40, 0.05]:[1.20, 1.12, 0.30]" \\
        --out runs/inspect/mask_center_gate.html
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Plotly helpers
# ---------------------------------------------------------------------------


def _aabb_edges_trace(aabb_min: np.ndarray, aabb_max: np.ndarray, name: str, color: str, dash: str = "solid"):
    import plotly.graph_objects as go
    x0, y0, z0 = aabb_min
    x1, y1, z1 = aabb_max
    edges = [
        ((x0, y0, z0), (x1, y0, z0)), ((x1, y0, z0), (x1, y1, z0)),
        ((x1, y1, z0), (x0, y1, z0)), ((x0, y1, z0), (x0, y0, z0)),
        ((x0, y0, z1), (x1, y0, z1)), ((x1, y0, z1), (x1, y1, z1)),
        ((x1, y1, z1), (x0, y1, z1)), ((x0, y1, z1), (x0, y0, z1)),
        ((x0, y0, z0), (x0, y0, z1)), ((x1, y0, z0), (x1, y0, z1)),
        ((x1, y1, z0), (x1, y1, z1)), ((x0, y1, z0), (x0, y1, z1)),
    ]
    xs, ys, zs = [], [], []
    for a, b in edges:
        xs += [a[0], b[0], None]
        ys += [a[1], b[1], None]
        zs += [a[2], b[2], None]
    return go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines",
        line=dict(color=color, width=4, dash=dash),
        name=name, hoverinfo="skip",
    )


def _scatter(points: np.ndarray, label: str, color: str, *, size: float = 2.0, visible="legendonly"):
    import plotly.graph_objects as go
    if len(points) == 0:
        return go.Scatter3d(
            x=[], y=[], z=[], mode="markers", name=f"{label} (0)",
            marker=dict(size=size, color=color),
            visible=visible,
        )
    return go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode="markers",
        marker=dict(size=size, color=color, opacity=0.75),
        name=f"{label} ({len(points)})",
        visible=visible,
        hovertemplate=(label + "<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>"),
    )


# ---------------------------------------------------------------------------
# AABB parsing
# ---------------------------------------------------------------------------


def _parse_aabb_arg(s: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse '[xmin,ymin,zmin]:[xmax,ymax,zmax]' → (min, max)."""
    if ":" not in s:
        raise SystemExit(f"AABB must look like '[xmin,ymin,zmin]:[xmax,ymax,zmax]'; got {s!r}")
    a, b = s.split(":", 1)
    mn = np.asarray(json.loads(a), dtype=np.float64)
    mx = np.asarray(json.loads(b), dtype=np.float64)
    if mn.shape != (3,) or mx.shape != (3,):
        raise SystemExit(f"AABB min/max must each have 3 floats; got {mn.shape}/{mx.shape}")
    if (mn >= mx).any():
        raise SystemExit(f"AABB min must be < max element-wise; got {mn.tolist()} vs {mx.tolist()}")
    return mn, mx


def _mask_inside(points: np.ndarray, mn: np.ndarray, mx: np.ndarray) -> np.ndarray:
    return ((points >= mn) & (points <= mx)).all(axis=1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scene", required=True, type=Path,
                   help="Scene YAML to load (its gsplat + scene_edits are the baseline).")
    p.add_argument("--out", required=True, type=Path,
                   help="Output HTML path.")
    p.add_argument("--edit-name", default=None,
                   help="When the scene has multiple scene_edits, classify against "
                        "this one (default: the first edit).")
    p.add_argument("--add-include", action="append", default=[], metavar="AABB",
                   help="Candidate include AABB '[mn]:[mx]' (repeatable; unioned with the edit's includes).")
    p.add_argument("--add-exclude", action="append", default=[], metavar="AABB",
                   help="Candidate exclude AABB '[mn]:[mx]' (repeatable; unioned with the edit's excludes).")
    p.add_argument("--neighborhood", default=None, metavar="AABB",
                   help="Crop the visualization to this AABB in MOCAP. Default: AABB "
                        "around the union of all include/exclude boxes plus a 0.4 m buffer.")
    p.add_argument("--max-points", type=int, default=40000,
                   help="Subsample to at most this many points after neighborhood crop.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--show-outside", action="store_true",
                   help="Default-show the 'outside everything' gray cloud (off by default).")
    args = p.parse_args(argv)

    # Lazy imports.
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise SystemExit("plotly required (already in pyproject; run uv sync)") from e

    from falsify.io import build_frame_graph, load_yaml
    from falsify.sim.scene_edits import load_scene_edits
    from nerfstudio.utils.eval_utils import eval_setup

    scene_cfg = load_yaml(args.scene)
    scene_dir = args.scene.parent
    fg = build_frame_graph(scene_cfg, base_path=scene_dir)
    edits = load_scene_edits(scene_cfg)

    # Pick the edit to classify against.
    selected_edit = None
    if edits:
        if args.edit_name is None:
            selected_edit = edits[0]
        else:
            for e in edits:
                if e.name == args.edit_name:
                    selected_edit = e
                    break
            if selected_edit is None:
                raise SystemExit(
                    f"--edit-name {args.edit_name!r} not in scene's edits: "
                    f"{[e.name for e in edits]}"
                )

    # Gather inclusion / exclusion AABBs in MOCAP.
    include_aabbs: list[tuple[np.ndarray, np.ndarray, str]] = []
    exclude_aabbs: list[tuple[np.ndarray, np.ndarray, str]] = []
    if selected_edit is not None:
        include_aabbs.append((
            np.asarray(selected_edit.target_aabb_min),
            np.asarray(selected_edit.target_aabb_max),
            f"{selected_edit.name}:include",
        ))
        for k, box in enumerate(selected_edit.include_aabbs):
            include_aabbs.append((np.asarray(box.min), np.asarray(box.max),
                                  f"{selected_edit.name}:include_extra_{k}"))
        for k, box in enumerate(selected_edit.exclude_aabbs):
            exclude_aabbs.append((np.asarray(box.min), np.asarray(box.max),
                                  f"{selected_edit.name}:exclude_{k}"))
    for i, s in enumerate(args.add_include):
        mn, mx = _parse_aabb_arg(s)
        include_aabbs.append((mn, mx, f"candidate_include_{i}"))
    for i, s in enumerate(args.add_exclude):
        mn, mx = _parse_aabb_arg(s)
        exclude_aabbs.append((mn, mx, f"candidate_exclude_{i}"))

    if not include_aabbs and not exclude_aabbs:
        raise SystemExit(
            "no AABBs to classify against; either the scene has no scene_edits or "
            "you need --add-include / --add-exclude"
        )

    # Compute neighborhood crop (AABB in MOCAP).
    if args.neighborhood is not None:
        nb_min, nb_max = _parse_aabb_arg(args.neighborhood)
    else:
        all_boxes = include_aabbs + exclude_aabbs
        nb_min = np.min([b[0] for b in all_boxes], axis=0) - 0.4
        nb_max = np.max([b[1] for b in all_boxes], axis=0) + 0.4

    print(f"[scene]   {args.scene}")
    print(f"[edit]    {selected_edit.name if selected_edit else '(none)'}")
    print(f"[mask]    {len(include_aabbs)} include, {len(exclude_aabbs)} exclude")
    print(f"[nb]      MOCAP min={nb_min.tolist()}  max={nb_max.tolist()}")

    # ---- Load gsplat means (heavy step) ------------------------------
    def _resolve(rel: str) -> Path:
        pp = Path(rel)
        return pp if pp.is_absolute() else (scene_dir / pp).resolve()

    gsplat_yml = _resolve(scene_cfg["gsplat_config_yml"])
    data_cwd = _resolve(scene_cfg["gsplat_data_cwd"]) if "gsplat_data_cwd" in scene_cfg else None

    print(f"[gsplat]  loading {gsplat_yml}")
    prev_cwd = os.getcwd()
    if data_cwd is not None:
        os.chdir(data_cwd)
    try:
        _, pipeline, _, _ = eval_setup(gsplat_yml, eval_num_rays_per_chunk=None, test_mode="test")
    finally:
        os.chdir(prev_cwd)

    means_ns = pipeline.model.means.detach().cpu().numpy().astype(np.float64)
    print(f"[gsplat]  {means_ns.shape[0]:,} Gaussians loaded")

    # NS → MOCAP via FrameGraph.
    T_ns_to_mocap = fg.transform("ns", "mocap")
    s = getattr(T_ns_to_mocap, "s", 1.0)
    R = T_ns_to_mocap.R
    t = T_ns_to_mocap.t
    means_mocap = (s * (R @ means_ns.T) + t[:, None]).T

    # Crop to neighborhood.
    in_nb = _mask_inside(means_mocap, nb_min, nb_max)
    print(f"[nb]      {in_nb.sum():,} of {means_mocap.shape[0]:,} Gaussians in neighborhood")
    cropped = means_mocap[in_nb]

    # Subsample.
    if cropped.shape[0] > args.max_points:
        rng = np.random.default_rng(args.seed)
        sub_idx = rng.choice(cropped.shape[0], size=args.max_points, replace=False)
        sub_idx.sort()
        cropped = cropped[sub_idx]
    print(f"[sub]     showing {cropped.shape[0]:,} points")

    # Classify.
    in_any_inc = np.zeros(cropped.shape[0], dtype=bool)
    for mn, mx, _ in include_aabbs:
        in_any_inc |= _mask_inside(cropped, mn, mx)
    in_any_exc = np.zeros(cropped.shape[0], dtype=bool)
    for mn, mx, _ in exclude_aabbs:
        in_any_exc |= _mask_inside(cropped, mn, mx)

    cyan_mask = in_any_inc & ~in_any_exc          # will move
    red_mask  = in_any_inc &  in_any_exc          # stranded
    orange_mask = ~in_any_inc & in_any_exc        # in exclude only
    gray_mask = ~in_any_inc & ~in_any_exc          # outside everything

    print(f"[class]   moved    = {int(cyan_mask.sum()):,}")
    print(f"[class]   stranded = {int(red_mask.sum()):,}  ← these stay at the OLD location")
    print(f"[class]   in-exclude only = {int(orange_mask.sum()):,}")
    print(f"[class]   outside everything = {int(gray_mask.sum()):,}")

    # ---- Build plotly figure -----------------------------------------
    traces = []

    # Wireframes.
    for mn, mx, name in include_aabbs:
        traces.append(_aabb_edges_trace(mn, mx, name, color="cyan", dash="solid"))
    for mn, mx, name in exclude_aabbs:
        traces.append(_aabb_edges_trace(mn, mx, name, color="orange", dash="dash"))
    traces.append(_aabb_edges_trace(nb_min, nb_max, "neighborhood", "white", dash="dot"))

    # Classified scatters.
    traces.append(_scatter(cropped[cyan_mask],   "moved (include ∧ ¬exclude)", "cyan",    visible=True))
    traces.append(_scatter(cropped[red_mask],    "stranded (include ∧ exclude)", "red",   visible=True))
    traces.append(_scatter(cropped[orange_mask], "in-exclude only",            "orange", visible=True))
    traces.append(_scatter(cropped[gray_mask],   "outside everything",         "lightgray",
                            visible=True if args.show_outside else "legendonly"))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"Gaussian mask authoring — {scene_cfg.get('scene_key', args.scene.stem)}",
        scene=dict(
            xaxis_title="x_mocap",
            yaxis_title="y_mocap",
            zaxis_title="z_mocap",
            aspectmode="data",
            bgcolor="black",
        ),
        paper_bgcolor="black",
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(itemsizing="constant"),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.out, include_plotlyjs="cdn", auto_open=False)
    print(f"\n[done] wrote {args.out}")
    print()
    print("Read the legend:")
    print("  cyan      = Gaussians that the move_gate edit WILL move")
    print("  red       = Gaussians inside include AND exclude — these get stranded")
    print("              (root cause of leftover gate parts in the rendered scene)")
    print("  orange    = inside exclude only (correctly NOT moved)")
    print("  gray      = outside everything (hidden by default; click in legend to show)")
    print()
    print("Iterate the AABBs (or pass --add-include / --add-exclude) and re-run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
