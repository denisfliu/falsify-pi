"""N-up subplot grid of campaign rollouts, sharing one scene context.

Each campaign gets its own 3-D subplot, sharing the same axis ranges and
camera so trajectories are visually comparable. Same visual encoding as
`plot_campaign_compare.py`:

  - line color   — `posthoc_outcome`
  - cyan         — recovery trajectories (open-circle seed marker)
  - scene context per subplot — gate + table clouds (scene_edits applied),
    gate AABB wireframe, goal point, and the safety YAML's goal-tolerance
    region (box wireframe when `goal_tolerance_half_extents` is set,
    sphere wireframe otherwise)

Designed for parameter sweeps — e.g. action-horizon = {10, 25, 50}.

Usage:

    PYTHONPATH=src python scripts/plot_campaign_grid.py \\
        --campaign "chunk=10:runs/eval_campaigns/...chunk10..." \\
        --campaign "chunk=25:runs/eval_campaigns/...chunk25..." \\
        --campaign "chunk=50:runs/eval_campaigns/...chunk50..." \\
        --out runs/eval_campaigns/chunk_sweep.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
# Reuse all the heavy lifters from the A/B script.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import plot_campaign_compare as cc  # noqa: E402


def _parse_campaign(spec: str) -> tuple[str, Path]:
    """Parse `label:path` (or just `path`) into a (label, Path) pair."""
    if ":" in spec and not spec.startswith("/"):
        label, _, raw = spec.partition(":")
        return label, Path(raw)
    p = Path(spec)
    return p.name, p


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--campaign", required=True, action="append",
        help="`label:path` pair (or just `path`); repeat once per campaign.",
    )
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-cloud-points", type=int, default=4000)
    ap.add_argument(
        "--title", default=None,
        help="Override the top-level title; default lists campaign labels.",
    )
    args = ap.parse_args(argv)

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    campaigns: list[tuple[str, Path, list[dict]]] = []
    for spec in args.campaign:
        label, path = _parse_campaign(spec)
        if not path.is_dir():
            raise SystemExit(f"Campaign dir not found: {path}")
        trials = cc._gather_trials(path)
        if not trials:
            raise SystemExit(f"Zero loadable trials under {path}")
        campaigns.append((label, path, trials))

    n = len(campaigns)
    fig = make_subplots(
        rows=1, cols=n,
        specs=[[{"type": "scene"}] * n],
        subplot_titles=[
            f"{label} — {cc._summary(trials)}"
            for label, _, trials in campaigns
        ],
        horizontal_spacing=0.01,
    )

    # We render scene context once per subplot, against each campaign's
    # trial set independently — every panel ends up with the same gate +
    # table + goal box because they all share the same scene YAML.
    for col, (_, _, trials) in enumerate(campaigns, start=1):
        _add_scene_context_subplot(fig, trials, col=col,
                                   max_cloud_points=args.max_cloud_points)

    legend_seen: set[str] = set()
    for col, (label, _, trials) in enumerate(campaigns, start=1):
        _add_trials_subplot(fig, trials, col=col, label=label,
                            legend_seen=legend_seen)

    # Shared scene layout — bind every scene/sceneN dict in one shot.
    shared_scene = dict(
        xaxis=dict(title="mocap x (m)"),
        yaxis=dict(title="mocap y (m)"),
        zaxis=dict(title="mocap z (m, up)"),
        aspectmode="data",
        camera=dict(eye=dict(x=1.6, y=1.6, z=1.0)),
    )
    layout_kwargs = {"scene": shared_scene}
    for i in range(2, n + 1):
        layout_kwargs[f"scene{i}"] = shared_scene

    title = args.title or (
        "Campaign grid — "
        + " | ".join(label for label, _, _ in campaigns)
        + "<br><sub>Color = posthoc_outcome (green=SUCCESS, red=COLLISION_GATE, "
        "orange=MISS_GATE, purple=COLLISION_OTHER, gray=OOB). "
        "Cyan = recovery (open circle = last-safe seed). "
        "Dotted dark-green prism = safety goal box.</sub>"
    )
    fig.update_layout(
        title=title,
        height=780,
        margin=dict(l=0, r=0, t=110, b=0),
        legend=dict(itemsizing="constant"),
        **layout_kwargs,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(args.out), include_plotlyjs="cdn")
    print(f"[plot] wrote {args.out}  ({args.out.stat().st_size // 1024} KB)")
    return 0


# ---- subplot-aware variants of the cc helpers --------------------------
# cc._add_scene_context / _add_trials draw to a single-figure go.Figure
# via .add_trace; for subplots we need the same logic but with row/col
# kwargs. Easiest path: replicate the bodies, calling .add_trace(...,
# row=1, col=col).


def _add_scene_context_subplot(fig, trials, *, col: int, max_cloud_points: int):
    from collections import defaultdict
    import numpy as np
    import plotly.graph_objects as go
    from falsify.geometry import PointCloud
    from falsify.io import load_yaml, build_frame_graph
    from falsify.sim.scene_edits import apply_edits_to_scene_object, load_scene_edits
    from falsify.visualization import read_ply, subsample

    by_scene: dict[str, dict] = {}
    for t in trials:
        s = t["summary"]
        k = s["scene_key"]
        if k in by_scene:
            continue
        by_scene[k] = {"scene_path": REPO_ROOT / s["scene"], "scene_key": k}

    for key, info in by_scene.items():
        scene_path = info["scene_path"]
        scene_cfg = load_yaml(scene_path)
        scene_dir = scene_path.parent
        fg = build_frame_graph(scene_cfg, base_path=scene_dir)
        edits = load_scene_edits(scene_cfg)

        # Gate AABB.
        region = scene_cfg.get("gate_region") or {}
        if region:
            aabb_min = np.asarray(region["aabb_min"], dtype=np.float64)
            aabb_max = np.asarray(region["aabb_max"], dtype=np.float64)
            xs, ys, zs = cc._gate_aabb_wireframe(aabb_min, aabb_max)
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs, mode="lines",
                line=dict(color=cc._SCENE_AABB_COLOR.get(key, "gray"), width=3),
                name=f"{key} gate AABB",
                legendgroup=f"context_{key}",
                showlegend=(col == 1),
            ), row=1, col=col)

        # scene_objects — scene_edits applied so the gate sits at its
        # in-rollout position (mandatory for center_* scenes).
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
                    if new_pts.shape[0] % cloud.points.shape[0] != 0:
                        raise ValueError(
                            f"scene_edit on {entry['name']}: cloud grew from "
                            f"{cloud.points.shape[0]} → {new_pts.shape[0]} points, "
                            f"not an integer multiple"
                        )
                    reps = new_pts.shape[0] // cloud.points.shape[0]
                    new_colors = np.tile(cloud.colors, (reps, 1))
                cloud = PointCloud(points=new_pts, frame=cloud.frame, colors=new_colors)
            pts = np.asarray(cloud.points, dtype=np.float64)
            colour = entry.get("color", (0.5, 0.5, 0.5))
            rgb = f"rgb({int(255*colour[0])},{int(255*colour[1])},{int(255*colour[2])})"
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers",
                marker=dict(size=1.2, color=rgb, opacity=0.55),
                name=f"{key}:{entry['name']}",
                legendgroup=f"context_{key}",
                showlegend=(col == 1),
                hoverinfo="skip",
            ), row=1, col=col)

        # Goal + tolerance region.
        goal = scene_cfg.get("goal_position_mocap")
        if goal:
            g = np.asarray(goal, dtype=np.float64)
            fig.add_trace(go.Scatter3d(
                x=[g[0]], y=[g[1]], z=[g[2]],
                mode="markers+text",
                marker=dict(size=6, color="rgba(50,200,50,0.95)", symbol="diamond"),
                text=[f"goal:{key}"], textposition="top center",
                name=f"{key} goal",
                legendgroup=f"context_{key}",
                showlegend=(col == 1),
            ), row=1, col=col)

            region = cc._safety_goal_region_for_scene(scene_path)
            if region is not None:
                kind, geom = region
                if kind == "box":
                    xs, ys, zs = cc._box_wireframe(g, geom)
                    label = (
                        f"{key} goal box "
                        f"(half_extents={[round(v,2) for v in geom.tolist()]})"
                    )
                else:
                    xs, ys, zs = cc._sphere_wireframe(g, geom)
                    label = f"{key} goal sphere (r={geom} m)"
                fig.add_trace(go.Scatter3d(
                    x=xs, y=ys, z=zs, mode="lines",
                    line=dict(color="rgb(10,90,30)", width=4, dash="dot"),
                    name=label,
                    legendgroup=f"context_{key}",
                    showlegend=(col == 1),
                    hoverinfo="skip",
                ), row=1, col=col)


def _add_trials_subplot(fig, trials, *, col: int, label: str, legend_seen):
    import numpy as np
    import plotly.graph_objects as go

    for t in trials:
        s = t["summary"]
        scene_key = s["scene_key"]
        outcome = s.get("posthoc_outcome", "UNKNOWN")
        trial_idx = s["trial_index"]
        scene_path = REPO_ROOT / s["scene"]

        rollout_pos_ned = np.load(t["rollout_npz"], allow_pickle=True)["positions_ned"]
        rollout_mocap = cc._ned_to_mocap_via_scene(scene_path, rollout_pos_ned)

        group = f"{label}/{outcome}/{scene_key}/t{trial_idx:03d}"
        outcome_key = f"{label}/{outcome}"
        show_legend = outcome_key not in legend_seen
        legend_seen.add(outcome_key)

        fig.add_trace(go.Scatter3d(
            x=rollout_mocap[:, 0], y=rollout_mocap[:, 1], z=rollout_mocap[:, 2],
            mode="lines",
            line=dict(
                color=cc._ROLLOUT_COLOR.get(outcome, "rgb(120,120,120)"),
                width=3, dash="solid",
            ),
            name=f"{label}:{outcome}",
            legendgroup=outcome_key,
            showlegend=show_legend,
            hovertemplate=(
                f"<b>{label}</b><br>"
                f"{outcome} • {scene_key} • trial {trial_idx}<br>"
                "step=%{pointNumber}<br>"
                "mocap=(%{x:.2f}, %{y:.2f}, %{z:.2f})"
                "<extra></extra>"
            ),
        ), row=1, col=col)

        # Start + end markers — share the line's outcome_key legend group
        # so toggling the legend entry hides them together with the line.
        fig.add_trace(go.Scatter3d(
            x=[rollout_mocap[0, 0]], y=[rollout_mocap[0, 1]], z=[rollout_mocap[0, 2]],
            mode="markers",
            marker=dict(size=4, color="rgb(20,20,20)", symbol="circle"),
            name=f"start {scene_key} t{trial_idx:03d}",
            legendgroup=outcome_key, showlegend=False,
        ), row=1, col=col)
        fig.add_trace(go.Scatter3d(
            x=[rollout_mocap[-1, 0]], y=[rollout_mocap[-1, 1]], z=[rollout_mocap[-1, 2]],
            mode="markers",
            marker=dict(
                size=5,
                color=cc._ROLLOUT_COLOR.get(outcome, "rgb(120,120,120)"),
                symbol="x", line=dict(width=1.5, color="black"),
            ),
            name=f"end t{trial_idx:03d}",
            legendgroup=outcome_key, showlegend=False,
        ), row=1, col=col)

        # Recovery overlay.
        if t["recovery_npz"] is not None:
            recovery_pos_ned = np.load(t["recovery_npz"], allow_pickle=True)["positions_ned"]
            recovery_mocap = cc._ned_to_mocap_via_scene(scene_path, recovery_pos_ned)
            recovery_key = f"{label}/recovery"
            show_legend_rec = recovery_key not in legend_seen
            legend_seen.add(recovery_key)
            fig.add_trace(go.Scatter3d(
                x=recovery_mocap[:, 0], y=recovery_mocap[:, 1], z=recovery_mocap[:, 2],
                mode="lines",
                line=dict(color=cc._RECOVERY_COLOR, width=4, dash="solid"),
                name=f"{label}:recovery",
                legendgroup=recovery_key,
                showlegend=show_legend_rec,
                hovertemplate=(
                    f"<b>{label} recovery</b><br>"
                    f"{scene_key} • trial {trial_idx}<br>"
                    "step=%{pointNumber}<br>"
                    "mocap=(%{x:.2f}, %{y:.2f}, %{z:.2f})"
                    "<extra></extra>"
                ),
            ), row=1, col=col)
            fig.add_trace(go.Scatter3d(
                x=[recovery_mocap[0, 0]],
                y=[recovery_mocap[0, 1]],
                z=[recovery_mocap[0, 2]],
                mode="markers",
                marker=dict(size=6, color=cc._RECOVERY_COLOR, symbol="circle-open",
                            line=dict(width=2, color="black")),
                name=f"recv-seed t{trial_idx:03d}",
                legendgroup=recovery_key, showlegend=False,
            ), row=1, col=col)


if __name__ == "__main__":
    raise SystemExit(main())
