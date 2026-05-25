"""Single-3D-scene rollout comparison between two campaigns.

Like `plot_failures_with_recoveries.py`, but overlays every trial from
both campaigns into ONE 3-D plot so a pair of policies can be A/B
compared on the same trial cards without juggling subplot cameras.

Visual encoding:

  - line color        — `posthoc_outcome` (green=SUCCESS, red=COLLISION_GATE,
                        orange=MISS_GATE, purple=COLLISION_OTHER, gray=OOB)
  - line dash         — campaign (A=solid, B=dot)
  - cyan polylines    — recovery trajectories (open-circle marker at the
                        recovery seed = last-safe state)
  - scene objects     — gate + table PLY clouds, with the scene's
                        scene_edits applied so the gate is rendered at
                        its in-rollout position (mandatory for `center_*`
                        scenes whose gate is moved via rigid_transform_aabb)
  - gate AABB         — wireframe in scene's accent color
  - goal              — green diamond per scene_key

Usage:

    PYTHONPATH=src python scripts/eval/plot_campaign_compare.py \\
        --campaign-a runs/eval_campaigns/<A> \\
        --campaign-b runs/eval_campaigns/<B> \\
        --label-a "center_only" --label-b "center_and_real" \\
        --out runs/eval_campaigns/center_only_vs_real_center.html
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Rollout color per posthoc_outcome. Cyan reserved for recoveries.
_ROLLOUT_COLOR = {
    "SUCCESS":         "rgb( 46, 160,  67)",
    "MISS_GATE":       "rgb(231,  76,  60)",
    "COLLISION_GATE":  "rgb(178,  34,  34)",
    "COLLISION_OTHER": "rgb(150,  80, 180)",
    "OUT_OF_BOUNDS":   "rgb(120, 120, 120)",
}
_RECOVERY_COLOR = "rgb( 30, 180, 220)"

# Campaign → line dash (used in place of the per-scene dash from
# `plot_failures_with_recoveries.py`; here we collapse to one shared
# scene context, so dash is free to encode the A/B axis instead).
_CAMPAIGN_DASH = ["solid", "dot"]
_RECOVERY_CAMPAIGN_DASH = ["solid", "dash"]

_SCENE_AABB_COLOR = {
    "left_gate":              "rgba(80, 80, 80, 0.7)",
    "right_gate":             "rgba(120, 80, 30, 0.7)",
    "center_gate":            "rgba(20, 80, 120, 0.7)",
    "center_gate_from_left":  "rgba(20, 80, 120, 0.7)",
    "center_gate_from_right": "rgba(80, 20, 120, 0.7)",
}


def _gate_aabb_wireframe(aabb_min: np.ndarray, aabb_max: np.ndarray):
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


def _sphere_wireframe(center: np.ndarray, radius: float, n_lat: int = 9, n_lon: int = 12):
    """Latitude + longitude lines for a wireframe sphere at `center`.
    Cheap to draw and reads as a sphere in a 3-D plot."""
    xs, ys, zs = [], [], []
    # Latitude rings (constant z, full circle in xy).
    for k in range(1, n_lat):
        theta = np.pi * k / n_lat
        z = center[2] + radius * np.cos(theta)
        r = radius * np.sin(theta)
        phis = np.linspace(0, 2 * np.pi, 60)
        xs.extend((center[0] + r * np.cos(phis)).tolist() + [None])
        ys.extend((center[1] + r * np.sin(phis)).tolist() + [None])
        zs.extend([z] * len(phis) + [None])
    # Longitude half-rings.
    for k in range(n_lon):
        phi = 2 * np.pi * k / n_lon
        thetas = np.linspace(0, np.pi, 40)
        xs.extend((center[0] + radius * np.sin(thetas) * np.cos(phi)).tolist() + [None])
        ys.extend((center[1] + radius * np.sin(thetas) * np.sin(phi)).tolist() + [None])
        zs.extend((center[2] + radius * np.cos(thetas)).tolist() + [None])
    return xs, ys, zs


def _safety_goal_region_for_scene(scene_yaml: Path):
    """Best-effort lookup of the goal-tolerance region for a scene.

    Returns one of:
      ("box", half_extents: np.ndarray(3,))   — preferred when set
      ("sphere", radius: float)               — legacy fallback
      None                                    — neither field present
    """
    from falsify.io import load_yaml
    cfg = load_yaml(scene_yaml)
    scene_key = cfg.get("scene_key") or scene_yaml.stem
    candidate = REPO_ROOT / "configs" / "safety" / f"{scene_key}.yaml"
    if not candidate.is_file():
        return None
    safety = load_yaml(candidate) or {}
    miss = safety.get("miss_gate") or {}
    half_extents = miss.get("goal_tolerance_half_extents")
    if half_extents is not None:
        return ("box", np.asarray(half_extents, dtype=np.float64))
    v = miss.get("goal_tolerance_m")
    if v is not None:
        return ("sphere", float(v))
    return None


def _box_wireframe(center: np.ndarray, half_extents: np.ndarray):
    """12 edges of an axis-aligned rectangular prism centered at `center`."""
    lo = center - half_extents
    hi = center + half_extents
    return _gate_aabb_wireframe(lo, hi)


def _ned_to_mocap_via_scene(scene_yaml: Path, positions_ned: np.ndarray) -> np.ndarray:
    from falsify.geometry import Point
    from falsify.io import build_frame_graph, load_yaml
    scene_cfg = load_yaml(scene_yaml)
    fg = build_frame_graph(scene_cfg, base_path=scene_yaml.parent)
    ned_frame = fg.frame("ned")
    out = []
    for p in positions_ned:
        out.append(fg.convert(Point(np.asarray(p, dtype=np.float64), frame=ned_frame), to="mocap").xyz)
    return np.asarray(out, dtype=np.float64)


def _gather_trials(campaign_dir: Path):
    trials = []
    for path in sorted(campaign_dir.glob("*/trial_*/episode_summary.json")):
        s = json.loads(path.read_text())
        rollout_npz = path.parent / "rollout_states.npz"
        if not rollout_npz.is_file():
            continue
        recovery_npz = path.parent / "recovery_trajectory.npz"
        trials.append({
            "summary": s,
            "rollout_npz": rollout_npz,
            "recovery_npz": recovery_npz if recovery_npz.is_file() else None,
        })
    return trials


def _add_scene_context(fig, all_trials, *, max_cloud_points: int):
    """Draw gate AABB + scene_edit-applied scene-object clouds + goal,
    once per unique scene_yaml path across both campaigns."""
    from falsify.geometry import PointCloud
    from falsify.io import load_yaml, build_frame_graph
    from falsify.sim.scene_edits import apply_edits_to_scene_object, load_scene_edits
    from falsify.visualization import read_ply, subsample
    import plotly.graph_objects as go

    seen_scenes: dict[str, dict] = {}
    for t in all_trials:
        s = t["summary"]
        key = s["scene_key"]
        if key in seen_scenes:
            continue
        seen_scenes[key] = {
            "scene_path": REPO_ROOT / s["scene"],
            "scene_key": key,
        }

    for key, info in seen_scenes.items():
        scene_path = info["scene_path"]
        scene_cfg = load_yaml(scene_path)
        scene_dir = scene_path.parent
        fg = build_frame_graph(scene_cfg, base_path=scene_dir)
        edits = load_scene_edits(scene_cfg)

        # Gate AABB (mocap coords already track the moved gate per the YAML).
        region = scene_cfg.get("gate_region") or {}
        if region:
            aabb_min = np.asarray(region["aabb_min"], dtype=np.float64)
            aabb_max = np.asarray(region["aabb_max"], dtype=np.float64)
            xs, ys, zs = _gate_aabb_wireframe(aabb_min, aabb_max)
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs, mode="lines",
                line=dict(color=_SCENE_AABB_COLOR.get(key, "gray"), width=3),
                name=f"{key} gate AABB",
                legendgroup=f"context_{key}",
            ))

        # scene_objects PLY clouds — APPLY scene_edits so e.g. the
        # center_gate scene actually shows the gate in its center-anchored
        # position (the raw left_gate.ply is at the left position).
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
                hoverinfo="skip",
            ))

        # Goal + success-tolerance sphere. The runtime stops the rollout
        # (and posthoc rubber-stamps SUCCESS) as soon as the drone is
        # within `safety.miss_gate.goal_tolerance_m` of this point AND
        # has previously transited the gate AABB. Visualizing the
        # tolerance ball is the only way "why is this a SUCCESS?" makes
        # sense at a glance.
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
            ))

            # Pull goal-tolerance region out of the matching safety
            # YAML. Box config wins when present; sphere is the legacy
            # fallback. Per the campaign trial cards each trial points at
            # one safety YAML; we read the first trial's safety to
            # discover the tolerance for this scene.
            region = _safety_goal_region_for_scene(scene_path)
            if region is not None:
                kind, geom = region
                if kind == "box":
                    xs, ys, zs = _box_wireframe(g, geom)
                    label = (
                        f"{key} goal box "
                        f"(half_extents={[round(v,2) for v in geom.tolist()]})"
                    )
                else:
                    xs, ys, zs = _sphere_wireframe(g, geom)
                    label = f"{key} goal sphere (r={geom} m)"
                fig.add_trace(go.Scatter3d(
                    x=xs, y=ys, z=zs, mode="lines",
                    line=dict(color="rgb(10,90,30)", width=4, dash="dot"),
                    name=label,
                    legendgroup=f"context_{key}",
                    hoverinfo="skip",
                ))


def _add_trials(fig, trials, *, label, dash, recovery_dash, legend_seen):
    """Overlay rollout + (when present) recovery for one campaign."""
    import plotly.graph_objects as go

    for t in trials:
        s = t["summary"]
        scene_key = s["scene_key"]
        outcome = s.get("posthoc_outcome", "UNKNOWN")
        trial_idx = s["trial_index"]
        scene_path = REPO_ROOT / s["scene"]

        rollout_pos_ned = np.load(t["rollout_npz"], allow_pickle=True)["positions_ned"]
        rollout_mocap = _ned_to_mocap_via_scene(scene_path, rollout_pos_ned)

        group = f"{label}/{outcome}/{scene_key}/t{trial_idx:03d}"
        outcome_key = f"{label}/{outcome}"
        show_legend = outcome_key not in legend_seen
        legend_seen.add(outcome_key)

        fig.add_trace(go.Scatter3d(
            x=rollout_mocap[:, 0], y=rollout_mocap[:, 1], z=rollout_mocap[:, 2],
            mode="lines",
            line=dict(
                color=_ROLLOUT_COLOR.get(outcome, "rgb(120,120,120)"),
                width=3,
                dash=dash,
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
        ))

        # Start marker. Shares the line's `outcome_key` legend group so
        # toggling the legend entry hides the markers together with the
        # rollout polyline.
        fig.add_trace(go.Scatter3d(
            x=[rollout_mocap[0, 0]], y=[rollout_mocap[0, 1]], z=[rollout_mocap[0, 2]],
            mode="markers",
            marker=dict(size=4, color="rgb(20,20,20)", symbol="circle"),
            name=f"start {scene_key} t{trial_idx:03d}",
            legendgroup=outcome_key,
            showlegend=False,
        ))
        # End marker (same legend group as the line).
        fig.add_trace(go.Scatter3d(
            x=[rollout_mocap[-1, 0]], y=[rollout_mocap[-1, 1]], z=[rollout_mocap[-1, 2]],
            mode="markers",
            marker=dict(
                size=5,
                color=_ROLLOUT_COLOR.get(outcome, "rgb(120,120,120)"),
                symbol="x", line=dict(width=1.5, color="black"),
            ),
            name=f"end t{trial_idx:03d}",
            legendgroup=outcome_key,
            showlegend=False,
        ))

        # Recovery overlay (when present).
        if t["recovery_npz"] is not None:
            recovery_pos_ned = np.load(t["recovery_npz"], allow_pickle=True)["positions_ned"]
            recovery_mocap = _ned_to_mocap_via_scene(scene_path, recovery_pos_ned)
            recovery_key = f"{label}/recovery"
            show_legend_rec = recovery_key not in legend_seen
            legend_seen.add(recovery_key)
            fig.add_trace(go.Scatter3d(
                x=recovery_mocap[:, 0], y=recovery_mocap[:, 1], z=recovery_mocap[:, 2],
                mode="lines",
                line=dict(color=_RECOVERY_COLOR, width=4, dash=recovery_dash),
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
            ))
            fig.add_trace(go.Scatter3d(
                x=[recovery_mocap[0, 0]],
                y=[recovery_mocap[0, 1]],
                z=[recovery_mocap[0, 2]],
                mode="markers",
                marker=dict(size=6, color=_RECOVERY_COLOR, symbol="circle-open",
                            line=dict(width=2, color="black")),
                name=f"recv-seed t{trial_idx:03d}",
                legendgroup=recovery_key,
                showlegend=False,
            ))


def _summary(trials: list[dict]) -> str:
    by_outcome = defaultdict(int)
    for t in trials:
        by_outcome[t["summary"].get("posthoc_outcome", "UNKNOWN")] += 1
    return ", ".join(f"{v} {k}" for k, v in sorted(by_outcome.items()))


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--campaign-a", required=True, type=Path)
    ap.add_argument("--campaign-b", required=True, type=Path)
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-cloud-points", type=int, default=4000)
    args = ap.parse_args(argv)

    import plotly.graph_objects as go

    label_a = args.label_a or args.campaign_a.name
    label_b = args.label_b or args.campaign_b.name

    trials_a = _gather_trials(args.campaign_a)
    trials_b = _gather_trials(args.campaign_b)
    if not trials_a or not trials_b:
        raise SystemExit("Either campaign has zero loadable trials.")

    fig = go.Figure()

    # Shared scene context (gate + table with edits, AABB, goal).
    _add_scene_context(fig, trials_a + trials_b, max_cloud_points=args.max_cloud_points)

    legend_seen: set[str] = set()
    _add_trials(fig, trials_a, label=label_a,
                dash=_CAMPAIGN_DASH[0],
                recovery_dash=_RECOVERY_CAMPAIGN_DASH[0],
                legend_seen=legend_seen)
    _add_trials(fig, trials_b, label=label_b,
                dash=_CAMPAIGN_DASH[1],
                recovery_dash=_RECOVERY_CAMPAIGN_DASH[1],
                legend_seen=legend_seen)

    fig.update_layout(
        title=(
            f"Campaign A/B comparison — {label_a} vs {label_b}"
            f"<br><sub>{label_a}: {_summary(trials_a)} • "
            f"{label_b}: {_summary(trials_b)}"
            "<br>Color = posthoc_outcome (green=SUCCESS, red=COLLISION_GATE, "
            "orange=MISS_GATE). Dash = campaign "
            f"({label_a}=solid, {label_b}=dotted). Cyan = recovery "
            "trajectories (open circle = last-safe seed).</sub>"
        ),
        scene=dict(
            xaxis=dict(title="mocap x (m)"),
            yaxis=dict(title="mocap y (m)"),
            zaxis=dict(title="mocap z (m, up)"),
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.0)),
        ),
        height=800,
        margin=dict(l=0, r=0, t=110, b=0),
        legend=dict(itemsizing="constant"),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(args.out), include_plotlyjs="cdn")
    print(f"[plot] wrote {args.out}  ({args.out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
