"""Plot MISS_GATE + SKIPPED_GATE trajectories from an eval campaign.

Discovers every per-trial `episode_summary.json` under the campaign dir
whose `posthoc_outcome` is MISS_GATE or SKIPPED_GATE, loads its
`rollout_states.npz` (NED positions), converts to MOCAP via the scene's
FrameGraph, and dumps a single self-contained plotly HTML with:

  - one trace per trial trajectory, color-coded by outcome (red = MISS,
    orange = SKIP), dashed by scene_key.
  - the gate AABB per (scene, scene_key) as a wireframe box.
  - subsampled scene_objects point clouds (gate + table) per scene_key.
  - start markers + goal markers per scene_key.

MOCAP is the joint frame for all gate scenes, so the plot is one shared
coordinate system even though trials come from different scene files.

Usage:

    PYTHONPATH=src python scripts/plot_miss_skip_trajectories.py \\
        --campaign runs/eval_campaigns/pi07_nonhistory_pure_no_rtc_posthoc_20260519_124051 \\
        --out runs/eval_campaigns/pi07_nonhistory_pure_no_rtc_posthoc_20260519_124051/miss_skip.html
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent


# Outcome → colour. Covers every category emitted by safety/posthoc.py.
# Picks distinct hues so multi-outcome overlays remain readable.
_OUTCOME_COLOR = {
    "SUCCESS":            "rgb( 46, 184, 100)",   # green
    "MISS_GATE":          "rgb(231,  76,  60)",   # red
    "SKIPPED_GATE":       "rgb(241, 168,  16)",   # orange  (back-compat alias)
    "GOAL_NOT_REACHED":   "rgb( 80, 110, 220)",   # blue
    "COLLISION_GATE":     "rgb(178,  34,  34)",   # firebrick
    "COLLISION_OTHER":    "rgb(150,  80, 180)",   # purple
    "OUT_OF_BOUNDS":      "rgb(120, 120, 120)",   # grey
    "EXCESSIVE_VELOCITY": "rgb(180,  80,  20)",
    "EXCESSIVE_TILT":     "rgb(180,  80,  20)",
    "ERROR":              "rgb( 50,  50,  50)",
}
# Scene-key → line dash. Visual cue for which gate the trajectory targeted.
_SCENE_DASH = {
    "left_gate":              "solid",
    "right_gate":             "dot",
    "center_gate_from_left":  "dash",
    "center_gate_from_right": "dashdot",
}
# Scene-key → AABB-wireframe colour (subdued).
_SCENE_AABB_COLOR = {
    "left_gate":              "rgba(80, 80, 80, 0.7)",
    "right_gate":             "rgba(120, 80, 30, 0.7)",
    "center_gate_from_left":  "rgba(20, 80, 120, 0.7)",
    "center_gate_from_right": "rgba(80, 20, 120, 0.7)",
}


def _gate_aabb_wireframe(aabb_min: np.ndarray, aabb_max: np.ndarray):
    """12-edge wireframe of an axis-aligned box, returned as a flat list of
    (x, y, z) coordinates suitable for one plotly Scatter3d trace with
    None breaks between edges."""
    p = np.array([
        [aabb_min[0], aabb_min[1], aabb_min[2]],   # 0
        [aabb_max[0], aabb_min[1], aabb_min[2]],   # 1
        [aabb_max[0], aabb_max[1], aabb_min[2]],   # 2
        [aabb_min[0], aabb_max[1], aabb_min[2]],   # 3
        [aabb_min[0], aabb_min[1], aabb_max[2]],   # 4
        [aabb_max[0], aabb_min[1], aabb_max[2]],   # 5
        [aabb_max[0], aabb_max[1], aabb_max[2]],   # 6
        [aabb_min[0], aabb_max[1], aabb_max[2]],   # 7
    ])
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),    # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),    # top face
        (0, 4), (1, 5), (2, 6), (3, 7),    # verticals
    ]
    xs, ys, zs = [], [], []
    for a, b in edges:
        xs.extend([p[a, 0], p[b, 0], None])
        ys.extend([p[a, 1], p[b, 1], None])
        zs.extend([p[a, 2], p[b, 2], None])
    return xs, ys, zs


def _ned_to_mocap_via_scene(scene_yaml: Path, positions_ned: np.ndarray) -> np.ndarray:
    """Convert (N, 3) NED positions to MOCAP via the scene's FrameGraph.

    Lazy import inside so the script imports cleanly without falsify deps."""
    from falsify.geometry import Point
    from falsify.io import build_frame_graph, load_yaml
    scene_cfg = load_yaml(scene_yaml)
    fg = build_frame_graph(scene_cfg, base_path=scene_yaml.parent)
    mocap_pts = []
    ned_frame = fg.frame("ned")
    for p in positions_ned:
        ned_pt = Point(np.asarray(p, dtype=np.float64), frame=ned_frame)
        mocap_pts.append(fg.convert(ned_pt, to="mocap").xyz)
    return np.asarray(mocap_pts, dtype=np.float64)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--campaign", required=True, type=Path,
                    help="Campaign dir (the parent of <scene_key>/trial_NNN/).")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output HTML path.")
    ap.add_argument("--outcomes", nargs="+",
                    default=["MISS_GATE", "SKIPPED_GATE"],
                    help="Which posthoc_outcomes to include (default: "
                         "MISS_GATE + SKIPPED_GATE).")
    ap.add_argument("--max-cloud-points", type=int, default=4000,
                    help="Per scene_object PLY subsample target.")
    args = ap.parse_args(argv)

    import plotly.graph_objects as go
    from falsify.io import load_yaml
    from falsify.visualization import read_ply, subsample
    from falsify.geometry import PointCloud  # noqa: F401  (used downstream)

    # ---- discover trials -------------------------------------------------
    trials: list[dict] = []
    for path in sorted(args.campaign.glob("*/trial_*/episode_summary.json")):
        s = json.loads(path.read_text())
        if s.get("posthoc_outcome") not in args.outcomes:
            continue
        npz = path.parent / "rollout_states.npz"
        if not npz.is_file():
            print(f"[skip] {path.parent.name}: no rollout_states.npz")
            continue
        trials.append({
            "summary_path": path,
            "summary": s,
            "npz": npz,
        })

    if not trials:
        raise SystemExit("No matching trials found under "
                         f"{args.campaign} for outcomes {args.outcomes}.")

    print(f"[plot] {len(trials)} trials selected")

    # Group by scene so we load the scene context (PLYs, AABB, goal) once.
    by_scene_key: dict[str, list[dict]] = defaultdict(list)
    for t in trials:
        by_scene_key[t["summary"]["scene_key"]].append(t)

    fig = go.Figure()

    # ---- scene context (clouds, AABB, start, goal) per scene_key ---------
    for scene_key, ts in by_scene_key.items():
        scene_path = REPO_ROOT / ts[0]["summary"]["scene"]
        scene_cfg = load_yaml(scene_path)
        scene_dir = scene_path.parent

        # Gate region AABB (post-perturbation if any — we just take the
        # nominal block since these trials are start-jitter-only).
        region = scene_cfg.get("gate_region") or {}
        if region:
            aabb_min = np.asarray(region["aabb_min"], dtype=np.float64)
            aabb_max = np.asarray(region["aabb_max"], dtype=np.float64)
            xs, ys, zs = _gate_aabb_wireframe(aabb_min, aabb_max)
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                line=dict(color=_SCENE_AABB_COLOR.get(scene_key, "gray"), width=3),
                name=f"{scene_key} gate AABB",
                legendgroup=f"context_{scene_key}",
            ))

        # Scene objects (gate + table) — subsampled, in MOCAP (their authored frame).
        for entry in scene_cfg.get("scene_objects") or []:
            ply_path = scene_dir / entry["ply"]
            if not ply_path.is_file():
                continue
            from falsify.io import build_frame_graph
            fg = build_frame_graph(scene_cfg, base_path=scene_dir)
            cloud = read_ply(ply_path, fg.frame(entry["frame"]))
            cloud = subsample(cloud, args.max_cloud_points)
            pts = np.asarray(cloud.points, dtype=np.float64)
            colour = entry.get("color", (0.5, 0.5, 0.5))
            rgb = f"rgb({int(255*colour[0])},{int(255*colour[1])},{int(255*colour[2])})"
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers",
                marker=dict(size=1.2, color=rgb, opacity=0.45),
                name=f"{scene_key}:{entry['name']}",
                legendgroup=f"context_{scene_key}",
                showlegend=True,
            ))

        # Goal marker.
        goal = scene_cfg.get("goal_position_mocap")
        if goal:
            g = np.asarray(goal, dtype=np.float64)
            fig.add_trace(go.Scatter3d(
                x=[g[0]], y=[g[1]], z=[g[2]],
                mode="markers+text",
                marker=dict(size=6, color="rgba(50,200,50,0.95)", symbol="diamond"),
                text=[f"goal:{scene_key}"],
                textposition="top center",
                name=f"{scene_key} goal",
                legendgroup=f"context_{scene_key}",
            ))

    # ---- trajectories ---------------------------------------------------
    for t in trials:
        s = t["summary"]
        scene_key = s["scene_key"]
        outcome = s["posthoc_outcome"]
        trial_idx = s["trial_index"]
        scene_path = REPO_ROOT / s["scene"]

        npz = np.load(t["npz"], allow_pickle=True)
        positions_ned = npz["positions_ned"]
        positions_mocap = _ned_to_mocap_via_scene(scene_path, positions_ned)

        fig.add_trace(go.Scatter3d(
            x=positions_mocap[:, 0],
            y=positions_mocap[:, 1],
            z=positions_mocap[:, 2],
            mode="lines+markers",
            line=dict(
                color=_OUTCOME_COLOR.get(outcome, "gray"),
                width=4,
                dash=_SCENE_DASH.get(scene_key, "solid"),
            ),
            marker=dict(size=2, color=_OUTCOME_COLOR.get(outcome, "gray")),
            name=f"{outcome[:4]} {scene_key} t{trial_idx:03d}",
            legendgroup=outcome,
            hovertemplate=(
                f"<b>{outcome}</b><br>"
                f"scene={scene_key}<br>"
                f"trial={trial_idx}<br>"
                "step=%{pointNumber}<br>"
                "mocap=(%{x:.2f}, %{y:.2f}, %{z:.2f})"
                "<extra></extra>"
            ),
        ))
        # Start + end dots (small) for visual anchoring.
        fig.add_trace(go.Scatter3d(
            x=[positions_mocap[0, 0]],
            y=[positions_mocap[0, 1]],
            z=[positions_mocap[0, 2]],
            mode="markers",
            marker=dict(size=5, color="rgb(20,20,20)", symbol="circle"),
            name=f"start {scene_key} t{trial_idx:03d}",
            legendgroup=outcome,
            showlegend=False,
        ))
        fig.add_trace(go.Scatter3d(
            x=[positions_mocap[-1, 0]],
            y=[positions_mocap[-1, 1]],
            z=[positions_mocap[-1, 2]],
            mode="markers",
            marker=dict(size=5, color=_OUTCOME_COLOR.get(outcome, "gray"),
                        symbol="x", line=dict(width=1.5, color="black")),
            name=f"end {scene_key} t{trial_idx:03d}",
            legendgroup=outcome,
            showlegend=False,
        ))

    # ---- layout ---------------------------------------------------------
    # Dynamic title — list each outcome with its count.
    by_outcome: dict[str, int] = {}
    for t in trials:
        o = t["summary"]["posthoc_outcome"]
        by_outcome[o] = by_outcome.get(o, 0) + 1
    title_parts = " + ".join(f"{o} ({n})" for o, n in sorted(by_outcome.items()))
    fig.update_layout(
        title=(
            f"{title_parts} — {args.campaign.name}<br>"
            "<sub>Colour = post-hoc outcome (see safety/posthoc.py). "
            "Line dash: solid=left, dot=right, dash=center_from_left, dashdot=center_from_right. "
            "Black • = start, × = end.</sub>"
        ),
        scene=dict(
            xaxis_title="mocap x (m)",
            yaxis_title="mocap y (m)",
            zaxis_title="mocap z (m)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=80),
        legend=dict(itemsizing="constant"),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(args.out), include_plotlyjs="cdn")
    print(f"[plot] wrote {args.out}  (size {args.out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
