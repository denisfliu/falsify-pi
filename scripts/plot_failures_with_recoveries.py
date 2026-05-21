"""Overlay failed rollouts and their recovery trajectories.

For every failed trial in a campaign (post-hoc outcome in
``{MISS_GATE, COLLISION_GATE, COLLISION_OTHER, OUT_OF_BOUNDS}``) that
also wrote a ``recovery_trajectory.npz``, plot:

  - the rollout trajectory (NED → MOCAP) in the outcome's color
  - the recovery trajectory in cyan, paired by trial via legend group

Plus the per-scene context once (gate AABB wireframe, gate + table
clouds, goal marker) so the geometry is visible.

Usage:

    PYTHONPATH=src python scripts/plot_failures_with_recoveries.py \\
        --campaign runs/eval_campaigns/<campaign> \\
        --out runs/eval_campaigns/<campaign>/failures_recoveries.html
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent

# Failure outcomes that produce recovery NPZs (matches recovery YAML triggers).
_FAILED_OUTCOMES = frozenset({
    "MISS_GATE", "COLLISION_GATE", "COLLISION_OTHER", "OUT_OF_BOUNDS",
})

# Rollout colours per outcome.
_ROLLOUT_COLOR = {
    "SUCCESS":         "rgb( 46, 160,  67)",   # green
    "MISS_GATE":       "rgb(231,  76,  60)",   # red
    "COLLISION_GATE":  "rgb(178,  34,  34)",   # firebrick
    "COLLISION_OTHER": "rgb(150,  80, 180)",   # purple
    "OUT_OF_BOUNDS":   "rgb(120, 120, 120)",   # grey
    "GOAL_NOT_REACHED":"rgb(255, 165,   0)",   # orange
}
# Recovery trajectories share one colour — readability over per-outcome
# differentiation, since they all represent the same planner output.
_RECOVERY_COLOR = "rgb( 30, 180, 220)"   # cyan

_SCENE_DASH = {
    "left_gate":              "solid",
    "right_gate":             "dot",
    "center_gate_from_left":  "dash",
    "center_gate_from_right": "dashdot",
}
_SCENE_AABB_COLOR = {
    "left_gate":              "rgba(80, 80, 80, 0.7)",
    "right_gate":             "rgba(120, 80, 30, 0.7)",
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


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--campaign", required=True, type=Path,
                    help="Campaign dir (parent of <scene_key>/trial_NNN/).")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output HTML path.")
    ap.add_argument("--outcomes", nargs="+",
                    default=sorted(_FAILED_OUTCOMES),
                    help="Failure outcomes to include (default: all four).")
    ap.add_argument("--include-successes", action="store_true",
                    help="Also overlay SUCCESS trials in green. Has no "
                         "recovery NPZ to draw (none was triggered), so the "
                         "rollout polyline is shown alone.")
    ap.add_argument("--max-cloud-points", type=int, default=4000)
    args = ap.parse_args(argv)

    import plotly.graph_objects as go
    from falsify.io import load_yaml, build_frame_graph
    from falsify.visualization import read_ply, subsample

    # When --include-successes is on, treat SUCCESS as a first-class
    # outcome alongside the failure set. The success branch is allowed
    # to skip the recovery_npz existence check (no recovery is triggered
    # for a SUCCESS rollout, so the NPZ won't be there).
    success_outcomes = {"SUCCESS"} if args.include_successes else set()
    accepted_outcomes = set(args.outcomes) | success_outcomes

    # ---- discover trials to plot ---------------------------------------
    trials: list[dict] = []
    n_failed_no_recovery = 0
    n_success = 0
    for path in sorted(args.campaign.glob("*/trial_*/episode_summary.json")):
        s = json.loads(path.read_text())
        outcome = s.get("posthoc_outcome")
        if outcome not in accepted_outcomes:
            continue
        rollout_npz = path.parent / "rollout_states.npz"
        recovery_npz = path.parent / "recovery_trajectory.npz"
        if not rollout_npz.is_file():
            continue
        if outcome == "SUCCESS":
            # SUCCESS trials don't fire a recovery; just plot the rollout.
            trials.append({"summary": s,
                           "rollout_npz": rollout_npz,
                           "recovery_npz": None})
            n_success += 1
            continue
        if not recovery_npz.is_file():
            n_failed_no_recovery += 1
            continue
        trials.append({"summary": s,
                       "rollout_npz": rollout_npz,
                       "recovery_npz": recovery_npz})

    if not trials:
        raise SystemExit(
            f"No matching trials under {args.campaign} "
            f"(outcomes={sorted(accepted_outcomes)})."
        )
    print(f"[plot] {len(trials) - n_success} failed trials with recoveries  "
          f"({n_failed_no_recovery} failed trials had no recovery NPZ); "
          f"{n_success} SUCCESS trials")

    by_scene_key: dict[str, list[dict]] = defaultdict(list)
    for t in trials:
        by_scene_key[t["summary"]["scene_key"]].append(t)

    fig = go.Figure()

    # ---- scene context (clouds, AABB, goal) per scene_key ----------------
    for scene_key, ts in by_scene_key.items():
        scene_path = REPO_ROOT / ts[0]["summary"]["scene"]
        scene_cfg = load_yaml(scene_path)
        scene_dir = scene_path.parent
        fg = build_frame_graph(scene_cfg, base_path=scene_dir)

        # Gate AABB.
        region = scene_cfg.get("gate_region") or {}
        if region:
            aabb_min = np.asarray(region["aabb_min"], dtype=np.float64)
            aabb_max = np.asarray(region["aabb_max"], dtype=np.float64)
            xs, ys, zs = _gate_aabb_wireframe(aabb_min, aabb_max)
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs, mode="lines",
                line=dict(color=_SCENE_AABB_COLOR.get(scene_key, "gray"), width=3),
                name=f"{scene_key} gate AABB",
                legendgroup=f"context_{scene_key}",
            ))

        # scene_objects clouds (subsampled).
        for entry in scene_cfg.get("scene_objects") or []:
            ply_path = scene_dir / entry["ply"]
            if not ply_path.is_file():
                continue
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
            ))

        # Goal.
        goal = scene_cfg.get("goal_position_mocap")
        if goal:
            g = np.asarray(goal, dtype=np.float64)
            fig.add_trace(go.Scatter3d(
                x=[g[0]], y=[g[1]], z=[g[2]],
                mode="markers+text",
                marker=dict(size=6, color="rgba(50,200,50,0.95)", symbol="diamond"),
                text=[f"goal:{scene_key}"], textposition="top center",
                name=f"{scene_key} goal",
                legendgroup=f"context_{scene_key}",
            ))

    # ---- rollouts + recoveries ------------------------------------------
    for t in trials:
        s = t["summary"]
        scene_key = s["scene_key"]
        outcome = s["posthoc_outcome"]
        trial_idx = s["trial_index"]
        scene_path = REPO_ROOT / s["scene"]

        rollout_pos_ned = np.load(t["rollout_npz"], allow_pickle=True)["positions_ned"]
        rollout_mocap = _ned_to_mocap_via_scene(scene_path, rollout_pos_ned)

        # SUCCESS trials carry no recovery NPZ — skip the recovery overlay
        # entirely for those. For everything else, load + transform once.
        recovery_mocap = None
        if t["recovery_npz"] is not None:
            recovery_pos_ned = np.load(
                t["recovery_npz"], allow_pickle=True,
            )["positions_ned"]
            recovery_mocap = _ned_to_mocap_via_scene(scene_path, recovery_pos_ned)

        group = f"{outcome}_{scene_key}_t{trial_idx:03d}"

        # Rollout: outcome-coloured solid (per scene dash).
        fig.add_trace(go.Scatter3d(
            x=rollout_mocap[:, 0], y=rollout_mocap[:, 1], z=rollout_mocap[:, 2],
            mode="lines",
            line=dict(
                color=_ROLLOUT_COLOR.get(outcome, "gray"),
                width=3,
                dash=_SCENE_DASH.get(scene_key, "solid"),
            ),
            name=f"ROLL {outcome[:4]} {scene_key} t{trial_idx:03d}",
            legendgroup=group,
            hovertemplate=(
                f"<b>rollout {outcome}</b><br>"
                f"scene={scene_key}<br>"
                f"trial={trial_idx}<br>"
                "step=%{pointNumber}<br>"
                "mocap=(%{x:.2f}, %{y:.2f}, %{z:.2f})"
                "<extra></extra>"
            ),
        ))
        # Rollout start/end markers.
        fig.add_trace(go.Scatter3d(
            x=[rollout_mocap[0, 0]], y=[rollout_mocap[0, 1]], z=[rollout_mocap[0, 2]],
            mode="markers",
            marker=dict(size=5, color="rgb(20,20,20)", symbol="circle"),
            name=f"start t{trial_idx:03d}",
            legendgroup=group,
            showlegend=False,
        ))
        fig.add_trace(go.Scatter3d(
            x=[rollout_mocap[-1, 0]], y=[rollout_mocap[-1, 1]], z=[rollout_mocap[-1, 2]],
            mode="markers",
            marker=dict(size=5, color=_ROLLOUT_COLOR.get(outcome, "gray"),
                        symbol="x", line=dict(width=1.5, color="black")),
            name=f"end t{trial_idx:03d}",
            legendgroup=group,
            showlegend=False,
        ))
        # Recovery: cyan dashed (per scene dash too). SUCCESS trials have
        # no recovery — skip the recovery overlay for those.
        if recovery_mocap is not None:
            fig.add_trace(go.Scatter3d(
                x=recovery_mocap[:, 0], y=recovery_mocap[:, 1], z=recovery_mocap[:, 2],
                mode="lines",
                line=dict(
                    color=_RECOVERY_COLOR,
                    width=4,
                    dash=_SCENE_DASH.get(scene_key, "solid"),
                ),
                name=f"RECV {outcome[:4]} {scene_key} t{trial_idx:03d}",
                legendgroup=group,
                hovertemplate=(
                    f"<b>recovery</b><br>"
                    f"scene={scene_key}<br>"
                    f"trial={trial_idx}<br>"
                    "step=%{pointNumber}<br>"
                    "mocap=(%{x:.2f}, %{y:.2f}, %{z:.2f})"
                    "<extra></extra>"
                ),
            ))
            # Recovery seed marker (start of recovery — last-safe state).
            fig.add_trace(go.Scatter3d(
                x=[recovery_mocap[0, 0]], y=[recovery_mocap[0, 1]], z=[recovery_mocap[0, 2]],
                mode="markers",
                marker=dict(size=6, color=_RECOVERY_COLOR, symbol="circle-open",
                            line=dict(width=2, color="black")),
                name=f"recv-seed t{trial_idx:03d}",
                legendgroup=group,
                showlegend=False,
            ))

    # ---- layout ----------------------------------------------------------
    by_outcome = defaultdict(int)
    for t in trials:
        by_outcome[t["summary"]["posthoc_outcome"]] += 1
    counts = " + ".join(f"{k} ({v})" for k, v in sorted(by_outcome.items()))

    fig.update_layout(
        title=(
            f"Failed rollouts + planned recoveries — {args.campaign.name}<br>"
            f"<sub>{counts} • rollout in outcome colour, recovery in cyan, "
            "open-circle marks the recovery seed (last-safe state). "
            "Line dash: solid=left, dot=right, dash=center_from_left, dashdot=center_from_right.</sub>"
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
    print(f"[plot] wrote {args.out}  ({args.out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
