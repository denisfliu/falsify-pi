"""Per-campaign HTML reports — trajectories overlay + outcome stacked bars.

Two public entry points consume a campaign dir (parent of
``<scene_key>/trial_NNN/episode_summary.json``) and write into
``<campaign_dir>/viz/``:

  - ``emit_trajectories_html`` — single 3-D plot with one
    scene-edits-applied context per scene_key and per-trial rollout +
    recovery overlays, legend-grouped so the viewer can isolate by
    scene × outcome.
  - ``emit_outcome_charts_html`` — per-scene stacked bars matching the
    visual contract of ``runs/eval_campaigns/summary_charts_20260519.html``.

Both are pure consumers of the on-disk campaign artifacts — no rerun,
no GPU. The thin CLI ``scripts/plot_eval_run.py`` wraps them so old
campaigns can be backfilled with viz.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]


# Outcome → color. Shared across both reports.
ROLLOUT_COLOR = {
    "SUCCESS":          "rgb( 46, 160,  67)",
    "MISS_GATE":        "rgb(231,  76,  60)",
    "COLLISION_GATE":   "rgb(178,  34,  34)",
    "COLLISION_OTHER":  "rgb(150,  80, 180)",
    "OUT_OF_BOUNDS":    "rgb(120, 120, 120)",
    "GOAL_NOT_REACHED": "rgb(255, 165,   0)",
    "ERROR":            "rgb( 80,  80,  80)",
}
RECOVERY_COLOR = "rgb( 30, 180, 220)"

# Order outcomes appear in stacked bars / legend. SUCCESS first (green
# bottom of stack), worst-failure-types after.
OUTCOME_ORDER = [
    "SUCCESS", "MISS_GATE", "GOAL_NOT_REACHED",
    "COLLISION_GATE", "COLLISION_OTHER", "OUT_OF_BOUNDS", "ERROR",
]


# ----------------------------- helpers -----------------------------------

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
    xs, ys, zs = [], [], []
    for k in range(1, n_lat):
        theta = np.pi * k / n_lat
        z = center[2] + radius * np.cos(theta)
        r = radius * np.sin(theta)
        phis = np.linspace(0, 2 * np.pi, 60)
        xs.extend((center[0] + r * np.cos(phis)).tolist() + [None])
        ys.extend((center[1] + r * np.sin(phis)).tolist() + [None])
        zs.extend([z] * len(phis) + [None])
    for k in range(n_lon):
        phi = 2 * np.pi * k / n_lon
        thetas = np.linspace(0, np.pi, 40)
        xs.extend((center[0] + radius * np.sin(thetas) * np.cos(phi)).tolist() + [None])
        ys.extend((center[1] + radius * np.sin(thetas) * np.sin(phi)).tolist() + [None])
        zs.extend((center[2] + radius * np.cos(thetas)).tolist() + [None])
    return xs, ys, zs


def _box_wireframe(center: np.ndarray, half_extents: np.ndarray):
    return _gate_aabb_wireframe(center - half_extents, center + half_extents)


def _safety_goal_region(scene_yaml: Path):
    """Return ('box', half_extents) | ('sphere', radius) | None for the
    scene's matching safety YAML. Same lookup as
    ``scripts/plot_campaign_compare.py::_safety_goal_region_for_scene``."""
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


def _ned_to_mocap_via_scene(scene_yaml: Path, positions_ned: np.ndarray) -> np.ndarray:
    from falsify.geometry import Point
    from falsify.io import build_frame_graph, load_yaml
    scene_cfg = load_yaml(scene_yaml)
    fg = build_frame_graph(scene_cfg, base_path=scene_yaml.parent)
    ned_frame = fg.frame("ned")
    out = [
        fg.convert(Point(np.asarray(p, dtype=np.float64), frame=ned_frame), to="mocap").xyz
        for p in positions_ned
    ]
    return np.asarray(out, dtype=np.float64)


def _gather_trials(campaign_dir: Path) -> list[dict]:
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


def _load_policy_manifest(campaign_dir: Path) -> dict:
    p = campaign_dir / "policy_manifest.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def _policy_subtitle(policy_manifest: dict) -> str:
    """One-line policy traceability for chart subtitles."""
    trace = (policy_manifest or {}).get("traceability") or {}
    bits = []
    variant = trace.get("variant")
    if variant:
        bits.append(variant)
    wandb_run = trace.get("wandb_run")
    if wandb_run:
        bits.append(f"wandb={wandb_run}")
    step = trace.get("step")
    if step is not None:
        bits.append(f"step={step}")
    return " · ".join(bits)


# ----------------------------- scene context -----------------------------

def _draw_scene_context(fig, trials, *, max_cloud_points: int):
    """Per unique scene_key draw: scene_edit-applied scene PLYs, nominal
    gate AABB, goal marker + tolerance region. Toggleable via
    legendgroup ``context_<scene_key>``."""
    from falsify.geometry import PointCloud
    from falsify.io import load_yaml, build_frame_graph
    from falsify.sim.scene_edits import apply_edits_to_scene_object, load_scene_edits
    from falsify.visualization import read_ply, subsample
    import plotly.graph_objects as go

    seen: dict[str, Path] = {}
    for t in trials:
        sk = t["summary"]["scene_key"]
        if sk not in seen:
            seen[sk] = REPO_ROOT / t["summary"]["scene"]

    for scene_key, scene_path in seen.items():
        scene_cfg = load_yaml(scene_path)
        scene_dir = scene_path.parent
        fg = build_frame_graph(scene_cfg, base_path=scene_dir)
        edits = load_scene_edits(scene_cfg)
        ctx_group = f"context_{scene_key}"

        region = scene_cfg.get("gate_region") or {}
        if region:
            aabb_min = np.asarray(region["aabb_min"], dtype=np.float64)
            aabb_max = np.asarray(region["aabb_max"], dtype=np.float64)
            xs, ys, zs = _gate_aabb_wireframe(aabb_min, aabb_max)
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs, mode="lines",
                line=dict(color="rgba(80,80,80,0.8)", width=3),
                name=f"{scene_key}: nominal gate AABB",
                legendgroup=ctx_group, hoverinfo="skip",
            ))

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
                            "not an integer multiple"
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
                marker=dict(size=1.2, color=rgb, opacity=0.5),
                name=f"{scene_key}: {entry['name']}",
                legendgroup=ctx_group, hoverinfo="skip",
            ))

        goal = scene_cfg.get("goal_position_mocap")
        if goal:
            g = np.asarray(goal, dtype=np.float64)
            fig.add_trace(go.Scatter3d(
                x=[g[0]], y=[g[1]], z=[g[2]],
                mode="markers+text",
                marker=dict(size=6, color="rgba(50,200,50,0.95)", symbol="diamond"),
                text=[f"goal:{scene_key}"], textposition="top center",
                name=f"{scene_key}: goal",
                legendgroup=ctx_group,
            ))
            tol = _safety_goal_region(scene_path)
            if tol is not None:
                kind, geom = tol
                if kind == "box":
                    xs, ys, zs = _box_wireframe(g, geom)
                    label = (f"{scene_key}: goal box "
                             f"(half_extents={[round(v,2) for v in geom.tolist()]})")
                else:
                    xs, ys, zs = _sphere_wireframe(g, geom)
                    label = f"{scene_key}: goal sphere (r={geom} m)"
                fig.add_trace(go.Scatter3d(
                    x=xs, y=ys, z=zs, mode="lines",
                    line=dict(color="rgb(10,90,30)", width=3, dash="dot"),
                    name=label, legendgroup=ctx_group, hoverinfo="skip",
                ))


def _draw_trial_overlays(fig, trials):
    """Per-trial rollout polyline + start/end markers; per-trial
    recovery polyline when present; per-trial perturbed gate AABB
    when ``gate_perturbation`` is set on the trial.

    Legend groups:
      ``<scene_key>/<outcome>``           → rollouts (one legend entry
                                            per group; subsequent trials
                                            ride along, showlegend=False)
      ``<scene_key>/recovery``            → all recoveries for the scene
      ``<scene_key>/perturbed_gates``     → all per-trial AABB wireframes
    """
    import plotly.graph_objects as go

    legend_seen: set[str] = set()
    for t in trials:
        s = t["summary"]
        scene_key = s["scene_key"]
        outcome = s.get("posthoc_outcome") or (
            s["failure"]["type"] if s.get("failure") else "SUCCESS"
        )
        trial_idx = s["trial_index"]
        scene_path = REPO_ROOT / s["scene"]
        color = ROLLOUT_COLOR.get(outcome, "rgb(120,120,120)")

        rollout_ned = np.load(t["rollout_npz"], allow_pickle=True)["positions_ned"]
        rollout_mocap = _ned_to_mocap_via_scene(scene_path, rollout_ned)

        rollout_key = f"{scene_key}/{outcome}"
        show_rollout = rollout_key not in legend_seen
        legend_seen.add(rollout_key)

        fig.add_trace(go.Scatter3d(
            x=rollout_mocap[:, 0], y=rollout_mocap[:, 1], z=rollout_mocap[:, 2],
            mode="lines",
            line=dict(color=color, width=3),
            name=f"{scene_key} · {outcome}",
            legendgroup=rollout_key, showlegend=show_rollout,
            hovertemplate=(
                f"<b>{scene_key} · {outcome}</b><br>"
                f"trial {trial_idx}<br>"
                "step=%{pointNumber}<br>"
                "mocap=(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>"
            ),
        ))
        fig.add_trace(go.Scatter3d(
            x=[rollout_mocap[0, 0]], y=[rollout_mocap[0, 1]], z=[rollout_mocap[0, 2]],
            mode="markers",
            marker=dict(size=4, color="rgb(20,20,20)", symbol="circle"),
            legendgroup=rollout_key, showlegend=False,
            name=f"start t{trial_idx:03d}", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter3d(
            x=[rollout_mocap[-1, 0]], y=[rollout_mocap[-1, 1]], z=[rollout_mocap[-1, 2]],
            mode="markers",
            marker=dict(size=5, color=color, symbol="x",
                        line=dict(width=1.5, color="black")),
            legendgroup=rollout_key, showlegend=False,
            name=f"end t{trial_idx:03d}", hoverinfo="skip",
        ))

        if t["recovery_npz"] is not None:
            recv_ned = np.load(t["recovery_npz"], allow_pickle=True)["positions_ned"]
            recv_mocap = _ned_to_mocap_via_scene(scene_path, recv_ned)
            recv_key = f"{scene_key}/recovery"
            show_recv = recv_key not in legend_seen
            legend_seen.add(recv_key)
            fig.add_trace(go.Scatter3d(
                x=recv_mocap[:, 0], y=recv_mocap[:, 1], z=recv_mocap[:, 2],
                mode="lines",
                line=dict(color=RECOVERY_COLOR, width=3, dash="dash"),
                name=f"{scene_key} · recovery",
                legendgroup=recv_key, showlegend=show_recv,
                hovertemplate=(
                    f"<b>{scene_key} · recovery</b><br>"
                    f"trial {trial_idx}<br>"
                    "step=%{pointNumber}<br>"
                    "mocap=(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>"
                ),
            ))
            fig.add_trace(go.Scatter3d(
                x=[recv_mocap[0, 0]], y=[recv_mocap[0, 1]], z=[recv_mocap[0, 2]],
                mode="markers",
                marker=dict(size=6, color=RECOVERY_COLOR, symbol="circle-open",
                            line=dict(width=2, color="black")),
                legendgroup=recv_key, showlegend=False,
                name=f"recv-seed t{trial_idx:03d}", hoverinfo="skip",
            ))

        # Per-trial perturbed gate AABB (only when the trial had a
        # gate perturbation; for unperturbed runs `gate_aabb_mocap`
        # equals the nominal AABB and adding it would just duplicate
        # the context wireframe).
        if s.get("gate_perturbation") and s.get("gate_aabb_mocap") is not None:
            aabb = np.asarray(s["gate_aabb_mocap"], dtype=np.float64)
            xs, ys, zs = _gate_aabb_wireframe(aabb[0], aabb[1])
            pert_key = f"{scene_key}/perturbed_gates"
            show_pert = pert_key not in legend_seen
            legend_seen.add(pert_key)
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs, mode="lines",
                line=dict(color=color, width=1.5),
                name=f"{scene_key} · perturbed gate AABB",
                legendgroup=pert_key, showlegend=show_pert,
                hovertemplate=(
                    f"<b>perturbed gate</b><br>"
                    f"{scene_key} · trial {trial_idx}<br>"
                    f"Δxyz={s['gate_perturbation']['delta_xyz']}<br>"
                    f"Δyaw={s['gate_perturbation']['delta_yaw_rad']:.3f} rad"
                    "<extra></extra>"
                ),
            ))


# ----------------------------- public API --------------------------------

def emit_trajectories_html(
    campaign_dir: Path,
    out_path: Optional[Path] = None,
    *,
    max_cloud_points: int = 4000,
) -> Path:
    """Write a single-figure rollouts overlay for one campaign.

    Returns the written path. If ``out_path`` is None, writes to
    ``<campaign_dir>/viz/trajectories.html``.
    """
    import plotly.graph_objects as go

    trials = _gather_trials(campaign_dir)
    if not trials:
        raise RuntimeError(f"no loadable trials in {campaign_dir}")

    fig = go.Figure()
    _draw_scene_context(fig, trials, max_cloud_points=max_cloud_points)
    _draw_trial_overlays(fig, trials)

    by_outcome = defaultdict(int)
    for t in trials:
        by_outcome[
            t["summary"].get("posthoc_outcome") or "UNKNOWN"
        ] += 1
    outcome_str = " · ".join(f"{v} {k}" for k, v in sorted(by_outcome.items()))
    n_total = len(trials)
    n_succ = by_outcome.get("SUCCESS", 0)
    pct = (100.0 * n_succ / n_total) if n_total else 0.0

    manifest = _load_policy_manifest(campaign_dir)
    subtitle_bits = [
        f"{n_succ}/{n_total} SUCCESS ({pct:.1f}%)",
        outcome_str,
    ]
    pol_sub = _policy_subtitle(manifest)
    if pol_sub:
        subtitle_bits.insert(0, pol_sub)

    fig.update_layout(
        title=(
            f"Rollouts — {campaign_dir.name}<br>"
            f"<sub>{'  ·  '.join(subtitle_bits)}<br>"
            "Color = posthoc outcome (green=SUCCESS, red=COLLISION_GATE, "
            "orange=MISS_GATE). Cyan dashed = recovery trajectory "
            "(open-circle = last-safe seed). Thin colored AABB = per-trial "
            "perturbed gate pose.</sub>"
        ),
        scene=dict(
            xaxis=dict(title="mocap x (m)"),
            yaxis=dict(title="mocap y (m)"),
            zaxis=dict(title="mocap z (m, up)"),
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.0)),
        ),
        height=860,
        margin=dict(l=0, r=0, t=110, b=0),
        legend=dict(itemsizing="constant", itemclick="toggle",
                    itemdoubleclick="toggleothers"),
    )

    out_path = out_path or (campaign_dir / "viz" / "trajectories.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    return out_path


def emit_outcome_charts_html(
    campaign_dir: Path,
    out_path: Optional[Path] = None,
) -> Path:
    """Per-scene stacked-bar chart of post-hoc outcomes for one campaign.

    Reproduces the visual style of
    ``runs/eval_campaigns/summary_charts_20260519.html``: bar per
    scene_key on the x-axis, stacked by outcome with the shared palette,
    each segment labeled with ``n/total (pct%)``.
    """
    import plotly.graph_objects as go

    cs_path = campaign_dir / "campaign_summary.json"
    if not cs_path.is_file():
        raise RuntimeError(f"missing campaign_summary.json under {campaign_dir}")
    cs = json.loads(cs_path.read_text())

    # Re-derive per-scene × outcome counts from the per-trial list (the
    # top-level ``by_outcome`` is global, not per-scene). Walk the
    # campaign's trial dirs since ``campaign_summary.json["trials"]`` is
    # also written but is large and we want to be tolerant of older
    # summaries that omitted it.
    per_scene: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    scene_totals: dict[str, int] = defaultdict(int)
    for t in _gather_trials(campaign_dir):
        s = t["summary"]
        sk = s["scene_key"]
        outcome = s.get("posthoc_outcome") or (
            s["failure"]["type"] if s.get("failure") else "SUCCESS"
        )
        per_scene[sk][outcome] += 1
        scene_totals[sk] += 1

    if not per_scene:
        raise RuntimeError(f"no trials with episode_summary.json under {campaign_dir}")

    scene_keys = sorted(per_scene.keys())
    # Outcomes actually present, ordered by OUTCOME_ORDER first then any
    # leftover alphabetically.
    present = {o for sk in scene_keys for o in per_scene[sk]}
    ordered_outcomes = [o for o in OUTCOME_ORDER if o in present] + sorted(
        o for o in present if o not in OUTCOME_ORDER
    )

    fig = go.Figure()
    for outcome in ordered_outcomes:
        ys, texts = [], []
        for sk in scene_keys:
            n = per_scene[sk].get(outcome, 0)
            total = scene_totals[sk]
            ys.append(n)
            texts.append(f"{n}/{total} ({100*n/total:.0f}%)" if n else "")
        fig.add_trace(go.Bar(
            x=scene_keys, y=ys,
            name=outcome,
            marker=dict(color=ROLLOUT_COLOR.get(outcome, "rgb(120,120,120)")),
            text=texts, textposition="inside", insidetextanchor="middle",
            hovertemplate=f"<b>%{{x}}</b><br>{outcome}: %{{y}}<extra></extra>",
        ))

    manifest = _load_policy_manifest(campaign_dir)
    pol_sub = _policy_subtitle(manifest)
    scenario = cs.get("scenario", "(unknown)")
    n_total = sum(scene_totals.values())
    n_succ = sum(per_scene[sk].get("SUCCESS", 0) for sk in scene_keys)
    title = f"{scenario} — {campaign_dir.name}<br>"
    sub = [f"{n_succ}/{n_total} SUCCESS ({100*n_succ/max(n_total,1):.1f}%)"]
    if pol_sub:
        sub.insert(0, pol_sub)
    title += f"<sub>{'  ·  '.join(sub)}</sub>"

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        barmode="stack",
        xaxis=dict(tickangle=-20, title=""),
        yaxis=dict(title="trials"),
        height=520, width=max(720, 180 * len(scene_keys) + 200),
        margin=dict(t=110, b=80, l=70, r=30),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18,
                    xanchor="center", x=0.5),
    )

    out_path = out_path or (campaign_dir / "viz" / "outcome_charts.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    return out_path


def emit_sweep_grid_html(
    campaign_dirs: list[Path],
    out_path: Path,
    *,
    row_axis: str = "policy",
    col_axis: str = "scenario",
) -> Path:
    """Cross-campaign 2-D facet of per-scene stacked outcome bars.

    Each cell of the grid is the same kind of stacked-bar chart as
    ``emit_outcome_charts_html`` produces for a single campaign — but
    laid out as rows × cols according to ``row_axis`` / ``col_axis``
    (each ∈ {"policy", "scenario"}). The 4×4 sweep produces a 4-row
    (policy) × 4-col (scenario) grid; smaller sweeps shrink the grid.

    Reads ``campaign_summary.json`` + per-trial ``episode_summary.json``
    from each dir. Falls back to (UNKNOWN-policy / UNKNOWN-scenario)
    when a manifest is missing — the cell still renders.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if row_axis == col_axis or row_axis not in ("policy", "scenario") \
            or col_axis not in ("policy", "scenario"):
        raise ValueError("row_axis/col_axis must be 'policy' or 'scenario' and differ")

    # Per-campaign roll-up: (row_key, col_key) → per-scene outcome counts
    cells: dict[tuple[str, str], dict[str, dict[str, int]]] = {}
    scene_totals: dict[tuple[str, str], dict[str, int]] = {}
    cell_meta: dict[tuple[str, str], dict] = {}

    for cdir in campaign_dirs:
        cs_path = cdir / "campaign_summary.json"
        if not cs_path.is_file():
            continue
        cs = json.loads(cs_path.read_text())
        manifest = _load_policy_manifest(cdir)
        # row/col keys
        scenario = cs.get("scenario") or "UNKNOWN"
        # Policy id: prefer the new layout's parent-dir name (== YAML
        # stem), fall back to the policy_config_path stem.
        policy_id = cdir.parent.name
        if policy_id in ("eval_campaigns", "legacy") or policy_id.startswith("run-"):
            pc = manifest.get("policy_config_path") or ""
            policy_id = Path(pc).stem if pc else "UNKNOWN"
        rk = {"policy": policy_id, "scenario": scenario}[row_axis]
        ck = {"policy": policy_id, "scenario": scenario}[col_axis]
        key = (rk, ck)

        per_scene: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        totals: dict[str, int] = defaultdict(int)
        for t in _gather_trials(cdir):
            s = t["summary"]
            sk = s["scene_key"]
            outcome = s.get("posthoc_outcome") or (
                s["failure"]["type"] if s.get("failure") else "SUCCESS"
            )
            per_scene[sk][outcome] += 1
            totals[sk] += 1
        if not per_scene:
            continue
        cells[key] = per_scene
        scene_totals[key] = totals
        cell_meta[key] = {
            "scenario": scenario, "policy_id": policy_id,
            "n_total": sum(totals.values()),
            "n_succ": sum(per_scene[sk].get("SUCCESS", 0) for sk in totals),
            "campaign_dir": str(cdir),
        }

    if not cells:
        raise RuntimeError("no campaign dirs had loadable trials")

    # Stable row/col orders: pull both axes from observed keys, but
    # use OUTCOME_ORDER-style preference for known scenarios and a
    # natural sort for policy ids.
    SCENARIO_ORDER = ["pure", "gate_perturbed_small", "gate_perturbed_large",
                      "compositional"]
    def _sorted(axis: str, values: set[str]) -> list[str]:
        if axis == "scenario":
            ordered = [v for v in SCENARIO_ORDER if v in values]
            return ordered + sorted(v for v in values if v not in SCENARIO_ORDER)
        return sorted(values)

    rows = _sorted(row_axis, {k[0] for k in cells})
    cols = _sorted(col_axis, {k[1] for k in cells})

    # Subplot titles (cell headers): "<row_key> / <col_key>: n_succ/n_total (pct%)"
    titles = []
    for r in rows:
        for c in cols:
            m = cell_meta.get((r, c))
            if m is None:
                titles.append(f"{r} / {c}: (no data)")
            else:
                pct = 100 * m["n_succ"] / max(m["n_total"], 1)
                titles.append(
                    f"<b>{r}</b> / <b>{c}</b><br>"
                    f"<sub>{m['n_succ']}/{m['n_total']} SUCCESS ({pct:.1f}%)</sub>"
                )

    fig = make_subplots(
        rows=len(rows), cols=len(cols),
        subplot_titles=titles,
        vertical_spacing=0.07, horizontal_spacing=0.04,
    )

    # All outcomes seen anywhere in the sweep (preserve OUTCOME_ORDER).
    all_outcomes = set()
    for per_scene in cells.values():
        for d in per_scene.values():
            all_outcomes.update(d.keys())
    ordered_outcomes = [o for o in OUTCOME_ORDER if o in all_outcomes] + sorted(
        o for o in all_outcomes if o not in OUTCOME_ORDER
    )

    # One legend entry per outcome (shared via legendgroup); only the
    # first cell renders the legend marker.
    legend_seen: set[str] = set()
    for ri, r in enumerate(rows, start=1):
        for ci, c in enumerate(cols, start=1):
            per_scene = cells.get((r, c))
            if per_scene is None:
                continue
            totals = scene_totals[(r, c)]
            scene_keys = sorted(per_scene.keys())
            for outcome in ordered_outcomes:
                ys, texts = [], []
                for sk in scene_keys:
                    n = per_scene[sk].get(outcome, 0)
                    tot = totals[sk]
                    ys.append(n)
                    texts.append(f"{n}/{tot} ({100*n/tot:.0f}%)" if n else "")
                show_legend = outcome not in legend_seen
                legend_seen.add(outcome)
                fig.add_trace(
                    go.Bar(
                        x=scene_keys, y=ys, name=outcome,
                        marker=dict(color=ROLLOUT_COLOR.get(outcome, "rgb(120,120,120)")),
                        text=texts, textposition="inside",
                        insidetextanchor="middle",
                        legendgroup=outcome, showlegend=show_legend,
                        hovertemplate=(
                            f"<b>%{{x}}</b><br>{outcome}: %{{y}}<extra></extra>"
                        ),
                    ),
                    row=ri, col=ci,
                )
            fig.update_xaxes(tickangle=-25, row=ri, col=ci)

    # Top-level sweep stats.
    total_trials = sum(m["n_total"] for m in cell_meta.values())
    total_succ = sum(m["n_succ"] for m in cell_meta.values())
    pct = 100 * total_succ / max(total_trials, 1)
    title = (
        f"Eval sweep — {len(rows)} {row_axis} × {len(cols)} {col_axis} "
        f"= {len(cell_meta)} campaigns, {total_trials} trials<br>"
        f"<sub>{total_succ}/{total_trials} SUCCESS ({pct:.1f}%) overall · "
        "color = posthoc outcome (green=SUCCESS, red=COLLISION_GATE, "
        "orange=MISS_GATE)</sub>"
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        barmode="stack",
        height=max(360, 320 * len(rows) + 80),
        width=max(900, 360 * len(cols) + 200),
        margin=dict(t=120, b=80, l=70, r=30),
        legend=dict(orientation="h", yanchor="bottom", y=-0.05,
                    xanchor="center", x=0.5),
    )
    # Subplot title font tweak — the default size is too small for the
    # 2-line headers we generate above.
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(size=12)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    return out_path


__all__ = ["emit_trajectories_html", "emit_outcome_charts_html",
           "emit_sweep_grid_html",
           "ROLLOUT_COLOR", "RECOVERY_COLOR", "OUTCOME_ORDER"]
