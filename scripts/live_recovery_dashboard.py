"""Self-refreshing HTML dashboard for in-flight recovery_collection runs.

Polls the newest ``run-*`` dir under each watched ``<policy>/<scene>``
pair, builds a single HTML with:
  - per-scene progress counters (n_recoveries / target, n_trials, outcome
    histogram, last update),
  - a 3-D plot of all collected rollouts (colored by posthoc outcome) +
    recovery polylines (cyan dashed) overlaid on each scene's
    scene-edits-applied PLY context.

The HTML carries a ``<meta http-equiv="refresh">`` tag so an open
browser tab reloads the file from disk every N seconds. The script
itself regenerates the file in place on its own poll interval; the
browser refresh just re-fetches whatever's on disk at that moment.

Usage (sit alongside a running collection):

    PYTHONPATH=src python scripts/live_recovery_dashboard.py \\
        --policy-id nonhistory_real_synth_31ohxgxv_5000 \\
        --scenes left_gate right_gate \\
        --out runs/recovery_collection/_live_dashboard.html

Stop with Ctrl-C (or kill the background process). One-shot mode for
testing:

    PYTHONPATH=src python scripts/live_recovery_dashboard.py --once …
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent.parent


def _newest_run_dir(policy_id: str, scene_key: str) -> Optional[Path]:
    """Return the newest ``run-*`` dir under
    runs/recovery_collection/<policy_id>/<scene_key>/, or None if none yet."""
    pat = REPO_ROOT / "runs" / "recovery_collection" / policy_id / scene_key / "run-*"
    matches = sorted(glob.glob(str(pat)))
    return Path(matches[-1]) if matches else None


def _gather_stats(run_dir: Path) -> dict:
    """Cheap stats — count trial summaries + recoveries, read manifest
    for target. No PLY / Plotly cost here."""
    out = {
        "run_name": run_dir.name,
        "n_trials": 0,
        "n_recoveries": 0,
        "target": None,
        "outcomes": {},
        "started_at": None,
        "finished_at": None,
    }
    manifest_path = run_dir / "collection_manifest.json"
    if manifest_path.is_file():
        try:
            m = json.loads(manifest_path.read_text())
            out["target"] = m.get("target_n_recoveries")
            out["started_at"] = m.get("started_at")
            out["finished_at"] = m.get("finished_at")
        except Exception:
            pass

    recoveries_dir = run_dir / "recoveries"
    if recoveries_dir.is_dir():
        out["n_recoveries"] = len(list(recoveries_dir.glob("recovery_*.npz")))

    # episode_summary.json under <scene_key>/trial_NNN/ — count + tally
    for summary_path in run_dir.glob("*/trial_*/episode_summary.json"):
        out["n_trials"] += 1
        try:
            s = json.loads(summary_path.read_text())
            o = s.get("posthoc_outcome") or "UNKNOWN"
            out["outcomes"][o] = out["outcomes"].get(o, 0) + 1
        except Exception:
            pass
    return out


def _outcome_pill(name: str, count: int) -> str:
    """Inline-styled colored pill matching the eval_report palette."""
    colors = {
        "SUCCESS":         "#2ea043",
        "MISS_GATE":       "#e74c3c",
        "COLLISION_GATE":  "#b22222",
        "COLLISION_OTHER": "#9650b4",
        "OUT_OF_BOUNDS":   "#787878",
        "GOAL_NOT_REACHED": "#ffa500",
        "UNKNOWN":         "#999999",
    }
    bg = colors.get(name, "#777")
    return (
        f'<span style="background:{bg};color:white;'
        f'padding:2px 8px;border-radius:10px;'
        f'margin-right:6px;font-size:0.85em;white-space:nowrap;">'
        f'{name}={count}</span>'
    )


def _build_figure(run_dirs: list[Path]):
    """Aggregate trials across all watched dirs into one Plotly figure
    that reuses the eval_report draw helpers."""
    import plotly.graph_objects as go
    from falsify.visualization.eval_report import (
        _draw_scene_context, _draw_trial_overlays, _gather_trials,
    )

    fig = go.Figure()
    all_trials = []
    for rd in run_dirs:
        if rd is None or not rd.is_dir():
            continue
        all_trials.extend(_gather_trials(rd))

    if all_trials:
        _draw_scene_context(fig, all_trials, max_cloud_points=2500)
        _draw_trial_overlays(fig, all_trials)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="mocap x (m)"),
            yaxis=dict(title="mocap y (m)"),
            zaxis=dict(title="mocap z (m, up)"),
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.0)),
        ),
        height=720,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(itemsizing="constant", itemclick="toggle",
                    itemdoubleclick="toggleothers"),
    )
    return fig


def _scene_to_trace_indices(fig) -> dict[str, list[int]]:
    """Map ``scene_key → [trace_index, ...]`` for the per-scene filter
    checkboxes. Inferred from each trace's ``legendgroup``:

      - ``context_<scene>``          → scene-context wireframes / clouds
      - ``<scene>/<outcome>``        → rollout polylines + start/end markers
      - ``<scene>/recovery``         → recovery polyline + seed marker
      - ``<scene>/perturbed_gates``  → per-trial perturbed AABB wireframes

    Anything that doesn't parse falls under ``__unscoped__`` and stays
    visible regardless of the checkbox state.
    """
    out: dict[str, list[int]] = {}
    for i, tr in enumerate(fig.data):
        lg = (tr.legendgroup or "").strip()
        scene_key: str
        if lg.startswith("context_"):
            scene_key = lg[len("context_"):]
        elif "/" in lg:
            scene_key = lg.split("/", 1)[0]
        else:
            scene_key = "__unscoped__"
        out.setdefault(scene_key, []).append(i)
    return out


def _render_html(stats_per_scene: list[dict], fig, refresh_s: int,
                 scene_indices: dict[str, list[int]]) -> str:
    rows = []
    for s in stats_per_scene:
        target = s["target"] if s["target"] is not None else "?"
        pct = (100 * s["n_recoveries"] / s["target"]) if s["target"] else 0
        bar_inner = ""
        if s["target"]:
            bar_inner = (
                f'<div style="background:#2ea043;width:{min(pct,100):.0f}%;'
                f'height:100%;border-radius:6px;"></div>'
            )
        outcome_pills = "".join(
            _outcome_pill(k, v)
            for k, v in sorted(s["outcomes"].items(), key=lambda kv: -kv[1])
        ) or '<span style="color:#999;font-size:0.85em;">(no trials yet)</span>'
        status = "✅ done" if s["finished_at"] else ("⏳ running" if s["n_trials"] > 0 else "⏸ waiting")
        rows.append(
            f"<tr>"
            f'<td><b>{s["scene_key"]}</b><br>'
            f'<span style="color:#888;font-size:0.8em;">{s["run_name"] or "(no run dir yet)"}</span></td>'
            f'<td style="white-space:nowrap;">{s["n_recoveries"]}/{target} '
            f'<span style="color:#888;">({pct:.0f}%)</span></td>'
            f'<td style="width:200px;">'
            f'<div style="background:#eee;height:10px;border-radius:6px;overflow:hidden;">{bar_inner}</div></td>'
            f'<td>{s["n_trials"]}</td>'
            f'<td>{status}</td>'
            f'<td style="line-height:1.8;">{outcome_pills}</td>'
            f"</tr>"
        )

    # Per-scene checkboxes drive Plotly.restyle on the trace indices
    # for each scene's whole legendgroup family (context + rollouts +
    # recoveries + perturbed-gate wireframes). Build the checkbox HTML
    # only for scenes that produced traces — unscoped traces (gridlines
    # etc.) stay visible no matter what.
    scenes_with_traces = [s["scene_key"] for s in stats_per_scene
                          if scene_indices.get(s["scene_key"])]
    scene_checkboxes = "".join(
        f'<label class="scene-toggle">'
        f'<input type="checkbox" class="scene-cb" data-scene="{sk}" checked> '
        f'{sk}'
        f'</label>'
        for sk in scenes_with_traces
    )

    plot_div = fig.to_html(include_plotlyjs="cdn", full_html=False,
                           div_id="trajectories-plot")
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    indices_json = json.dumps(
        {k: v for k, v in scene_indices.items() if k != "__unscoped__"}
    )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Recovery collection — live</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          margin: 18px; color: #222; }}
  h2 {{ margin: 0 0 4px 0; }}
  .meta {{ color: #666; font-size: 0.85em; margin-bottom: 8px; }}
  .controls {{ display: flex; flex-wrap: wrap; align-items: center;
               gap: 18px; padding: 10px 14px; background: #f6f8fa;
               border-radius: 6px; margin-bottom: 14px;
               font-size: 0.9em; }}
  .controls .group-title {{ font-weight: 600; margin-right: 4px;
                            color: #555; }}
  .scene-toggle {{ cursor: pointer; user-select: none;
                   padding: 2px 6px; border-radius: 4px; }}
  .scene-toggle:hover {{ background: #e1e4e8; }}
  .scene-toggle input {{ vertical-align: middle; margin-right: 4px; }}
  table {{ border-collapse: collapse; margin-bottom: 16px; }}
  th, td {{ padding: 8px 12px; border-bottom: 1px solid #eee;
            text-align: left; vertical-align: middle; }}
  th {{ background: #f6f8fa; font-weight: 600; font-size: 0.9em;
        text-transform: uppercase; letter-spacing: 0.05em; color: #555; }}
  #refresh-status {{ color: #888; font-style: italic; }}
</style>
</head>
<body>
<h2>Recovery collection — live dashboard</h2>
<div class="meta">last regenerated <b>{now}</b></div>

<div class="controls">
  <label class="scene-toggle">
    <input type="checkbox" id="auto-refresh-cb" checked>
    <b>auto-refresh</b> every {refresh_s}s
    <span id="refresh-status"></span>
  </label>
  <span class="group-title">|  scenes:</span>
  {scene_checkboxes}
  <span style="color:#888;font-style:italic;margin-left:8px;">
    (toggle a scene to hide its context + ALL its trajectories)
  </span>
</div>

<table>
  <tr><th>Scene</th><th>Recoveries</th><th>Progress</th>
      <th>Trials</th><th>Status</th><th>Outcomes (per-trial)</th></tr>
  {''.join(rows)}
</table>
<div>{plot_div}</div>

<script>
(function() {{
  // Per-scene trace index map: {{scene_key: [trace_idx, ...]}}.
  const SCENE_INDICES = {indices_json};
  const REFRESH_MS = {refresh_s} * 1000;
  const graphDiv = document.getElementById("trajectories-plot");

  // ---- Hierarchical filter: per-scene checkbox toggles all traces ----
  function setSceneVisible(scene, visible) {{
    const idxs = SCENE_INDICES[scene];
    if (!idxs || !idxs.length) return;
    // visible=true → restore (true); visible=false → hide ('legendonly'
    // keeps the trace's legend entry interactive even when hidden by
    // the scene checkbox, so per-outcome toggles still work after
    // re-enabling the scene).
    const target = visible ? true : "legendonly";
    Plotly.restyle(graphDiv, {{visible: target}}, idxs);
  }}

  document.querySelectorAll(".scene-cb").forEach(cb => {{
    cb.addEventListener("change", () => {{
      setSceneVisible(cb.dataset.scene, cb.checked);
    }});
  }});

  // ---- Auto-refresh: JS timer, controllable via checkbox ----
  // We use window.location.reload so the page re-fetches the file the
  // background dashboard script regenerates every poll-seconds. The
  // checkbox cancels pending refresh and stops scheduling new ones.
  let refreshTimer = null;
  const refreshCb = document.getElementById("auto-refresh-cb");
  const status = document.getElementById("refresh-status");

  function tickdown(remaining_s) {{
    if (!refreshCb.checked) {{ status.textContent = " (paused)"; return; }}
    status.textContent = ` (next reload in ${{remaining_s}}s)`;
    if (remaining_s <= 0) {{ window.location.reload(); return; }}
    refreshTimer = setTimeout(() => tickdown(remaining_s - 1), 1000);
  }}

  function scheduleRefresh() {{
    if (refreshTimer) clearTimeout(refreshTimer);
    if (refreshCb.checked) tickdown(Math.floor(REFRESH_MS / 1000));
    else status.textContent = " (paused)";
  }}

  refreshCb.addEventListener("change", scheduleRefresh);
  scheduleRefresh();
}})();
</script>
</body>
</html>"""


def write_dashboard(policy_id: str, scenes: list[str], out: Path,
                    refresh_s: int) -> dict:
    """Single tick: discover newest run dir per scene, gather stats,
    build figure, write HTML atomically. Returns a summary dict."""
    stats_per_scene = []
    run_dirs = []
    for sk in scenes:
        rd = _newest_run_dir(policy_id, sk)
        run_dirs.append(rd)
        if rd is None:
            stats_per_scene.append({
                "scene_key": sk, "n_trials": 0, "n_recoveries": 0,
                "target": None, "outcomes": {}, "started_at": None,
                "finished_at": None, "run_name": None,
            })
        else:
            s = _gather_stats(rd)
            s["scene_key"] = sk
            stats_per_scene.append(s)

    fig = _build_figure(run_dirs)
    scene_indices = _scene_to_trace_indices(fig)
    html = _render_html(stats_per_scene, fig, refresh_s, scene_indices)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(html)
    tmp.replace(out)
    return {
        "stats": stats_per_scene,
        "size_kb": out.stat().st_size // 1024,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policy-id", required=True,
                    help="Policy YAML stem to scope discovery to (e.g. "
                         "nonhistory_real_synth_31ohxgxv_5000).")
    ap.add_argument("--scenes", nargs="+", required=True,
                    help="Scene keys to watch (e.g. left_gate right_gate).")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output HTML path.")
    ap.add_argument("--refresh-seconds", type=int, default=30,
                    help="Browser auto-refresh interval (meta tag).")
    ap.add_argument("--poll-seconds", type=int, default=15,
                    help="Disk-regen interval (the script's own loop).")
    ap.add_argument("--once", action="store_true",
                    help="Generate once and exit (for testing).")
    args = ap.parse_args()

    while True:
        t0 = time.time()
        try:
            info = write_dashboard(args.policy_id, args.scenes, args.out,
                                   args.refresh_seconds)
            dt = time.time() - t0
            counts = " · ".join(
                f"{s['scene_key']}={s['n_recoveries']}/{s.get('target') or '?'} "
                f"({s['n_trials']} trials)"
                for s in info["stats"]
            )
            print(f"[dashboard] wrote {args.out} ({info['size_kb']} KB, "
                  f"{dt:.1f}s) · {counts}")
        except Exception as e:  # noqa: BLE001
            import traceback as _tb
            print(f"[dashboard] error: {e}\n{_tb.format_exc()}")
        if args.once:
            break
        time.sleep(args.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
