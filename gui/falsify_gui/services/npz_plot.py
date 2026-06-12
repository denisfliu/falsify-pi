"""On-demand Plotly rendering of trajectory NPZs (GUI venv numpy — never torch).

Handles both artifact schemas:
- rollout_states.npz: times, positions_ned, quaternions_xyzw, velocities,
  failure_step, failure_type
- recovery_trajectory.npz / planned trajectories: times, positions_ned,
  quaternions_xyzw [, prompt, source]

Output is a self-contained HTML cached by (path, mtime).
"""
from __future__ import annotations

import hashlib

import numpy as np

from ..paths import PLOTS_CACHE, resolve_runs_path


def plot_npz(rel: str) -> str:
    """Render (or reuse cached) HTML for the NPZ at repo-relative `rel`.
    Returns the cache file name under /gui-cache/plots/."""
    src = resolve_runs_path(rel)
    key = hashlib.sha1(f"{rel}:{src.stat().st_mtime_ns}".encode()).hexdigest()[:16]
    out = PLOTS_CACHE / f"{key}.html"
    if out.exists():
        return out.name

    import plotly.graph_objects as go

    data = np.load(src, allow_pickle=False)
    pos = np.asarray(data["positions_ned"], dtype=float)
    times = np.asarray(data["times"], dtype=float) if "times" in data else np.arange(len(pos))
    # NED → plot axes: x=north, y=east, z=up
    x, y, z = pos[:, 0], pos[:, 1], -pos[:, 2]
    hover = [f"step {i}<br>t={t:.2f}s<br>ned=({px:.2f}, {py:.2f}, {pz:.2f})"
             for i, (t, px, py, pz) in enumerate(zip(times, pos[:, 0], pos[:, 1], pos[:, 2]))]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z, mode="lines",
        line=dict(width=4, color=times, colorscale="Viridis"),
        text=hover, hoverinfo="text", name="trajectory"))
    fig.add_trace(go.Scatter3d(
        x=[x[0]], y=[y[0]], z=[z[0]], mode="markers",
        marker=dict(size=6, color="#3fb950"), name="start"))

    def scalar(key):
        # str fields (failure_type, prompt, source) are object arrays that
        # need allow_pickle — skip them rather than unpickle
        try:
            return data[key] if key in data else None
        except ValueError:
            return None

    title = src.name
    if scalar("failure_step") is not None:
        fs = int(scalar("failure_step"))
        ft = scalar("failure_type")
        ft = str(ft) if ft is not None else ""
        if 0 <= fs < len(pos):
            fig.add_trace(go.Scatter3d(
                x=[x[fs]], y=[y[fs]], z=[z[fs]], mode="markers",
                marker=dict(size=7, color="#f85149", symbol="x"),
                name=f"failure @ {fs}"))
        title += f" — {ft} @ step {fs}"
    fig.add_trace(go.Scatter3d(
        x=[x[-1]], y=[y[-1]], z=[z[-1]], mode="markers",
        marker=dict(size=6, color="#d29922"), name="end"))

    fig.update_layout(
        title=title, template="plotly_dark", height=620,
        scene=dict(aspectmode="data",
                   xaxis_title="N (m)", yaxis_title="E (m)", zaxis_title="up (m)"),
        margin=dict(l=0, r=0, t=40, b=0))
    fig.write_html(out, include_plotlyjs="cdn")
    return out.name
