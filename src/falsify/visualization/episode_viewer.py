"""Interactive HTML replay of an episode using plotly.

Renders the nominal trajectory, recovery trajectory (if any), and markers in
a chosen frame. Falls back to a no-op if plotly is unavailable.

The output file is a self-contained html — open it in any browser.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from falsify.geometry import FrameGraph


def html_replay(
    episode,
    frame_graph: FrameGraph,
    out_path: str | Path,
    *,
    view_frame: str = "ned",
) -> Optional[Path]:
    """Write an html replay to `out_path`. Returns the path, or None if plotly is missing."""
    try:
        import plotly.graph_objects as go  # type: ignore
    except ImportError:
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = go.Figure()

    # Nominal trajectory.
    if episode.trace.states:
        traj = episode.trace.trajectory()
        converted = frame_graph.convert(traj, to=view_frame)
        fig.add_trace(go.Scatter3d(
            x=converted.positions[:, 0],
            y=converted.positions[:, 1],
            z=converted.positions[:, 2],
            mode="lines+markers",
            name="nominal",
            line=dict(color="rgb(50,165,242)", width=4),
            marker=dict(size=2),
        ))

    # Recovery trajectory.
    if episode.recovery_trajectory is not None:
        rec = frame_graph.convert(episode.recovery_trajectory, to=view_frame)
        fig.add_trace(go.Scatter3d(
            x=rec.positions[:, 0],
            y=rec.positions[:, 1],
            z=rec.positions[:, 2],
            mode="lines+markers",
            name="recovery",
            line=dict(color="rgb(217,77,77)", width=4),
            marker=dict(size=2),
        ))

    # Markers.
    from .frame_debugger import _markers, DEFAULT_COLORS
    markers = {}
    if episode.trace.states:
        markers["start"] = episode.trace.states[0].pos
    if episode.goal is not None:
        markers["goal"] = episode.goal
    if episode.failure is not None:
        markers["failure"] = episode.failure.failure_state.pos
        if episode.failure.last_safe_state is not None:
            markers["last_safe"] = episode.failure.last_safe_state.pos

    for name, pt in markers.items():
        converted = frame_graph.convert(pt, to=view_frame)
        c = DEFAULT_COLORS.get(name, (0.5, 0.5, 0.5))
        fig.add_trace(go.Scatter3d(
            x=[converted.xyz[0]], y=[converted.xyz[1]], z=[converted.xyz[2]],
            mode="markers",
            name=name,
            marker=dict(
                size=8,
                color=f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})",
            ),
        ))

    fig.update_layout(
        title=f"falsify episode (frame: {view_frame})",
        scene=dict(
            xaxis_title=f"x [{view_frame}]",
            yaxis_title=f"y [{view_frame}]",
            zaxis_title=f"z [{view_frame}]",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    return out_path
