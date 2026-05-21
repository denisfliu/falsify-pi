"""Plot the offline action-prediction eval output as a single HTML.

Reads ``<run_dir>/per_step.npz`` + ``<run_dir>/summary.json`` produced by
``scripts/eval_action_prediction.py``. Lays out:

  - Per-dim pred-vs-gt scatter (one subplot per action dim).
  - Per-dim error trace, concatenated across episodes (vertical bands
    separate episodes).
  - Per-dim error histogram.

Usage:
    PYTHONPATH=src python scripts/plot_action_prediction.py \\
        --run runs/action_prediction/<name> [--out <run>/plot.html]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


DIM_LABELS = ["Δx", "Δy", "Δz", "Δyaw", "a4", "a5", "a6"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, type=Path,
                    help="Action-prediction run dir (containing per_step.npz + summary.json).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output HTML (default: <run>/plot.html).")
    args = ap.parse_args()

    run = args.run.resolve()
    out = args.out or (run / "plot.html")

    d = np.load(run / "per_step.npz")
    gt, pred, err = d["gt"], d["pred"], d["err"]
    ep_ids = d["episode_ids"]
    frame_idx = d["frame_indices"]
    summary = json.loads((run / "summary.json").read_text())
    n_dim = gt.shape[1]
    labels = DIM_LABELS[:n_dim] + [f"a{i}" for i in range(len(DIM_LABELS), n_dim)]

    # Skip dims that are constant zero in BOTH gt and pred to declutter.
    active = [i for i in range(n_dim) if not (np.allclose(gt[:, i], 0) and np.allclose(pred[:, i], 0))]
    n_active = len(active)

    fig = make_subplots(
        rows=n_active, cols=1,
        subplot_titles=[f"{labels[i]} — predicted vs dataset action" for i in active],
        vertical_spacing=0.06, shared_xaxes=True,
    )

    ep_boundaries = np.where(np.diff(ep_ids) != 0)[0] + 1
    steps = np.arange(len(gt))

    for row, dim in enumerate(active, start=1):
        # Ground-truth (dataset) action — solid black.
        fig.add_trace(go.Scatter(
            x=steps, y=gt[:, dim], mode="lines",
            line=dict(color="rgba(20,20,20,0.85)", width=1.5),
            name=("dataset (gt)" if row == 1 else None),
            legendgroup="gt", showlegend=(row == 1),
        ), row=row, col=1)
        # Predicted first-action of chunk — colored.
        fig.add_trace(go.Scatter(
            x=steps, y=pred[:, dim], mode="lines",
            line=dict(color="rgba(220,60,60,0.85)", width=1.5),
            name=("predicted (chunk[0])" if row == 1 else None),
            legendgroup="pred", showlegend=(row == 1),
        ), row=row, col=1)
        # Episode boundary verticals.
        for b in ep_boundaries:
            fig.add_vline(x=int(b), line=dict(color="rgba(0,0,0,0.2)", width=1),
                          row=row, col=1)
        fig.update_yaxes(title_text=labels[dim], row=row, col=1)
        if row == n_active:
            fig.update_xaxes(title_text="step (concatenated across episodes)",
                             row=row, col=1)

    n_eps = summary["n_episodes"]
    n_frames = summary["n_frames_total"]
    mse = summary["mse_overall"]
    rmse_per_dim = summary.get("rmse_per_dim", [])
    rmse_str = "  ".join(f"{labels[i]}={rmse_per_dim[i]:.3e}"
                         for i in active if i < len(rmse_per_dim))
    title = (f"Action prediction (MPC-style chunk[0]) — "
             f"{summary['gateway_url']}<br>"
             f"<sub>{n_eps} eps · {n_frames} frames · overall MSE={mse:.3e} · "
             f"RMSE: {rmse_str}</sub>")
    fig.update_layout(
        title=title, height=260 * n_active + 80, width=1200,
        margin=dict(l=60, r=20, t=80, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
