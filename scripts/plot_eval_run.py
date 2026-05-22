"""Re-render per-campaign viz from on-disk artifacts (no rollout needed).

Reads ``<campaign_dir>/campaign_summary.json`` and the per-trial
``episode_summary.json`` / ``rollout_states.npz`` files; writes
``<campaign_dir>/viz/trajectories.html`` and
``<campaign_dir>/viz/outcome_charts.html``.

Useful for:
  - backfilling viz into older campaign dirs that pre-date the
    auto-emit hook in ``scripts/run_eval_campaign.py``
  - iterating on the renderer code in
    ``src.falsify.visualization.eval_report`` without re-running the
    campaign

Usage:

    PYTHONPATH=src python scripts/plot_eval_run.py <campaign_dir>
    PYTHONPATH=src python scripts/plot_eval_run.py <campaign_dir> --trajectories-only
    PYTHONPATH=src python scripts/plot_eval_run.py <campaign_dir> --charts-only
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("campaign_dir", type=Path,
                    help="Path to a campaign output dir.")
    ap.add_argument("--max-cloud-points", type=int, default=4000,
                    help="Per-PLY subsample budget for the trajectories plot.")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--trajectories-only", action="store_true")
    grp.add_argument("--charts-only", action="store_true")
    args = ap.parse_args(argv)

    if not args.campaign_dir.is_dir():
        raise SystemExit(f"not a directory: {args.campaign_dir}")

    from falsify.visualization.eval_report import (
        emit_outcome_charts_html, emit_trajectories_html,
    )

    if not args.charts_only:
        p = emit_trajectories_html(
            args.campaign_dir, max_cloud_points=args.max_cloud_points,
        )
        print(f"[plot] wrote {p}  ({p.stat().st_size // 1024} KB)")
    if not args.trajectories_only:
        p = emit_outcome_charts_html(args.campaign_dir)
        print(f"[plot] wrote {p}  ({p.stat().st_size // 1024} KB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
