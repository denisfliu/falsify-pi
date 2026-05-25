"""Re-render per-campaign viz from on-disk artifacts (no rollout needed).

Reads ``<campaign_dir>/campaign_summary.json`` and the per-trial
``episode_summary.json`` / ``rollout_states.npz`` files; writes
``<campaign_dir>/viz/trajectories.html`` and
``<campaign_dir>/viz/outcome_charts.html``.

Useful for:
  - backfilling viz into older campaign dirs that pre-date the
    auto-emit hook in ``scripts/eval/run_eval_campaign.py``
  - iterating on the renderer code in
    ``src.falsify.visualization.eval_report`` without re-running the
    campaign

Usage:

    PYTHONPATH=src python scripts/eval/plot_eval_run.py <campaign_dir>
    PYTHONPATH=src python scripts/eval/plot_eval_run.py <campaign_dir> --trajectories-only
    PYTHONPATH=src python scripts/eval/plot_eval_run.py <campaign_dir> --charts-only
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
    ap.add_argument("--no-drone-obb", dest="show_drone_obb",
                    action="store_false",
                    help="Skip emitting per-trial drone-body OBB traces. "
                         "By default the OBB traces are emitted (hidden) "
                         "with a 'Drone OBB' toggle button at top-right.")
    ap.set_defaults(show_drone_obb=True)
    ap.add_argument("--obb-stride", type=int, default=25,
                    help="Sample one OBB every N rollout steps. Default 25 "
                         "(~1 box/s at 30 Hz). Lower = denser OBBs.")
    ap.add_argument("--obb-half-extents", type=float, nargs=3, default=None,
                    metavar=("HX", "HY", "HZ"),
                    help="Override body-frame OBB half-extents (FRD: x fwd, "
                         "y right, z down). Default reads from the per-scene "
                         "safety YAML's drone_body.half_extents.")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--trajectories-only", action="store_true")
    grp.add_argument("--charts-only", action="store_true")
    args = ap.parse_args(argv)

    if not args.campaign_dir.is_dir():
        raise SystemExit(f"not a directory: {args.campaign_dir}")

    import numpy as np
    from falsify.visualization.eval_report import (
        emit_outcome_charts_html, emit_trajectories_html,
    )

    if not args.charts_only:
        p = emit_trajectories_html(
            args.campaign_dir,
            max_cloud_points=args.max_cloud_points,
            show_drone_obb=args.show_drone_obb,
            obb_stride=args.obb_stride,
            obb_half_extents=(
                np.asarray(args.obb_half_extents, dtype=np.float64)
                if args.obb_half_extents is not None else None
            ),
        )
        print(f"[plot] wrote {p}  ({p.stat().st_size // 1024} KB)")
    if not args.trajectories_only:
        p = emit_outcome_charts_html(args.campaign_dir)
        print(f"[plot] wrote {p}  ({p.stat().st_size // 1024} KB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
