"""Print a per-scenario summary for one or more eval campaigns.

Reads ``campaign_summary.json`` from each given campaign dir and prints
success / failure-type breakdowns. When more than one campaign is given,
the breakdowns line up side-by-side for direct policy comparison on the
same trial cards.

Usage:

    PYTHONPATH=src python scripts/eval/summarize_eval_campaign.py \\
        runs/eval_campaigns/pi07_history_pure \\
        runs/eval_campaigns/pi07_nonhistory_pure
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _load(camp_dir: Path) -> dict:
    cs_path = camp_dir / "campaign_summary.json"
    if not cs_path.is_file():
        raise SystemExit(f"missing campaign_summary.json under {camp_dir}")
    return json.loads(cs_path.read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("campaigns", nargs="+", type=Path,
                    help="One or more campaign output dirs.")
    args = ap.parse_args()

    summaries = [(c, _load(c)) for c in args.campaigns]

    for camp_dir, cs in summaries:
        print(f"\n=== {camp_dir.name} ({cs['scenario']}) ===")
        print(f"  policy: {cs['policy_config']}")
        n = cs["n_trials_total"]
        succ = cs["n_succeeded"]
        rate = succ / n * 100 if n else 0.0
        print(f"  total trials: {n}")
        print(f"  succeeded:    {succ} ({rate:.1f}%)")
        print(f"  failure type breakdown:")
        for ftype, count in sorted(cs["by_failure_type"].items(),
                                   key=lambda kv: -kv[1]):
            if ftype == "NONE":
                continue
            pct = count / n * 100 if n else 0.0
            print(f"    {ftype:24s} {count:4d}  ({pct:.1f}%)")
        print(f"  per-scene success:")
        for scene_key, stats in sorted(cs["by_scene"].items()):
            print(f"    {scene_key:20s} {stats['succeeded']:3d} / {stats['n']:3d}")
        print(f"  elapsed: {cs['elapsed_total_s']:.0f}s")

    if len(summaries) >= 2:
        print(f"\n=== cross-campaign comparison ===")
        all_ftypes = sorted({ft for _, cs in summaries for ft in cs["by_failure_type"]})
        header = f"{'failure_type':<24s}" + "".join(
            f"{c.name:>20s}" for c, _ in summaries)
        print(header)
        for ft in all_ftypes:
            row = f"{ft:<24s}"
            for _, cs in summaries:
                n = cs["n_trials_total"]
                cnt = cs["by_failure_type"].get(ft, 0)
                row += f"{cnt:>10d} ({cnt/n*100:5.1f}%)"
            print(row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
