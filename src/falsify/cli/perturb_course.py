"""Generate variant course YAMLs from one base course by perturbing a waypoint.

The intended use case is corrective-maneuver dataset generation: shift the
``approach`` waypoint (or whichever one you pick) up / down / left / right
of its nominal position, plan and render each variant as one episode, and
the resulting parquet teaches the policy how to recover.

Each invocation writes a directory of ``<base>__<mode>_<NNN>.yaml`` files
that the rest of the pipeline ingests as-is:

  perturb_course → directory of course YAMLs
       │
       ▼  (for each variant)
  plan_trajectory  →  trajectory.npz
       │
       ▼
  export_training_data  →  episode_NNNNNN.parquet

To wire this end-to-end across all variants in one shot, point
``export_training_data --trajectories-dir`` at a directory of planned
NPZs (one per variant) — see the ``falsify-orchestrate-batch`` skill.

Example::

    PYTHONPATH=src .venv/bin/python -m falsify.cli.perturb_course \\
        --course configs/courses/through_left_gate.yaml \\
        --waypoint approach \\
        --out configs/courses/through_left_gate_variants/ \\
        --samples-per-mode 5 \\
        --magnitude-range 0.2 0.5 \\
        --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from falsify.planning import (
    load_course, save_course, sample_variants,
)


VALID_MODES = ("center", "up", "down", "left", "right")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--course", required=True, type=Path,
                   help="Base course YAML.")
    p.add_argument("--waypoint", required=True, type=str,
                   help="Name of the waypoint to perturb (e.g. 'approach').")
    p.add_argument("--out", required=True, type=Path,
                   help="Output directory for variant YAMLs.")
    p.add_argument("--modes", nargs="*", default=list(VALID_MODES),
                   choices=VALID_MODES,
                   help=f"Subset of directions to generate. Default: all of {VALID_MODES}.")
    p.add_argument("--samples-per-mode", type=int, default=1,
                   help="Number of variants per direction (default 1).")
    p.add_argument("--magnitude-range", nargs=2, type=float,
                   default=[0.2, 0.5], metavar=("LO", "HI"),
                   help="Uniform sampling range for perturbation magnitudes (m). "
                        "Default: 0.2 0.5. ``center`` always uses 0.0.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    base = load_course(args.course)
    variants = sample_variants(
        base, args.waypoint,
        modes=tuple(args.modes),
        magnitude_range_m=tuple(args.magnitude_range),
        n_per_mode=args.samples_per_mode,
        seed=args.seed,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[base] {base.name}  perturbing waypoint={args.waypoint!r}")
    print(f"[out ] {args.out}  ({len(variants)} variants)")
    for v in variants:
        out_path = args.out / f"{v.course.name}.yaml"
        save_course(v.course, out_path)
        # Find the perturbed waypoint's new p for the log line.
        new_p = next(w.p.tolist() for w in v.course.waypoints if w.name == args.waypoint)
        print(f"  [{v.direction:<6}] mag={v.magnitude_m:.3f}m  "
              f"{args.waypoint}={new_p}  →  {out_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
