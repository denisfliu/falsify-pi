"""Generate stochastic variant course YAMLs from one base course.

Per variant the sampler:
  1. With probability ``corrective_perturbations.probability`` applies a
     discrete corrective shift (mode uniformly drawn from
     ``corrective_perturbations.modes``, magnitude uniformly from
     ``corrective_perturbations.magnitude_range_m``) to
     ``corrective_perturbations.target_waypoint``.
  2. For every waypoint not in ``trajectory_perturbations.exclude_waypoints``,
     adds a uniform-in-ball displacement of radius
     ``trajectory_perturbations.radius_m``.

Total variant count comes from ``trajectory_perturbations.samples``. Every
variant is unique by construction (the trajectory noise is i.i.d.).

  perturb_course → directory of course YAMLs (one per variant)
       │
       ▼
  plan_trajectory → trajectory.npz
       │
       ▼
  export_training_data → episode_NNNNNN.parquet

Example::

    PYTHONPATH=src .venv/bin/python -m falsify.cli.perturb_course \\
        --course configs/courses/through_center_gate_from_left.yaml \\
        --out configs/courses/through_center_gate_from_left_check/

CLI flags override the corresponding YAML fields when given.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from falsify.planning import (
    CorrectivePerturbation,
    TrajectoryPerturbation,
    load_course,
    sample_stochastic_variants,
    save_course,
)


VALID_MODES = ("up", "down", "left", "right")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--course", required=True, type=Path,
                   help="Base course YAML.")
    p.add_argument("--out", required=True, type=Path,
                   help="Output directory for variant YAMLs.")
    # Corrective overrides.
    p.add_argument("--waypoint", type=str, default=None,
                   help="Corrective target waypoint name "
                        "(overrides corrective_perturbations.target_waypoint).")
    p.add_argument("--modes", nargs="*", default=None, choices=VALID_MODES,
                   help="Subset of corrective directions "
                        "(overrides corrective_perturbations.modes).")
    p.add_argument("--magnitude-range", nargs=2, type=float, default=None,
                   metavar=("LO", "HI"),
                   help="Corrective magnitude sampling range, meters "
                        "(overrides corrective_perturbations.magnitude_range_m).")
    p.add_argument("--probability", type=float, default=None,
                   help="Per-sample probability of applying a corrective shift "
                        "(overrides corrective_perturbations.probability).")
    # Trajectory overrides.
    p.add_argument("--samples", type=int, default=None,
                   help="Total variants to generate "
                        "(overrides trajectory_perturbations.samples).")
    p.add_argument("--radius", type=float, default=None,
                   help="Per-waypoint spherical noise radius, meters "
                        "(overrides trajectory_perturbations.radius_m).")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed (overrides trajectory_perturbations.seed).")
    args = p.parse_args(argv)

    base = load_course(args.course)
    cp = base.corrective_perturbations
    tp = base.trajectory_perturbations
    if tp is None:
        raise SystemExit(
            "Course YAML is missing a ``trajectory_perturbations`` block, "
            "which the stochastic sampler uses to set total sample count and noise radius."
        )

    # Apply CLI overrides on top of the YAML blocks.
    if cp is not None:
        cp = replace(
            cp,
            target_waypoint=args.waypoint if args.waypoint is not None else cp.target_waypoint,
            modes=tuple(args.modes) if args.modes is not None else cp.modes,
            magnitude_range_m=(tuple(args.magnitude_range)
                               if args.magnitude_range is not None
                               else cp.magnitude_range_m),
            probability=args.probability if args.probability is not None else cp.probability,
        )
    elif args.waypoint is not None or args.modes is not None or \
         args.magnitude_range is not None or args.probability is not None:
        # User specified corrective flags but no YAML block — build one.
        cp = CorrectivePerturbation(
            target_waypoint=args.waypoint or "approach",
            modes=tuple(args.modes) if args.modes is not None else VALID_MODES,
            magnitude_range_m=(tuple(args.magnitude_range)
                               if args.magnitude_range is not None else (0.1, 0.3)),
            probability=(args.probability if args.probability is not None else 1.0),
        )

    tp = replace(
        tp,
        samples=args.samples if args.samples is not None else tp.samples,
        radius_m=args.radius if args.radius is not None else tp.radius_m,
        seed=args.seed if args.seed is not None else tp.seed,
    )

    variants = sample_stochastic_variants(base, corrective=cp, trajectory=tp)

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[base] {base.name}")
    if cp is not None:
        print(f"[cor ] target={cp.target_waypoint!r}  modes={list(cp.modes)}  "
              f"mag_range={list(cp.magnitude_range_m)}m  p={cp.probability}")
    else:
        print(f"[cor ] no corrective block — every variant is baseline+noise")
    print(f"[trj ] radius={tp.radius_m}m  samples={tp.samples}  "
          f"exclude={list(tp.exclude_waypoints)}  seed={tp.seed}")
    print(f"[out ] {args.out}  ({len(variants)} variants)")

    counts: dict[str, int] = {}
    for v in variants:
        # Variants are single instances — strip both perturbation blocks so a
        # variant YAML doesn't carry the generator config for its parent.
        course_to_save = replace(
            v.course,
            corrective_perturbations=None,
            trajectory_perturbations=None,
        )
        out_path = args.out / f"{v.course.name}.yaml"
        save_course(course_to_save, out_path)
        counts[v.direction] = counts.get(v.direction, 0) + 1
    for mode, count in sorted(counts.items()):
        print(f"  [{mode:<6}] {count} variant(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
