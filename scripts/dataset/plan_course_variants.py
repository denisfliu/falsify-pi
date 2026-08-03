"""Sample a Course's embedded perturbation recipe into N planned NPZs.

A corrective-maneuver course (e.g. ``through_center_gate_from_left.yaml``)
carries two embedded blocks:

  - ``corrective_perturbations`` — per-sample Bernoulli shift of one
    waypoint (up/down/left/right) to teach off-axis recovery, and
  - ``trajectory_perturbations`` — ``samples: N`` + a uniform-in-ball
    jitter radius applied to every non-excluded waypoint.

``falsify.planning.sample_stochastic_variants`` turns those blocks into N
unique ``Course`` variants; this driver plans each variant to a canonical
Trajectory NPZ that ``falsify.cli.export_training_data --trajectories-dir``
renders into one training parquet per episode.

This is the front half of the synth-dataset pipeline (the original driver
was removed in a cleanup pass); the back half is the exporter. Reusable
for any course with the embedded blocks — parameterized by ``--course``.

Example::

    PYTHONPATH=src python scripts/dataset/plan_course_variants.py \
        --course configs/courses/through_center_gate_from_left.yaml \
        --scene  configs/scenes/center_gate.yaml \
        --planner mpc \
        --out-dir runs/trajectories/synth_center_from_left
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--course", required=True, type=Path)
    p.add_argument("--scene", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path,
                   help="Directory to write <label>.npz variants into.")
    p.add_argument("--planner", choices=["spline", "mpc"], default="mpc",
                   help="Trajectory planner. mpc = dynamically feasible "
                        "(camera banks; preferred for training data). "
                        "spline = fast geometric (yaw-only attitude).")
    p.add_argument("--prompt", type=str, default="",
                   help="Embedded in each Trajectory NPZ as the task prompt.")
    p.add_argument("--mpc-frame", type=Path, default=None,
                   help="FiGS-schema drone frame JSON (mpc planner only). "
                        "Defaults to configs/frames/figs/carl.json.")
    p.add_argument("--safety", type=Path, default=None,
                   help="Safety YAML for plan validation. Default: "
                        "configs/safety/<scene-stem>.yaml when it exists.")
    p.add_argument("--allow-invalid", action="store_true",
                   help="Write variants that fail validation (diagnostic; "
                        "never feed these to training).")
    p.add_argument("--ignore-collision", action="store_true",
                   help="Validate kinematics (bounds/speed/tilt) only, "
                        "skipping the drone-OBB gate-collision check. Use "
                        "when the gate collision geometry postdates the "
                        "courses and clipping is a known, tolerated artifact.")
    p.add_argument("--limit", type=int, default=None,
                   help="Plan only the first N variants (smoke-testing).")
    args = p.parse_args(argv)

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from falsify.io import build_frame_graph, load_yaml
    from falsify.planning import (
        load_course, plan_mpc, plan_spline, validate_trajectory,
    )
    from falsify.planning.perturbations import sample_stochastic_variants
    from falsify.training import save_trajectory

    scene_cfg = load_yaml(args.scene)
    fg = build_frame_graph(scene_cfg, base_path=args.scene.parent)
    base = load_course(args.course)

    if base.trajectory_perturbations is None:
        raise SystemExit(
            f"{args.course} has no `trajectory_perturbations:` block — "
            "nothing to sample (see through_center_gate_from_left.yaml)"
        )
    variants = sample_stochastic_variants(
        base,
        corrective=base.corrective_perturbations,
        trajectory=base.trajectory_perturbations,
    )
    if args.limit is not None:
        variants = variants[:args.limit]

    safety_path = args.safety
    if safety_path is None:
        candidate = REPO_ROOT / "configs" / "safety" / f"{args.scene.stem}.yaml"
        safety_path = candidate if candidate.is_file() else None
    safety_cfg = load_yaml(safety_path) if safety_path else None
    if safety_cfg is None:
        print(f"[plan] WARN: no safety YAML for {args.scene.stem!r} — "
              "variants NOT validated")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_ok = n_bad = 0
    for i, v in enumerate(variants):
        if args.planner == "spline":
            traj = plan_spline(v.course, fg, prompt=args.prompt)
        else:
            traj = plan_mpc(v.course, fg, prompt=args.prompt,
                            frame_cfg=args.mpc_frame)

        ok = True
        if safety_cfg is not None:
            res = validate_trajectory(
                traj, fg, scene_cfg=scene_cfg, scene_dir=args.scene.parent,
                safety_cfg=safety_cfg, ignore_collision=args.ignore_collision,
            )
            ok = res.ok
            if not ok:
                n_bad += 1
                print(f"  [{i:03d}] {v.label}: INVALID — {res.summary()[:80]}")
                if not args.allow_invalid:
                    continue

        # Zero-padded index prefix so the exporter's filename sort matches
        # generation order regardless of per-mode label counters.
        out = args.out_dir / f"{i:03d}_{v.label}.npz"
        save_trajectory(out, traj)
        n_ok += 1
        print(f"  [{i:03d}] {v.label}: {len(traj)} frames -> {out.name}"
              f"{'' if ok else '  (INVALID, kept)'}")

    print(f"[plan] {args.planner}: wrote {n_ok} NPZ(s) to {args.out_dir} "
          f"({n_bad} failed validation)")
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
