"""Plan a Trajectory NPZ from a Course YAML.

Two planners are wired:

- ``--planner spline`` (default) — cubic spline through the waypoint
  positions, sampled at the course's fps. Honours the course's
  ``yaw_mode``. Fast (~ms); no dynamics.
- ``--planner mpc`` — FiGS ``VehicleRateMPC`` over a
  ``min_time_snap`` reference; dynamically feasible. Same recovery
  backend used by ``falsify.recovery.CoursedMpcPlanner``.

Output is a canonical Trajectory NPZ that
``falsify.cli.export_training_data`` consumes directly — schema is
identical regardless of planner.

Example::

    .venv/bin/python -m falsify.cli.plan_trajectory \\
        --course configs/courses/through_left_gate.yaml \\
        --scene configs/scenes/left_gate.yaml \\
        --planner mpc \\
        --out runs/courses/through_left_gate/trajectory.npz \\
        --prompt "go through the gate and hover over the stuffed animal"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from falsify.io import build_frame_graph, load_yaml
from falsify.planning import load_course, plan_mpc, plan_spline
from falsify.training import save_trajectory


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--course", required=True, type=Path)
    p.add_argument("--scene", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path,
                   help="Path to write the Trajectory NPZ.")
    p.add_argument("--prompt", type=str, default="",
                   help="Embed in the Trajectory NPZ as the task prompt.")
    p.add_argument("--planner", choices=["spline", "mpc"], default="spline",
                   help="Planner backend. spline = cubic-spline through "
                        "positions (geometric). mpc = FiGS VehicleRateMPC "
                        "tracking a min-time-snap reference (dynamically "
                        "feasible; first call pays ~30s acados JIT).")
    p.add_argument("--mpc-frame", type=Path, default=None,
                   help="FiGS-schema drone frame JSON. Defaults to "
                        "configs/frames/figs/carl.json. mpc planner only.")
    args = p.parse_args(argv)

    scene_cfg = load_yaml(args.scene)
    fg = build_frame_graph(scene_cfg, base_path=args.scene.parent)
    course = load_course(args.course)

    if args.planner == "spline":
        traj = plan_spline(course, fg, prompt=args.prompt)
    elif args.planner == "mpc":
        traj = plan_mpc(course, fg, prompt=args.prompt, frame_cfg=args.mpc_frame)
    else:
        raise SystemExit(f"unsupported planner {args.planner!r}")

    out_path = save_trajectory(args.out, traj)
    print(f"[plan] {args.planner}: {len(traj)} frames over {traj.duration_s:.2f}s")
    print(f"[done] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
