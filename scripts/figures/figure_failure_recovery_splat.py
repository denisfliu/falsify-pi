"""Same plot as ``figure_failure_recovery.py`` but the backdrop is a
photorealistic gsplat render, not a matplotlib point-cloud scatter. The
trajectories are projected to 2-D pixel coordinates using the same
OpenCV-pinhole pose / intrinsics that drove the render, then drawn on
top via a 2-D matplotlib axes.

Requires the working gsplat CUDA path (source ``tools/env.sh`` first).
Falls back gracefully with a clear error if CUDA JIT fails.

Discovery flags mirror ``figure_failure_recovery.py``: ``--trial-dir``
(one trial), ``--run-dir`` (multiple), or ``--scene-key-dir`` (glob the
scene_key parent across all ``run-*`` collection dirs).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _ned_to_mocap(points_ned: np.ndarray, fg) -> np.ndarray:
    from falsify.geometry import Point
    ned = fg.frame("ned")
    return np.stack([fg.convert(Point.of(*p, ned), to="mocap").xyz
                     for p in points_ned], axis=0)


def _mocap_to_ned(point_mocap: np.ndarray, fg) -> np.ndarray:
    from falsify.geometry import Point
    return fg.convert(Point.of(*point_mocap, fg.frame("mocap")), to="ned").xyz


def _look_at_ned(eye_ned: np.ndarray, target_ned: np.ndarray,
                 world_up_ned: np.ndarray) -> np.ndarray:
    """Camera-to-world matrix (4x4) for an OpenCV-style camera (x=right,
    y=down, z=forward-into-image). Same convention as
    ``scripts/figures/render_scene_overview.py``."""
    eye = np.asarray(eye_ned, dtype=np.float64)
    tgt = np.asarray(target_ned, dtype=np.float64)
    up = np.asarray(world_up_ned, dtype=np.float64)
    fwd = tgt - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.column_stack([right, down, fwd])
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = eye
    return M


def _project_to_pixels(points_ned: np.ndarray, M_c2w: np.ndarray,
                       intr: dict) -> tuple[np.ndarray, np.ndarray]:
    """Returns (uv: (N, 2), in_front: (N,) bool). uv is float pixel coords;
    a point with `in_front=False` is behind the camera and should be hidden."""
    R = M_c2w[:3, :3]; t = M_c2w[:3, 3]
    # world → cam: x_c = R.T @ (x_w - t). Columns of R are [right, down, fwd].
    cam = (R.T @ (points_ned - t).T).T
    z = cam[:, 2]
    in_front = z > 1e-3
    u = intr["fx"] * cam[:, 0] / np.where(in_front, z, 1.0) + intr["cx"]
    v = intr["fy"] * cam[:, 1] / np.where(in_front, z, 1.0) + intr["cy"]
    return np.stack([u, v], axis=-1), in_front


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--trial-dir", type=Path)
    src.add_argument("--run-dir", type=Path, nargs="+")
    src.add_argument("--scene-key-dir", type=Path)
    ap.add_argument("--max-trials", type=int, default=60)
    ap.add_argument("--scene", type=Path, default=None,
                    help="Override scene YAML (defaults to the trial's).")
    ap.add_argument("--eye", type=float, nargs=3, default=(-1.5, 0.0, 2.0),
                    help="Camera eye in MOCAP.")
    ap.add_argument("--focal", type=float, nargs=3, default=(1.5, 0.0, 1.0),
                    help="Look-at target in MOCAP.")
    ap.add_argument("--fov-deg", type=float, default=110.0)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=1200)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dpi", type=int, default=200,
                    help="matplotlib DPI for the overlay save.")
    ap.add_argument("--rollout-alpha", type=float, default=0.6)
    ap.add_argument("--recovery-alpha", type=float, default=0.75)
    ap.add_argument("--rollout-color", default="#e74c3c")
    ap.add_argument("--recovery-color", default="#2ecc71")
    ap.add_argument("--rollout-lw", type=float, default=1.4)
    ap.add_argument("--recovery-lw", type=float, default=1.6)
    ap.add_argument("--marker-scale", type=float, default=1.0)
    ap.add_argument("--no-legend", action="store_true")
    args = ap.parse_args()

    from falsify.io import load_yaml, build_frame_graph

    # Resolve trial list (same logic as figure_failure_recovery.py).
    trials: list[Path] = []
    if args.trial_dir is not None:
        t = args.trial_dir if args.trial_dir.is_absolute() \
            else (REPO_ROOT / args.trial_dir).resolve()
        trials = [t]
    else:
        run_dirs: list[Path] = []
        if args.run_dir is not None:
            run_dirs = [d if d.is_absolute() else (REPO_ROOT / d).resolve()
                        for d in args.run_dir]
        else:
            parent = args.scene_key_dir if args.scene_key_dir.is_absolute() \
                else (REPO_ROOT / args.scene_key_dir).resolve()
            run_dirs = sorted(d for d in parent.glob("run-*") if d.is_dir())
            print(f"[scene-key-dir] {parent.relative_to(REPO_ROOT)} → "
                  f"{len(run_dirs)} run dir(s)")
        for rd in run_dirs:
            for p in sorted(rd.glob("*/trial_*/episode_summary.json")):
                td = p.parent
                if (td / "rollout_states.npz").exists() and \
                   (td / "recovery_trajectory.npz").exists():
                    trials.append(td)
        if len(trials) > args.max_trials:
            print(f"[run] {len(trials)} trials found; capping to {args.max_trials}")
            trials = trials[: args.max_trials]
        else:
            print(f"[run] {len(trials)} trial(s)")

    if not trials:
        raise SystemExit("no trials found")

    summary0 = json.loads((trials[0] / "episode_summary.json").read_text())
    scene_path = args.scene or Path(summary0["scene"])
    scene_yaml = scene_path if scene_path.is_absolute() \
        else (REPO_ROOT / scene_path).resolve()
    scene_cfg = load_yaml(scene_yaml)
    fg = build_frame_graph(scene_cfg, base_path=scene_yaml.parent)
    print(f"[scene] {scene_yaml.relative_to(REPO_ROOT)}")

    # --- gsplat backdrop ---
    from falsify.geometry import Pose
    from falsify.sim.renderer import GSplatRenderer

    eye_mocap = np.asarray(args.eye, dtype=np.float64)
    focal_mocap = np.asarray(args.focal, dtype=np.float64)
    eye_ned = _mocap_to_ned(eye_mocap, fg)
    focal_ned = _mocap_to_ned(focal_mocap, fg)
    up_world_ned = np.array([0.0, 0.0, -1.0])   # MOCAP +z = NED -z
    M_c2w = _look_at_ned(eye_ned, focal_ned, up_world_ned)

    fov = np.deg2rad(args.fov_deg)
    fx = args.width / (2.0 * np.tan(fov / 2.0)); fy = fx
    intr = dict(width=args.width, height=args.height,
                fx=fx, fy=fy, cx=args.width / 2.0, cy=args.height / 2.0)

    print(f"[render] gsplat backdrop {args.width}x{args.height} "
          f"fov={args.fov_deg}° eye_mocap={eye_mocap.tolist()} "
          f"focal_mocap={focal_mocap.tolist()}")
    renderer = GSplatRenderer.from_scene_cfg(scene_cfg, scene_dir=scene_yaml.parent)
    pose = Pose.from_matrix(M_c2w, fg.frame("ned"))
    rgb, _ = renderer.render(pose, intr)
    print(f"[render] done, mean luminance={rgb.mean():.1f}")

    # --- trajectories ---
    pairs: list[dict] = []
    for td in trials:
        s = json.loads((td / "episode_summary.json").read_text())
        roll = np.load(td / "rollout_states.npz", allow_pickle=True)
        rec = np.load(td / "recovery_trajectory.npz", allow_pickle=True)
        pos_ned = roll["positions_ned"]
        rec_ned = rec["positions_ned"]
        failure_step = int(roll["failure_step"])
        seed_step_val = ((s.get("recovery") or {}).get("seed_step")) if s.get("recovery") else None
        if seed_step_val is None:
            # Match rec[0] back to rollout to recover the seed step.
            d = np.linalg.norm(pos_ned - rec_ned[0], axis=1)
            seed_step_val = int(np.argmin(d))
        pairs.append({
            "trial_dir": td,
            "failure_type": (s.get("failure") or {}).get("type"),
            "rollout_ned": pos_ned,
            "recovery_ned": rec_ned,
            "failure_step": failure_step,
            "seed_step": int(seed_step_val),
            "start_ned": pos_ned[0],
            "failure_ned": pos_ned[failure_step],
            "seed_ned": pos_ned[int(seed_step_val)],
            "goal_ned": np.asarray(s["goal_ned"], dtype=np.float64),
        })

    # --- composite figure ---
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    fig = plt.figure(figsize=(args.width / args.dpi, args.height / args.dpi),
                     dpi=args.dpi)
    ax = fig.add_subplot(111)
    ax.imshow(rgb, interpolation="nearest")
    ax.set_xlim(0, args.width); ax.set_ylim(args.height, 0)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    is_overlay = len(pairs) > 1
    _used: set[str] = set()

    def _label(name: str) -> str | None:
        if name in _used:
            return None
        _used.add(name); return name

    def _draw_line(points_ned: np.ndarray, *, color, lw, alpha, label):
        uv, in_front = _project_to_pixels(points_ned, M_c2w, intr)
        if not in_front.any():
            return
        # Break the polyline at out-of-front points so we don't draw long
        # bogus segments across the camera plane.
        segs: list[np.ndarray] = []
        cur: list[np.ndarray] = []
        for p, ok in zip(uv, in_front):
            if ok:
                cur.append(p)
            else:
                if len(cur) >= 2:
                    segs.append(np.asarray(cur))
                cur = []
        if len(cur) >= 2:
            segs.append(np.asarray(cur))
        first = True
        for s in segs:
            ax.plot(s[:, 0], s[:, 1], color=color, lw=lw, alpha=alpha,
                    label=label if first else None)
            first = False

    def _draw_marker(point_ned: np.ndarray, *, color, marker, s, label, zorder=10):
        uv, in_front = _project_to_pixels(point_ned[None, :], M_c2w, intr)
        if not in_front[0]:
            return
        ax.scatter(uv[0, 0], uv[0, 1], color=color, marker=marker,
                   s=s * args.marker_scale,
                   edgecolors="white", linewidths=1.0, alpha=0.95,
                   zorder=zorder, label=label)

    # Sizes scale down a bit in overlay mode to stay readable.
    if is_overlay:
        sz = dict(start=22, seed=30, fail=42, goal=70)
        roll_lw = max(0.7, args.rollout_lw * 0.7)
        rec_lw = max(0.9, args.recovery_lw * 0.7)
        roll_a = args.rollout_alpha * 0.7
        rec_a = args.recovery_alpha * 0.8
    else:
        sz = dict(start=50, seed=80, fail=110, goal=130)
        roll_lw = args.rollout_lw
        rec_lw = args.recovery_lw
        roll_a = args.rollout_alpha
        rec_a = args.recovery_alpha

    for pp in pairs:
        _draw_line(pp["rollout_ned"],
                   color=args.rollout_color, lw=roll_lw, alpha=roll_a,
                   label=_label("VLA rollout"))
        _draw_line(pp["recovery_ned"],
                   color=args.recovery_color, lw=rec_lw, alpha=rec_a,
                   label=_label("MPC replan"))
        _draw_marker(pp["start_ned"], color="#1f77b4", marker="o",
                     s=sz["start"], label=_label("start"))
        _draw_marker(pp["seed_ned"], color="#ff8c00", marker="D",
                     s=sz["seed"], label=_label("replan seed"))
        _draw_marker(pp["failure_ned"], color="#d62728", marker="X",
                     s=sz["fail"], zorder=11,
                     label=_label("failure"))

    _draw_marker(pairs[0]["goal_ned"], color="#2ecc71", marker="*",
                 s=sz["goal"], label=_label("goal"))

    if not args.no_legend:
        leg = ax.legend(loc="upper left", fontsize=9, framealpha=0.85,
                        facecolor="white", edgecolor="#cccccc")
        leg.set_zorder(100)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0,
                facecolor="white")
    plt.close(fig)
    print(f"[done] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
