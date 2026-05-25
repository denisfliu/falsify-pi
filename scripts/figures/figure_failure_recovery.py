"""Render the "VLA fails, recovery MPC replans" panel.

Composes one PNG matching the cloud-render aesthetic of
``scripts/figures/render_scene_pointclouds.py``:

  - faint scene cloud (z-cull above 1.5 + painted-box gate cleanup)
  - nominal course (dashed gray)
  - VLA rollout (viridis gradient transitioning to bright red near the
    failure step)
  - last-safe seed (orange diamond)
  - failure point (red X)
  - recovery MPC trajectory (solid green)
  - goal (green circle)

Inputs are an eval-campaign trial directory plus the scene cache that
``render_scene_pointclouds.py`` produced. NED-frame artifacts (rollout +
recovery NPZs) get converted to MOCAP via the scene's FrameGraph.

Usage::

    PYTHONPATH=src:external/FiGS/src:external/splatnav \\
        .venv/bin/python scripts/figures/figure_failure_recovery.py \\
        --trial-dir runs/eval_campaigns/legacy/pi07_nh_real_center_pure_center_no_rtc_20260520_150831/center_gate_from_right/trial_001 \\
        --out runs/figures/figure_panels/D_failure_recovery.png
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _cache_path(scene_yaml: Path) -> Path:
    return REPO_ROOT / "runs" / "cache" / f"scene_{scene_yaml.stem}.npz"


def _ned_to_mocap(points_ned: np.ndarray, fg) -> np.ndarray:
    from falsify.geometry import Point
    out = np.empty_like(points_ned)
    ned = fg.frame("ned")
    for i, p in enumerate(points_ned):
        out[i] = fg.convert(Point.of(*p, ned), to="mocap").xyz
    return out


def _filter_cloud(means_mocap: np.ndarray, rgb: np.ndarray, scene_cfg: dict,
                  *, painted_payload: dict | None, z_cull: float,
                  crop: tuple[np.ndarray, np.ndarray] | None,
                  max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Mirror the cloud-render filter chain so the backdrop matches the
    other panels: crop AABB → z-cull (with gate-AABB protection) →
    painted-box exclusion → subsample. Local copy so we don't pull
    matplotlib through render_scene_pointclouds."""
    mask = np.ones(means_mocap.shape[0], dtype=bool)
    if crop is not None:
        cmn, cmx = crop
        mask &= ((means_mocap >= cmn) & (means_mocap <= cmx)).all(axis=1)

    # Gate AABBs (single or compositional).
    gate_aabbs: list[tuple[np.ndarray, np.ndarray]] = []
    blocks: list[dict] = []
    if isinstance(scene_cfg.get("gate_region"), dict):
        blocks.append(scene_cfg["gate_region"])
    if isinstance(scene_cfg.get("gate_regions"), list):
        blocks.extend(scene_cfg["gate_regions"])
    for b in blocks:
        if b.get("aabb_frame", "mocap") == "mocap":
            gate_aabbs.append((np.asarray(b["aabb_min"], dtype=np.float64),
                               np.asarray(b["aabb_max"], dtype=np.float64)))
    above = means_mocap[:, 2] > z_cull
    protected = np.zeros(means_mocap.shape[0], dtype=bool)
    for mn, mx in gate_aabbs:
        protected |= ((means_mocap >= mn) & (means_mocap <= mx)).all(axis=1)
    mask &= ~(above & ~protected)

    if painted_payload is not None:
        boxes = painted_payload["boxes"]
        sg = painted_payload["source_gate"]
        src_anchor = np.asarray(sg["anchor"], dtype=np.float64)
        src_normal = np.asarray(sg["normal"], dtype=np.float64)
        src_angle = float(np.arctan2(src_normal[1], src_normal[0]))
        # Per-gate inverse rotation into source-gate-local AABB test.
        target_gates = [b for b in blocks
                        if b.get("aabb_frame", "mocap") == "mocap"
                        and "anchor" in b and "normal" in b]
        for gate in target_gates:
            tgt_anchor = np.asarray(gate["anchor"], dtype=np.float64)
            tgt_normal = np.asarray(gate["normal"], dtype=np.float64)
            theta = float(np.arctan2(tgt_normal[1], tgt_normal[0])) - src_angle
            c, s = np.cos(-theta), np.sin(-theta)
            Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            cand_idx = np.where(mask)[0]
            if cand_idx.size == 0:
                break
            local = (Rz @ (means_mocap[cand_idx] - tgt_anchor).T).T + src_anchor
            drop = np.zeros(cand_idx.size, dtype=bool)
            for b in boxes:
                bmn = np.asarray(b["min"], dtype=np.float64)
                bmx = np.asarray(b["max"], dtype=np.float64)
                drop |= ((local >= bmn) & (local <= bmx)).all(axis=1)
            mask[cand_idx[drop]] = False

    pts = means_mocap[mask]
    cols = rgb[mask]
    if pts.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]; cols = cols[idx]
    return pts, cols


def _spline_course(waypoints_mocap: np.ndarray, samples: int = 200) -> np.ndarray:
    """Quick cubic-spline interpolation through course waypoints for a
    smooth dashed line. Falls back to a polyline if scipy is missing."""
    try:
        from scipy.interpolate import CubicSpline
        t = np.linspace(0.0, 1.0, waypoints_mocap.shape[0])
        cs = CubicSpline(t, waypoints_mocap, axis=0, bc_type="natural")
        ts = np.linspace(0.0, 1.0, samples)
        return cs(ts)
    except Exception:
        return waypoints_mocap


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--trial-dir", type=Path)
    src.add_argument("--run-dir", type=Path, nargs="+",
                     help="One or more recovery-collection run dirs. The "
                          "script auto-discovers every trial under each "
                          "with both rollout_states.npz and "
                          "recovery_trajectory.npz and overlays them all.")
    src.add_argument("--scene-key-dir", type=Path,
                     help="A scene_key parent (e.g. "
                          "runs/recovery_collection/<policy>/center_from_right) "
                          "— glob over its run-* children.")
    ap.add_argument("--max-trials", type=int, default=50,
                    help="Cap on number of trial pairs overlaid in --run-dir mode.")
    ap.add_argument("--course", type=Path, default=None,
                    help="Course YAML for the dashed nominal trajectory. "
                         "Default: configs/courses/through_<scene_key>.yaml.")
    ap.add_argument("--scene", type=Path, default=None,
                    help="Scene YAML (defaults to the one in episode_summary).")
    ap.add_argument("--painted-boxes-json", type=Path,
                    default=Path("runs/figures/exclude_aabbs_center_gate.json"))
    ap.add_argument("--crop", default="-1.6,-1.5,-0.1:3.5,1.5,2.3")
    ap.add_argument("--z-cull", type=float, default=1.5)
    ap.add_argument("--max-points", type=int, default=200_000)
    ap.add_argument("--cloud-alpha", type=float, default=0.18)
    ap.add_argument("--cloud-point-size", type=float, default=0.35)
    ap.add_argument("--fov-deg", type=float, default=110.0)
    ap.add_argument("--eye", type=float, nargs=3, default=(-1.5, 0.0, 2.0))
    ap.add_argument("--focal", type=float, nargs=3, default=(1.5, 0.0, 1.0))
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--figsize", type=float, nargs=2, default=(10.0, 6.0))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from falsify.io import load_yaml, build_frame_graph

    # Resolve trial list.
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
            print(f"[run] {len(trials)} failed-with-recovery trials found; "
                  f"capping to first {args.max_trials}")
            trials = trials[: args.max_trials]
        else:
            print(f"[run] {len(trials)} failed-with-recovery trials")

    if not trials:
        raise SystemExit("no failed trials with recovery_trajectory.npz found")

    # First trial's summary pins scene + course; in --run-dir mode we
    # validate that every other trial agrees on scene_key (the collection
    # script is single-scene per run anyway).
    summary = json.loads((trials[0] / "episode_summary.json").read_text())
    scene_path = args.scene or Path(summary["scene"])
    scene_yaml = scene_path if scene_path.is_absolute() \
        else (REPO_ROOT / scene_path).resolve()
    scene_cfg = load_yaml(scene_yaml)
    fg = build_frame_graph(scene_cfg, base_path=scene_yaml.parent)
    if args.trial_dir is None:
        scene_keys = set()
        for t in trials:
            s = json.loads((t / "episode_summary.json").read_text())
            scene_keys.add(s.get("scene_key"))
        if len(scene_keys) > 1:
            print(f"[warn] mixed scene_keys: {scene_keys}")
    print(f"[scene] {scene_yaml.relative_to(REPO_ROOT)}")

    # --- backdrop cloud ---
    cache = _cache_path(scene_yaml)
    if not cache.exists():
        raise SystemExit(
            f"missing scene cache at {cache}. Run "
            "scripts/figures/render_scene_pointclouds.py once on this scene first."
        )
    data = np.load(cache)
    means_mocap = data["means_mocap"]; rgb = data["rgb"].astype(np.float32)
    print(f"[cache] {means_mocap.shape[0]:,} Gaussians from {cache.name}")

    crop = None
    if args.crop:
        lo, hi = args.crop.split(":")
        crop = (np.array([float(v) for v in lo.split(",")], dtype=np.float64),
                np.array([float(v) for v in hi.split(",")], dtype=np.float64))

    painted_payload = None
    if args.painted_boxes_json is not None and args.painted_boxes_json.exists():
        p = json.loads(args.painted_boxes_json.read_text())
        if isinstance(p, dict) and "boxes" in p and "source_gate" in p:
            painted_payload = p
            print(f"[painted] {len(p['boxes'])} boxes "
                  f"src={p['source_gate']['name']}@{p['source_gate']['anchor']}")

    pts, cols = _filter_cloud(
        means_mocap, rgb, scene_cfg,
        painted_payload=painted_payload, z_cull=args.z_cull,
        crop=crop, max_points=args.max_points, seed=args.seed,
    )
    print(f"[cloud] rendering {pts.shape[0]:,} points")

    # --- trajectories: gather all (rollout, recovery) pairs in mocap ---
    pairs: list[dict] = []
    for td in trials:
        s = json.loads((td / "episode_summary.json").read_text())
        roll = np.load(td / "rollout_states.npz", allow_pickle=True)
        rec = np.load(td / "recovery_trajectory.npz", allow_pickle=True)
        pos_ned = roll["positions_ned"]
        rec_ned = rec["positions_ned"]
        failure_step = int(roll["failure_step"])
        pos_mocap = _ned_to_mocap(pos_ned, fg)
        rec_mocap = _ned_to_mocap(rec_ned, fg)
        # Seed step: prefer summary, else derive by matching the recovery's
        # first state to the rollout. Recoveries are seeded to within
        # numerical noise so a min-distance match is unambiguous.
        seed_step_val = ((s.get("recovery") or {}).get("seed_step")) if s.get("recovery") else None
        if seed_step_val is None:
            d = np.linalg.norm(pos_mocap - rec_mocap[0], axis=1)
            seed_step_val = int(np.argmin(d))
        pairs.append({
            "trial_dir": td,
            "failure_type": (s.get("failure") or {}).get("type"),
            "rollout_mocap": pos_mocap,
            "recovery_mocap": rec_mocap,
            "failure_step": failure_step,
            "seed_step": int(seed_step_val),
            "start_mocap": pos_mocap[0],
            "failure_mocap": pos_mocap[failure_step],
            "seed_mocap": pos_mocap[int(seed_step_val)],
            "goal_mocap": _ned_to_mocap(np.asarray(s["goal_ned"])[None, :], fg)[0],
        })
    print(f"[trajectories] {len(pairs)} (rollout, recovery) pair(s)")
    # Convenience aliases for the camera bbox / legend below.
    pos_mocap = pairs[0]["rollout_mocap"]
    rec_mocap = pairs[0]["recovery_mocap"]
    failure_mocap = pairs[0]["failure_mocap"]
    seed_mocap = pairs[0]["seed_mocap"]
    start_mocap = pairs[0]["start_mocap"]
    goal_mocap = pairs[0]["goal_mocap"]
    failure_step = pairs[0]["failure_step"]
    seed_step = pairs[0]["seed_step"]

    # --- course ---
    course_yaml = args.course
    if course_yaml is None:
        scene_key = summary["scene_key"]
        course_yaml = REPO_ROOT / "configs" / "courses" / f"through_{scene_key}.yaml"
    else:
        course_yaml = course_yaml if course_yaml.is_absolute() \
            else (REPO_ROOT / course_yaml).resolve()
    course_pts = None
    if course_yaml.exists():
        course = load_yaml(course_yaml)
        wps = np.array([w["p"] for w in course["waypoints"]], dtype=np.float64)
        # course frame should be mocap on the shipped scenes; trust the yaml.
        if course.get("frame", "mocap") != "mocap":
            raise NotImplementedError(
                f"course frame {course['frame']!r} ≠ 'mocap' not handled here")
        course_pts = _spline_course(wps, samples=300)
        print(f"[course] {course_yaml.name} ({wps.shape[0]} waypoints)")
    else:
        print(f"[course] not found: {course_yaml}; skipping nominal overlay")

    # --- figure ---
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    # Tight axis box on the union of cloud + every trajectory's extent.
    chunks = [pts]
    for pp in pairs:
        chunks.append(pp["rollout_mocap"])
        chunks.append(pp["recovery_mocap"])
        chunks.append(pp["goal_mocap"][None, :])
    all_pts = np.vstack(chunks)
    mn = all_pts.min(axis=0); mx = all_pts.max(axis=0)
    pad_ax = 0.02 * (mx - mn).max()
    lo = mn - pad_ax; hi = mx + pad_ax
    extents = hi - lo

    fig = plt.figure(figsize=tuple(args.figsize), dpi=args.dpi)
    ax = fig.add_subplot(111, projection="3d")

    # Cloud backdrop.
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols,
               s=args.cloud_point_size, marker=".", linewidths=0,
               edgecolors="none", alpha=args.cloud_alpha)

    # Nominal course.
    if course_pts is not None:
        ax.plot(course_pts[:, 0], course_pts[:, 1], course_pts[:, 2],
                color="#555555", linestyle="--", linewidth=1.6, alpha=0.85,
                label="nominal course")

    # Plot every pair. In single-trial mode this draws one rich line each;
    # in run-dir mode we draw N pairs with shared dim colors so the eye
    # picks up the *pattern* of failures + recoveries rather than any one
    # trajectory. Legend entries are emitted once via `_used` gating.
    is_overlay = len(pairs) > 1
    if is_overlay:
        roll_lw, rec_lw, roll_alpha, rec_alpha = 1.2, 1.4, 0.55, 0.65
        roll_color = (0.86, 0.27, 0.27, 1.0)   # dim red
        rec_color  = (0.18, 0.55, 0.27, 1.0)   # dim green
        marker_alpha = 0.85
    else:
        roll_lw, rec_lw, roll_alpha, rec_alpha = 2.2, 2.6, 1.0, 0.95
        marker_alpha = 1.0

    base_cmap = plt.get_cmap("viridis")
    _used: set[str] = set()

    def _label(name: str) -> str | None:
        if name in _used:
            return None
        _used.add(name); return name

    for pp in pairs:
        pos = pp["rollout_mocap"]; recov = pp["recovery_mocap"]
        seg = np.stack([pos[:-1], pos[1:]], axis=1)
        if is_overlay:
            seg_colors = np.tile(np.array(roll_color), (len(seg), 1))
            seg_colors[:, 3] = roll_alpha
        else:
            n = pos.shape[0]
            fade_window = max(1, min(20, n // 4))
            seg_colors = base_cmap(np.linspace(0.05, 0.7, len(seg)))
            ramp = np.linspace(0.0, 1.0, fade_window)
            red = np.array([0.85, 0.10, 0.10, 1.0])
            for j, a in enumerate(ramp):
                idx = len(seg) - fade_window + j
                if idx >= 0:
                    seg_colors[idx] = (1 - a) * seg_colors[idx] + a * red
        lc = Line3DCollection(seg, colors=seg_colors, linewidth=roll_lw,
                              label=_label("VLA rollout"))
        ax.add_collection3d(lc)

        ax.plot(recov[:, 0], recov[:, 1], recov[:, 2],
                color=rec_color if is_overlay else "#2ca02c",
                linewidth=rec_lw, alpha=rec_alpha,
                label=_label(f"replan ({summary['recovery']['planner']} MPC)"
                             if not is_overlay else "replan (MPC)"))

        # Markers.
        ax.scatter(*pp["start_mocap"], color="#1f77b4",
                   s=40 if is_overlay else 70,
                   marker="o", edgecolors="white", linewidths=1.0,
                   zorder=10, alpha=marker_alpha, label=_label("start"))
        ax.scatter(*pp["seed_mocap"], color="#ff8c00",
                   s=55 if is_overlay else 110,
                   marker="D", edgecolors="white", linewidths=1.0,
                   zorder=10, alpha=marker_alpha,
                   label=_label("replan seed"
                                + (f" (step {pp['seed_step']})" if not is_overlay else "")))
        ax.scatter(*pp["failure_mocap"], color="#d62728",
                   s=80 if is_overlay else 160,
                   marker="X", edgecolors="white", linewidths=1.1,
                   zorder=11, alpha=marker_alpha,
                   label=_label("failure"
                                + (f" ({pp['failure_type']})" if not is_overlay else "")))

    # Single goal marker drawn from pair 0.
    ax.scatter(*goal_mocap, color="#2ca02c",
               s=130 if not is_overlay else 110, marker="*",
               edgecolors="white", linewidths=1.2, zorder=10,
               label="goal")

    # Camera.
    eye = np.asarray(args.eye); focal = np.asarray(args.focal)
    fwd = focal - eye
    azim_deg = float(np.rad2deg(np.arctan2(fwd[1], fwd[0])) + 180.0)
    horiz = float(np.hypot(fwd[0], fwd[1]))
    elev_deg = float(np.rad2deg(np.arctan2(-fwd[2], horiz)))
    ax.set_proj_type("persp", focal_length=1.0 / np.tan(
        np.deg2rad(args.fov_deg) / 2.0))
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(tuple(extents.tolist()))
    ax.view_init(elev=elev_deg, azim=azim_deg)
    ax.set_axis_off()
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((1.0, 1.0, 1.0, 0.0))
        pane.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.grid(False)

    leg = ax.legend(loc="upper left", fontsize=8, framealpha=0.85,
                    facecolor="white", edgecolor="#cccccc")
    leg.set_zorder(100)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0,
                facecolor="white")
    plt.close(fig)

    # PIL trim.
    from PIL import Image, ImageChops
    im = Image.open(args.out).convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    diff = ImageChops.difference(im, bg)
    bbox = diff.point(lambda x: 0 if x < 8 else x).getbbox()
    if bbox is not None and bbox != (0, 0, im.size[0], im.size[1]):
        im.crop(bbox).save(args.out)
    print(f"[done] wrote {args.out} ({im.size if bbox is None else (bbox[2]-bbox[0], bbox[3]-bbox[1])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
