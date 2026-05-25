"""Launch nerfstudio's viewer on a falsify scene with rollout + recovery
trajectories overlaid as viser scene primitives. Use this to find a
camera angle interactively and take a screenshot.

Inputs match ``scripts/figures/figure_failure_recovery_splat.py``:

  --trial-dir / --run-dir / --scene-key-dir

Trajectories are converted NED → NS (the frame nerfstudio's viewer
renders in) via the scene's FrameGraph, so they align with the live
gsplat.

Usage::

    source tools/env.sh
    .venv/bin/python scripts/recovery/viewer_with_recoveries.py \\
        --scene-key-dir runs/recovery_collection/nonhistory_real_synth_31ohxgxv_5000/left_gate \\
        --max-trials 40 --port 7007
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from threading import Lock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _discover_trials(args) -> list[Path]:
    """Just enumerate candidate trials — no filtering, no sampling."""
    if args.trial_dir is not None:
        t = args.trial_dir if args.trial_dir.is_absolute() \
            else (REPO_ROOT / args.trial_dir).resolve()
        return [t]
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
    out: list[Path] = []
    for rd in run_dirs:
        for p in sorted(rd.glob("*/trial_*/episode_summary.json")):
            td = p.parent
            if (td / "rollout_states.npz").exists() and \
               (td / "recovery_trajectory.npz").exists():
                out.append(td)
    return out


def _select_trials(candidates: list[Path], args, fg) -> list[Path]:
    """Apply altitude filter, then sampling, to a pre-discovered list."""
    trials = list(candidates)
    if args.max_altitude_mocap is not None and fg is not None:
        from falsify.geometry import Point
        ned_frame = fg.frame("ned")
        kept: list[Path] = []
        dropped = 0
        for td in trials:
            r = np.load(td / "rollout_states.npz", allow_pickle=True)
            pos_ned = r["positions_ned"]
            # mocap_z = max over rollout. Convert just by transforming
            # the highest-altitude NED point (smallest z_ned since up is -z).
            top_ned = pos_ned[np.argmin(pos_ned[:, 2])]
            top_mocap_z = fg.convert(Point.of(*top_ned, ned_frame), to="mocap").xyz[2]
            if top_mocap_z <= args.max_altitude_mocap:
                kept.append(td)
            else:
                dropped += 1
        if dropped:
            print(f"[altitude-filter] dropped {dropped} trial(s) whose rollout "
                  f"exceeds MOCAP z = {args.max_altitude_mocap}; "
                  f"{len(kept)} remain")
        trials = kept
    if not trials:
        return trials
    if len(trials) > args.max_trials:
        if args.sample == "diverse":
            trials = _farthest_point_sample(trials, args.max_trials, args.fps_seed)
            print(f"[run] {len(trials)} trial(s) via farthest-point sampling "
                  f"(seed={args.fps_seed})")
        else:
            print(f"[run] {len(trials)} → head-slicing to {args.max_trials}")
            trials = trials[: args.max_trials]
    else:
        print(f"[run] keeping all {len(trials)} trial(s) after filter")
    return trials


def _rollout_signature(td: Path, n_samples: int = 16) -> np.ndarray:
    """Coarse fixed-length signature for an individual rollout: 16 evenly-
    spaced positions concatenated. Two trials with the same general path
    shape end up close; trials that fail in different places end up far."""
    r = np.load(td / "rollout_states.npz", allow_pickle=True)
    pos = r["positions_ned"]
    n = pos.shape[0]
    if n == 0:
        return np.zeros(n_samples * 3, dtype=np.float64)
    idx = np.linspace(0, n - 1, n_samples).round().astype(int)
    return pos[idx].ravel().astype(np.float64)


def _farthest_point_sample(trials: list[Path], k: int, seed: int) -> list[Path]:
    """Greedy farthest-point sampling on rollout signatures. Picks an
    initial trial via `seed`, then iteratively appends the trial whose
    minimum L2 distance to the chosen set is largest."""
    feats = np.stack([_rollout_signature(t) for t in trials], axis=0)
    n = feats.shape[0]
    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, n))
    picked = [first]
    min_d = np.linalg.norm(feats - feats[first], axis=1)
    while len(picked) < k:
        nxt = int(np.argmax(min_d))
        picked.append(nxt)
        d_new = np.linalg.norm(feats - feats[nxt], axis=1)
        min_d = np.minimum(min_d, d_new)
        min_d[picked] = -np.inf
    return [trials[i] for i in picked]


def _ned_to_ns(points_ned: np.ndarray, fg) -> np.ndarray:
    """Convert (N, 3) NED points to NS via the FrameGraph."""
    from falsify.geometry import Point
    ned = fg.frame("ned")
    return np.stack([fg.convert(Point.of(*p, ned), to="ns").xyz
                     for p in points_ned], axis=0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--trial-dir", type=Path)
    src.add_argument("--run-dir", type=Path, nargs="+")
    src.add_argument("--scene-key-dir", type=Path)
    ap.add_argument("--max-trials", type=int, default=40)
    ap.add_argument("--sample", choices=("head", "diverse"), default="diverse",
                    help="When --max-trials < trials found: 'head' takes the "
                         "first N (deterministic order); 'diverse' (default) "
                         "runs farthest-point sampling on a coarse trajectory "
                         "signature so the picks span the failure landscape.")
    ap.add_argument("--fps-seed", type=int, default=0,
                    help="Seed for the initial pick in farthest-point sampling.")
    ap.add_argument("--max-altitude-mocap", type=float, default=None,
                    help="Drop trials whose rollout reaches MOCAP z above "
                         "this. Use to skip catastrophic upward-shoot "
                         "rollouts. e.g. 1.8 keeps the gate-altitude band.")
    ap.add_argument("--scene", type=Path, default=None,
                    help="Scene YAML (defaults to trial[0]'s scene).")
    ap.add_argument("--rollout-color", default="#e74c3c")
    ap.add_argument("--recovery-color", default="#2ecc71")
    ap.add_argument("--line-width", type=float, default=2.5)
    ap.add_argument("--marker-radius-ns", type=float, default=0.08,
                    help="Marker sphere radius in viser units. Default 0.08; "
                         "viser shows NS coords ×10, so this is ~1.2 cm in "
                         "MOCAP scale for the gate scenes.")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7007)
    args = ap.parse_args()

    candidates = _discover_trials(args)
    if not candidates:
        raise SystemExit("no trials found")
    print(f"[discover] {len(candidates)} candidate trial(s)")

    from falsify.io import load_yaml, build_frame_graph
    from falsify.sim.scene_edits import apply_edits_to_pipeline, load_scene_edits
    from nerfstudio.utils.eval_utils import eval_setup
    from nerfstudio.viewer.viewer import Viewer as ViewerState, VISER_NERFSTUDIO_SCALE_RATIO
    from nerfstudio.configs.base_config import ViewerConfig
    from nerfstudio.utils import writer
    print(f"[viser] using VISER_NERFSTUDIO_SCALE_RATIO = {VISER_NERFSTUDIO_SCALE_RATIO}")

    summary0 = json.loads((candidates[0] / "episode_summary.json").read_text())
    scene_path = args.scene or Path(summary0["scene"])
    scene_yaml = scene_path if scene_path.is_absolute() \
        else (REPO_ROOT / scene_path).resolve()
    scene_cfg = load_yaml(scene_yaml)
    fg = build_frame_graph(scene_cfg, base_path=scene_yaml.parent)
    print(f"[scene] {scene_yaml.relative_to(REPO_ROOT)}")

    trials = _select_trials(candidates, args, fg)
    if not trials:
        raise SystemExit("no trials survived filtering")

    # --- load gsplat (same path preview_scene_nsviewer.py uses) ---
    def _resolve(rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else (scene_yaml.parent / p).resolve()

    gsplat_yml = _resolve(scene_cfg["gsplat_config_yml"])
    data_cwd = _resolve(scene_cfg["gsplat_data_cwd"]) if "gsplat_data_cwd" in scene_cfg else None
    prev_cwd = os.getcwd()
    if data_cwd is not None:
        os.chdir(data_cwd)
    try:
        config, pipeline, _, step = eval_setup(
            gsplat_yml, eval_num_rays_per_chunk=None, test_mode="test")
    finally:
        os.chdir(prev_cwd)

    edits = load_scene_edits(scene_cfg) or []
    if edits:
        apply_edits_to_pipeline(pipeline, edits, fg)
        print(f"[edits] applied {len(edits)} scene_edit(s)")

    # Stay in data_cwd for the viewer too — the dataparser's `data:` field
    # is relative, and ViewerState.init_scene re-reads it. This matches the
    # pattern in falsify.cli.preview_scene_nsviewer.
    if data_cwd is not None:
        os.chdir(data_cwd)

    # --- start viewer (mirrors run_viewer._start_viewer up to the run loop) ---
    config.vis = "viewer"
    config.viewer = ViewerConfig(websocket_host=args.host,
                                  websocket_port=args.port,
                                  num_rays_per_chunk=config.viewer.num_rays_per_chunk)
    viewer_log_path = config.get_base_dir() / config.viewer.relative_log_filename
    lock = Lock()
    viewer_state = ViewerState(
        config.viewer,
        log_filename=viewer_log_path,
        datapath=pipeline.datamanager.get_datapath(),
        pipeline=pipeline,
        share=False,
        train_lock=lock,
    )
    config.logging.local_writer.enable = False
    writer.setup_local_writer(config.logging, max_iter=config.max_num_iterations,
                              banner_messages=viewer_state.viewer_info)
    viewer_state.init_scene(
        train_dataset=pipeline.datamanager.train_dataset,
        train_state="completed",
        eval_dataset=pipeline.datamanager.eval_dataset,
    )
    viewer_state.update_scene(step=step)
    server = viewer_state.viser_server
    print(f"[viser] http://{args.host}:{args.port}")

    # --- inject trajectories ---
    def _hex_to_rgb(h: str) -> tuple[int, int, int]:
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    roll_rgb = _hex_to_rgb(args.rollout_color)
    rec_rgb = _hex_to_rgb(args.recovery_color)

    folder = server.add_frame("/trajectories", show_axes=False)
    n_drawn = 0
    for i, td in enumerate(trials):
        s = json.loads((td / "episode_summary.json").read_text())
        roll = np.load(td / "rollout_states.npz", allow_pickle=True)
        rec = np.load(td / "recovery_trajectory.npz", allow_pickle=True)
        pos_ned = roll["positions_ned"]; rec_ned = rec["positions_ned"]
        failure_step = int(roll["failure_step"])
        seed_step_val = ((s.get("recovery") or {}).get("seed_step")) if s.get("recovery") else None
        if seed_step_val is None:
            d = np.linalg.norm(pos_ned - rec_ned[0], axis=1)
            seed_step_val = int(np.argmin(d))

        # NS → viser-display: nerfstudio rescales every NS-frame position
        # by VISER_NERFSTUDIO_SCALE_RATIO (=10) before sending to viser
        # (see nerfstudio/viewer/viewer.py:VISER_NERFSTUDIO_SCALE_RATIO).
        # Without the same scale our trajectories land at 1/10 the splat's
        # apparent size and look offset.
        S = VISER_NERFSTUDIO_SCALE_RATIO
        pos_ns = _ned_to_ns(pos_ned, fg) * S
        rec_ns = _ned_to_ns(rec_ned, fg) * S
        start_ns = pos_ns[0]
        seed_ns = pos_ns[int(seed_step_val)]
        fail_ns = pos_ns[failure_step]
        goal_ns = _ned_to_ns(np.asarray(s["goal_ned"])[None, :], fg)[0] * S

        server.add_spline_catmull_rom(
            f"/trajectories/trial_{i:03d}/rollout",
            positions=pos_ns.astype(np.float32),
            color=roll_rgb,
            line_width=args.line_width,
            segments=max(64, pos_ns.shape[0]),
        )
        server.add_spline_catmull_rom(
            f"/trajectories/trial_{i:03d}/recovery",
            positions=rec_ns.astype(np.float32),
            color=rec_rgb,
            line_width=args.line_width,
            segments=max(64, rec_ns.shape[0]),
        )
        # Markers.
        r = args.marker_radius_ns
        server.add_icosphere(
            f"/trajectories/trial_{i:03d}/start", radius=r,
            position=start_ns.astype(np.float32),
            color=(31, 119, 180),
        )
        server.add_icosphere(
            f"/trajectories/trial_{i:03d}/seed", radius=r * 1.3,
            position=seed_ns.astype(np.float32),
            color=(255, 140, 0),
        )
        server.add_icosphere(
            f"/trajectories/trial_{i:03d}/failure", radius=r * 1.5,
            position=fail_ns.astype(np.float32),
            color=(214, 39, 40),
        )
        if i == 0:
            server.add_icosphere(
                "/trajectories/goal", radius=r * 1.8,
                position=goal_ns.astype(np.float32),
                color=(46, 204, 113),
            )
        n_drawn += 1
    print(f"[overlay] drew {n_drawn} trial(s) into /trajectories")
    print("[overlay] rotate/zoom in viser; toggle the /trajectories folder in "
          "the Scene tab to hide overlays for clean screenshots")

    # Block forever (same as run_viewer's loop).
    while True:
        time.sleep(0.05)


if __name__ == "__main__":
    raise SystemExit(main())
