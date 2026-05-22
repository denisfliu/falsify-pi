"""Render a recovery-collection run dir into per-episode parquets, with
the per-trial GateRigidPerturbation re-applied to the gsplat before
each render.

This is the corrected counterpart to a naive
``export_training_data --trajectories-dir <recoveries>`` invocation:
``export_training_data`` is unaware of any per-trial perturbation, so
it renders every recovery against the **nominal** gate location even
though the recovery trajectory was planned for a perturbed gate. The
result is camera frames where the gate is 1–3 cm and 1–3° off from
where the drone is flying — silently broken training data.

This script walks the recovery-collection run's per-trial subdirs
(``<run>/<scene_key>/trial_*/``) and, for trials that produced a
recovery, reads the trial's ``trial_card.json`` for its
``gate_perturbation`` block, builds a ``GateRigidPerturbation``,
applies it via ``renderer.apply_dynamic_edits`` (which baseline-
restores per call so perturbations don't compound), then exports the
recovery NPZ via ``TrainingDataExporter.export_episode``.

The output layout matches ``falsify.cli.export_training_data``'s:
``<out>/episode_<NNNNNN>/episode_<NNNNNN>.parquet``. Hand the
parent dir to ``combine_lerobot`` as usual.

Usage:

    PYTHONPATH=src python scripts/render_recoveries_to_dataset.py \\
        --recovery-run-dir runs/recovery_collection/<policy>/<scene>/run-NNN-<ts> \\
        --scene configs/scenes/<scene>.yaml \\
        --frame configs/frames/carl_dual.yaml \\
        --embodiment configs/embodiments/carl_dual_mocap.yaml \\
        --out /tmp/<dataset>/<scene> \\
        --episode-index-base 0 --index-offset 0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve(p: str | Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (REPO_ROOT / pp).resolve()


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recovery-run-dir", required=True, type=Path,
                    help="Path to a run-NNN-<ts>/ dir produced by "
                         "scripts/collect_recovery_trajectories.py.")
    ap.add_argument("--scene", required=True, type=Path)
    ap.add_argument("--frame", required=True, type=Path)
    ap.add_argument("--embodiment", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path,
                    help="Output root. Each episode goes under "
                         "<out>/episode_<NNNNNN>/episode_<NNNNNN>.parquet.")
    ap.add_argument("--episode-index-base", type=int, default=0,
                    help="First episode_index for this scene's parquets. "
                         "Combine_lerobot will renumber on merge — this "
                         "controls the per-scene local numbering.")
    ap.add_argument("--index-offset", type=int, default=0,
                    help="Starting value for the per-frame global `index` "
                         "column. Set to the previous scene's frame total "
                         "if you want a single continuous index across "
                         "scenes before combine_lerobot.")
    ap.add_argument("--task-index", type=int, default=0)
    args = ap.parse_args(argv)

    run_dir = _resolve(args.recovery_run_dir)
    scene_path = _resolve(args.scene)
    frame_path = _resolve(args.frame)
    embodiment_path = _resolve(args.embodiment)

    if not run_dir.is_dir():
        raise SystemExit(f"run dir not found: {run_dir}")

    # Lazy imports — heavy.
    from falsify.io import build_frame_graph, load_yaml
    from falsify.perturbations import GateRigidPerturbation
    from falsify.sim.renderer import GSplatRenderer
    from falsify.training import (
        TrainingDataExporter, load_embodiment, load_trajectory,
    )

    scene_cfg = load_yaml(scene_path)
    frame_cfg = load_yaml(frame_path)
    embodiment = load_embodiment(embodiment_path)
    fg = build_frame_graph(scene_cfg, base_path=scene_path.parent)
    renderer = GSplatRenderer.from_scene_cfg(scene_cfg, scene_dir=scene_path.parent)

    exporter = TrainingDataExporter(
        scene_cfg=scene_cfg, frame_cfg=frame_cfg, frame_graph=fg,
        renderer=renderer.render, embodiment=embodiment,
    )

    # Discover trials that produced a recovery, in collection order
    # (== trial_index order since the collector saves them sequentially).
    candidates: list[tuple[Path, dict]] = []
    for trial_dir in sorted(run_dir.glob("*/trial_*")):
        recov_npz = trial_dir / "recovery_trajectory.npz"
        card_path = trial_dir / "trial_card.json"
        if not recov_npz.is_file() or not card_path.is_file():
            continue
        card = json.loads(card_path.read_text())
        candidates.append((trial_dir, card))

    if not candidates:
        raise SystemExit(f"no trials with recovery_trajectory.npz under {run_dir}")

    print(f"[render] discovered {len(candidates)} trial(s) with recoveries "
          f"in {run_dir.relative_to(REPO_ROOT) if run_dir.is_relative_to(REPO_ROOT) else run_dir}")

    args.out.mkdir(parents=True, exist_ok=True)
    global_index = args.index_offset
    t0 = time.time()
    rendered = 0
    import numpy as np  # local — exporter imports trigger heavier deps earlier

    for k, (trial_dir, card) in enumerate(candidates):
        gate_pert = card.get("gate_perturbation")

        # Apply per-trial perturbation to the gsplat before render. The
        # renderer's apply_dynamic_edits baseline-restores first, so
        # perturbations don't compound across episodes — even though we
        # reuse the same renderer instance.
        #
        # IMPORTANT: GateRigidPerturbation.set_absolute_deltas only
        # stores values in _absolute_delta_*; they're copied into the
        # `_delta_*` attributes that `apply()` actually reads ONLY when
        # `reset(rng)` runs. The orchestrator/PerturbationSuite calls
        # reset automatically; we call it ourselves here.
        if gate_pert is not None:
            pert = GateRigidPerturbation(
                offset_half_widths=(0.0, 0.0, 0.0),   # unused; absolute deltas below
                yaw_half_width_rad=0.0,
                scene_cfg=scene_cfg,
                name="gate_rigid_perturbation",
            )
            pert.set_absolute_deltas(
                delta_xyz=gate_pert["delta_xyz"],
                delta_yaw_rad=gate_pert["delta_yaw_rad"],
            )
            pert.reset(np.random.default_rng(0))   # materialize _delta_* from _absolute_delta_*
            pert.apply(renderer)
        else:
            # No perturbation on this trial — restore to baseline by
            # applying a zero edit (still baseline-restores first).
            renderer.apply_dynamic_edits([])

        traj = load_trajectory(trial_dir / "recovery_trajectory.npz")
        episode_index = args.episode_index_base + k
        ep_dir = args.out / f"episode_{episode_index:06d}"
        t_ep = time.time()
        result = exporter.export_episode(
            traj, ep_dir,
            episode_index=episode_index,
            index_offset=global_index,
            task_index=args.task_index,
        )
        dt = time.time() - t_ep
        global_index += result.n_frames
        rendered += 1
        gp_tag = (f"Δxyz={[round(v,3) for v in gate_pert['delta_xyz']]} "
                  f"Δyaw={gate_pert['delta_yaw_rad']:.3f}"
                  if gate_pert else "no-pert")
        print(f"[ep {episode_index:06d}] {result.n_frames} frames "
              f"({result.n_frames/embodiment.fps:.1f}s) → {ep_dir.relative_to(args.out.parent) if args.out.parent in ep_dir.parents else ep_dir}  "
              f"({dt:.1f}s wall) · {gp_tag}")

    elapsed = time.time() - t0
    print(f"[done] {rendered} episodes, {global_index - args.index_offset} frames "
          f"in {elapsed:.0f}s ({elapsed/rendered:.1f}s/ep average)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
