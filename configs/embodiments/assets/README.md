# Embodiment asset overlays

Static per-embodiment image overlays referenced by the embodiment YAMLs
(via `gripper_overlay_path` on a camera entry) and by the policy YAMLs
that share the same preprocess pipeline. Composited as the last step of
`CameraPostprocess.apply()` (see `src/falsify/policy/camera_postprocess.py`)
so the policy/exporter sees the same gripper occlusion the real drone's
camera sees.

## `carl_wrist_overlay_pinhole_rgb.png` — CANONICAL

256x256 RGBA, true-RGB channel order, **pinhole-rectified** image space.
The overlay used by the current convention (root `CLAUDE.md` § Image
convention): composite target images are KB4-undistorted to the
calibrated pinhole K, so the occlusion silhouette was derived from
already-rectified frames.

**Built 2026-06-12 from** `data/atomic_datasets/gate_scenes_real_combined_rgb`
(KB4-undistorted + RGB-swapped real teleop) via:

```bash
PYTHONPATH=src python scripts/dataset/build_wrist_overlay.py \
    --dataset data/atomic_datasets/gate_scenes_real_combined_rgb \
    --out configs/embodiments/assets/carl_wrist_overlay_pinhole_rgb.png
```

Same recipe and knobs as below (coverage came out 18.4%). Referenced by
`configs/embodiments/carl_dual_mocap.yaml` and all future (RGB-era)
policy YAMLs.

## `carl_wrist_overlay.png` — LEGACY (BGR + raw fisheye)

256x256 RGBA. Models the static occlusion of the Carl drone's downward
camera by its own struts/gripper — the "wrist_image" column in the
LeRobot v2.1 datasets we train on. BGR channel order, authored against
**distorted** (raw fisheye) frames. Kept frozen for the twelve shipped
v7/v9 BGR-era policy YAMLs; do not use for new work.

**Built from** `data/atomic_datasets/gate_scenes_real_combined`
(LeRobot v2.1, 100 episodes / 25425 frames @ 10 fps, real teleop —
archived 2026-06-12 to `data/atomic_datasets_archive/`).

**Recipe** (now implemented as `scripts/dataset/build_wrist_overlay.py`;
rerun it against a fresh `_rgb` dataset if the airframe changes):

1. Sample 600 frames (20 evenly-spaced episodes x 30 evenly-spaced frames)
   of the `wrist_image` column.
2. Compute per-pixel median across the 600 frames -> `med` (uint8 RGB).
3. Compute median luminance `lum = 0.299*R + 0.587*G + 0.114*B`.
4. Hard boolean mask: `mask = (lum < 60) & (y < H/2)`.
   Drop connected components <20 px, then `binary_fill_holes`.
5. Alpha: `gaussian_filter(mask * 255, sigma=1.2)` (soft ~2 px edge).
6. RGB layer: `med` inside the mask, `(0, 0, 0)` outside (so the soft
   alpha edge fades to black rather than bleeding the median's
   scene-tinted halo).
7. Save as RGBA PNG.

Coverage at the chosen knobs: ~18.3% of the frame. Only the top half
contains the gripper; the bottom half is always clear by construction.

If the camera mount or gripper hardware changes, re-derive against a
fresh real-data dataset and overwrite this PNG. The knobs above
(`thr=60`, `sigma=1.2`, sample size) are the ones that were dialed
in for this airframe — start from them.

## Disabling the overlay at runtime

Pass `--no-gripper-overlay` to any of the four production CLIs:
`falsify.cli.export_training_data`, `falsify.cli.run_vla_episode`,
`scripts/eval/run_eval_campaign.py`, `scripts/recovery/collect_recovery_trajectories.py`.

The flag is a per-run override — the YAMLs still declare the overlay
path, so omitting the flag picks the overlay back up. Use the flag to
build ablation datasets or to diagnose whether a policy is sensitive
to the overlay's presence. Resize + BGR channel swap are unaffected;
only the alpha-composite step is skipped. See
`src/falsify/training/CLAUDE.md § Gripper overlay` for full details.
