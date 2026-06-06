# Embodiment asset overlays

Static per-embodiment image overlays referenced by the embodiment YAMLs
(via `gripper_overlay_path` on a camera entry) and by the policy YAMLs
that share the same preprocess pipeline. Composited as the last step of
`CameraPostprocess.apply()` (see `src/falsify/policy/camera_postprocess.py`)
so the policy/exporter sees the same gripper occlusion the real drone's
camera sees.

## `carl_wrist_overlay.png`

256x256 RGBA. Models the static occlusion of the Carl drone's downward
camera by its own struts/gripper — the "wrist_image" column in the
LeRobot v2.1 datasets we train on.

**Built from** `data/atomic_datasets/gate_scenes_real_combined`
(LeRobot v2.1, 100 episodes / 25425 frames @ 10 fps, real teleop).

**Recipe** (regenerate by running the same steps if the airframe changes):

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
