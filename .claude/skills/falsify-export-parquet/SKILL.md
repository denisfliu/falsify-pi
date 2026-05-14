---
name: falsify-export-parquet
description: Render a canonical Trajectory NPZ against a falsify scene and emit one LeRobot-style training-data parquet (HuggingFace Image features, embedded PNG bytes). Schema matches DroneVLA2.0's `episode_NNNNNN.parquet` exactly.
---

# falsify-export-parquet

Renders a `Trajectory` NPZ through the dual-camera gsplat pipeline and
emits a parquet that drops straight into DroneVLA2.0's training inputs.
Per-episode output is one self-contained parquet plus a `manifest.json`.

## Inputs

| Input | Where it lives | What it controls |
|---|---|---|
| Trajectory NPZ | `--trajectory` or `--run-dir` | per-step pose sequence |
| Scene YAML | `--scene configs/scenes/<name>.yaml` | gsplat + FrameGraph |
| Drone-frame YAML | `--frame configs/frames/carl_dual.yaml` | camera intrinsics + body extrinsics |
| Embodiment YAML | `--embodiment configs/embodiments/<name>.yaml` | parquet schema (state, action, camera columns, channel order) |

## Procedure

```bash
CC=gcc-11 CXX=g++-11 \
  PYTHONPATH=src:external/FiGS/src:external/splatnav \
  .venv/bin/python -m falsify.cli.export_training_data \
    --run-dir runs/vla_20260512_160932 \
    --scene configs/scenes/left_gate.yaml \
    --frame configs/frames/carl_dual.yaml \
    --embodiment configs/embodiments/carl_dual_mocap.yaml \
    --out runs/datasets/left_gate \
    --episode-index 0 --index-offset 0
```

Three input modes (mutually exclusive):
- `--run-dir <run>` — reads the recorded VLA chunks via `from_vla_run_dir`.
- `--trajectory <path.npz>` — single canonical NPZ.
- `--trajectories-dir <dir>` — batch mode; one episode per `.npz` in the
  directory, sorted by filename. Reuses the loaded `GSplatRenderer`
  across episodes (the gsplat load is the slowest step).

## Outputs

Per episode:

```
<out>/episode_<NNNNNN>/
  episode_<NNNNNN>.parquet      # LeRobot/HF schema; matches DroneVLA2.0 exactly
  manifest.json                 # scene + embodiment + fps + prompt + source
```

Parquet columns (see `src/falsify/training/CLAUDE.md` for the full schema):

```
image, wrist_image, 3pov_1   struct<bytes: binary, path: string>
state                        fixed_size_list<float32>[N]
actions                      fixed_size_list<float32>[N]
timestamp                    float32
frame_index, episode_index, index, task_index   int64
```

Schema-level `huggingface` metadata blob is present so HF / LeRobot
loaders open it directly.

## Hands off to

- **`falsify-orchestrate-batch`** — to produce hundreds of episodes at once.
- **`falsify-combine-datasets`** — to merge many per-episode outputs (or
  multiple bundles) into one LeRobot v2.1 dataset with regenerated meta.
- Downstream training in DroneVLA2.0 (out of scope here): point at the
  combined dataset.

## Embodiments and channel order

The shipped `carl_dual_mocap` embodiment matches DroneVLA2.0's training
distribution exactly: `state = [x_mocap, y_mocap, z_mocap, yaw_mocap,
gripper, 0, 0]`, BGR channel order in PNGs (cv2 convention), 256x256
images, fps=10. Add new embodiments by copying the YAML — no code
changes needed.

## Gotchas

- `--episode-index` must be unique across episodes you intend to
  concatenate. `--index-offset` should be the running sum of previous
  episodes' `n_frames`.
- BGR vs RGB: the PNG bytes encode raw 3-channel data which a
  cv2-based reader sees as BGR. Don't decode with PIL and assume RGB
  (it will look swapped).
- The gsplat load is the slow step (~30 s). For >1 episode, prefer
  `--trajectories-dir` over invoking the CLI N times.
- Yaw is wrapped to `[-π, π]` for the action delta (matches
  `data_collection_lerobot.py`). The first action is all zeros.
