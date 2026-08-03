# `falsify.training/` — trajectory → training-data export pipeline

**Status:** done. Single-trajectory and batch export both work; the
in-progress modularity items are the trajectory-source skills documented
under `.claude/skills/falsify-trajectory-from-*`.

## Three-layer modularity contract

```
        Trajectory (NPZ)                Scene + Embodiment
              │                                 │
              ▼                                 ▼
              └──────────►  TrainingDataExporter
                                    │
                                    ▼
                            episode_<id>.parquet
                            manifest.json
```

Each layer is swappable independently:

1. **Trajectory** (where it came from). NPZ schema in `trajectory.py`. Any
   producer that emits `(times, positions_ned, quaternions_xyzw)` plays.
   Producers shipped today: `from_episode_trace(ep)` for live VLA
   rollouts; `from_vla_run_dir(run_dir)` for replaying a recorded run.
   Future producers (FiGS-MPC plans, SplatNav recovery, falsified
   variants) drop in as new functions returning a `Trajectory`.

2. **Scene** (where it's rendered). Standard falsify scene YAML — the
   same `FrameGraph`, `GSplatRenderer`, body↔world hinge the rest of the
   stack uses. Switch `left_gate` ↔ `right_gate` via `--scene`.

3. **Embodiment** (what training-data shape to emit). `embodiment.py` +
   `configs/embodiments/*.yaml`. Decides:
   - which scene cameras feed which parquet image column
   - channel order written to PNG bytes — **RGB since the 2026-06-12
     convention change** (root `CLAUDE.md` § Image convention); the BGR
     era is frozen in `carl_dual_mocap_legacy_bgr.yaml`
   - state and action vector layouts (yaw convention, gripper, padding)
   - yaw wrapping behavior and first-action convention

   Adding a new embodiment is a YAML edit; code only changes when a
   *new modality* is introduced (e.g., depth, lidar, IMU).

## Output format

Per-episode directory:

```
<out>/episode_<NNNNNN>/
  episode_<NNNNNN>.parquet
  manifest.json
```

Parquet schema matches `~/Downloads/episode_000008.parquet` (LeRobot
v2-style HuggingFace dataset) exactly:

| Column | Type | Notes |
|---|---|---|
| `image` | `struct<bytes: binary, path: string>` | forward camera, PNG bytes, channel order per embodiment |
| `wrist_image` | same | downward camera |
| `3pov_1` | same | zeros by default; static-image option available |
| `state` | `fixed_size_list<float32>[N]` | layout from embodiment |
| `actions` | `fixed_size_list<float32>[N]` | per-step delta with yaw wrap |
| `timestamp` | `float32` | seconds since episode start |
| `frame_index` | `int64` | 0..N-1 within the episode |
| `episode_index` | `int64` | caller-supplied (`--episode-index`) |
| `index` | `int64` | global frame index (`--index-offset`) |
| `task_index` | `int64` | 0 for single-task; references task table |

Schema metadata carries the HuggingFace `info.features` JSON so LeRobot/HF
loaders can ingest the parquet directly.

## Trajectory NPZ schema

Canonical NPZ contents (see `trajectory.py`):

```
times             (N,)        float64   monotonically increasing seconds
positions_ned     (N, 3)      float64   meters
quaternions_xyzw  (N, 4)      float64   body→NED, xyzw layout
velocities_ned    (N, 3)      float64   optional
prompt            <U…         optional task string
source            <U…         optional provenance tag
```

Producers attach arbitrary side-info as a sibling JSON, not in the NPZ —
keeps trajectory loads cheap.

## How the exporter is used

CLI (one-shot or directory batch):

```bash
.venv/bin/python -m falsify.cli.export_training_data \
    --run-dir runs/vla_20260512_160932 \
    --scene configs/scenes/left_gate.yaml \
    --frame configs/frames/carl_dual.yaml \
    --embodiment configs/embodiments/carl_dual_mocap.yaml \
    --out runs/datasets/left_gate \
    --episode-index 0
```

Python (orchestration; reuses the loaded `GSplatRenderer`):

```python
from falsify.training import TrainingDataExporter, load_embodiment, from_vla_run_dir
exporter = TrainingDataExporter(
    scene_cfg=scene_cfg, frame_cfg=frame_cfg, frame_graph=fg,
    renderer=renderer.render, embodiment=load_embodiment("…carl_dual_mocap.yaml"),
)
for i, run_dir in enumerate(run_dirs):
    traj = from_vla_run_dir(run_dir)
    exporter.export_episode(traj, out_root / f"episode_{i:06d}", episode_index=i)
```

## Adding a new embodiment

1. Copy `configs/embodiments/carl_dual_mocap.yaml`.
2. Adjust `cameras`/`state`/`actions` lists to match the target dataset's
   schema. New state-field names need a getter in
   `exporter._STATE_GETTERS`; new action-field names that follow the
   `d_<state_field>` pattern work out of the box.
3. Re-run the CLI with `--embodiment` pointing at the new YAML.

## Per-camera postprocess (`CameraPostprocess`)

Every consumer of a rendered camera image — the exporter here, plus
`PiGatewayPolicy` and `VLAPolicy` — runs the same three transforms on the
raw render before handing it downstream:

1. PIL bilinear resize to `image_size`²
2. Channel swap (only when the embodiment / policy declares
   `channel_order: "BGR"` — legacy era; the current convention is RGB,
   where this step is a no-op)
3. Composite an optional RGBA gripper overlay (see next section)

These live in `src/falsify/policy/camera_postprocess.py` (`CameraPostprocess`
dataclass). The exporter builds one instance per `cameras[*]` column at
init time and calls `pp.apply(rgb)` inside the per-frame render loop. The
two VLA policies do the same per camera. Train/eval parity is therefore a
property of the **code**, not of YAML-keeping discipline.

## Gripper overlay

The real downward (wrist) camera always shows the drone's own struts /
gripper at a static position in the frame. The sim gsplat render has no
such occlusion. To close the distribution gap, the exporter and policy
both composite an RGBA PNG onto the final post-resize / post-channel-
swap downward image.

- Canonical asset:
  `configs/embodiments/assets/carl_wrist_overlay_pinhole_rgb.png`
  (256×256 RGBA, soft-edge alpha; authored in RGB against the
  KB4-undistorted `gate_scenes_real_combined_rgb` dataset via
  `scripts/dataset/build_wrist_overlay.py` — recipe in the assets
  README). The legacy BGR/fisheye asset `carl_wrist_overlay.png` stays
  for the twelve shipped v7/v9 policy YAMLs.
- Embodiment YAML knob: `cameras[*].gripper_overlay_path` (string,
  optional). Only set on the `wrist_image` entry today.
- Policy YAML knob (PiGateway + VLA): `gripper_overlay_paths:` is a
  `{cam_name: path}` map. For every gate-scenes pi_gateway YAML the
  downward camera points at the same asset the embodiment uses, so the
  bytes the policy sends at eval are byte-identical to what the
  exporter wrote at training time.
- Implementation: `CameraPostprocess.from_paths(image_size,
  channel_order, overlay_path)` loads the PNG once at consumer init;
  `apply()` does straight-alpha compositing on the final image. The
  overlay must already be in the channel order it'll be composited
  against (RGB for the current convention; the legacy asset is BGR,
  matching the dataset each was derived from).

If the airframe / camera mount changes, re-derive the overlay against a
fresh (undistorted, RGB) real-data dataset via
`scripts/dataset/build_wrist_overlay.py` — no code changes needed.

### Disabling the overlay at runtime

Every production CLI that drives the pipeline accepts `--no-gripper-overlay`:

| CLI | Effect |
|---|---|
| `falsify.cli.export_training_data` | The embodiment is cloned with `gripper_overlay_path=None` on every camera before exporter init — output parquets have **no overlay** on `wrist_image` while keeping all other preprocess identical (resize + BGR swap). Use to build ablation datasets. |
| `falsify.cli.run_vla_episode` (pi_gateway path) | `PiGatewayConfig.gripper_overlay_paths` is reset to `{}` after YAML load. |
| `scripts/eval/run_eval_campaign.py` | Same override applied to every trial in the campaign. |
| `scripts/recovery/collect_recovery_trajectories.py` | Same override applied for every recovery trial. |

The flag is a *runtime* override — the YAMLs still declare the production
overlay path, so re-running without the flag picks the overlay back up.
For permanent removal, edit the YAML.

## Adding a new trajectory producer

1. Write a function returning a `Trajectory`. Convention:
   `from_<source>(...)` — e.g., `from_mpc_plan(course, drone_params)`.
2. Set `Trajectory.source` to a short tag for provenance.
3. The exporter is agnostic to producers — no changes needed there.

## Why not LeRobotDataset directly?

The user only needs the parquets (matching DroneVLA2.0's existing pipeline
ingests parquets straight from `~/Downloads/episode_NNNNNN.parquet`).
A `LeRobotDataset` wrapper would add a dependency and forces a particular
on-disk layout that's not needed yet. The Parquet writer here is a
`WriterBackend` Protocol in spirit — a `LeRobotWriter` can be added
later without touching the exporter.

## Verification

- Unit tests in `tests/test_training_export.py` cover the trajectory
  roundtrip, resampling, schema parity with the reference parquet, and
  yaw-wrap behaviour.
- End-to-end against a recorded VLA run: see the `falsify-export-parquet`
  skill (`.claude/skills/falsify-export-parquet/SKILL.md`) for the
  canonical invocation.
