---
name: falsify-orchestrate-batch
description: Bulk-generate many training-data parquets by chaining trajectory producers and export-parquet. Reuses the loaded GSplatRenderer across episodes so the 30 s gsplat load is paid once per scene, not per episode.
---

# falsify-orchestrate-batch

Top-level "make me hundreds of parquets" workflow. Reuses one
`GSplatRenderer` per scene (the gsplat load is the dominant cost) and
chains the per-trajectory skills.

## Two ways to drive it

### A. CLI batch mode (single scene, many NPZs)

If you already have a directory of Trajectory NPZs from any producer
(VLA replays, mocks, MPC, SplatNav), `export_training_data` consumes them
in sorted order, auto-numbering `episode_index` from your `--episode-index`
base and bookkeeping `index` across episodes.

```bash
.venv/bin/python -m falsify.cli.export_training_data \
    --trajectories-dir runs/trajectories/left_gate \
    --scene configs/scenes/left_gate.yaml \
    --frame configs/frames/carl_dual.yaml \
    --embodiment configs/embodiments/carl_dual_mocap.yaml \
    --out runs/datasets/left_gate \
    --episode-index 0 --index-offset 0
```

This is the right path when (scene × trajectory) products fit "one scene,
many trajectories".

### B. Python orchestration (many scenes × many sources)

For (scene × trajectory) cross-products and mixed sources, drive
`TrainingDataExporter` directly. Construct *one* renderer per scene; iterate.

```python
from pathlib import Path
from falsify.io import build_frame_graph, load_yaml
from falsify.sim.renderer import GSplatRenderer
from falsify.training import (
    TrainingDataExporter, load_embodiment, load_trajectory,
)

embodiment = load_embodiment("configs/embodiments/carl_dual_mocap.yaml")
frame_cfg = load_yaml("configs/frames/carl_dual.yaml")

global_index = 0
episode_counter = 0
for scene_path in ["configs/scenes/left_gate.yaml", "configs/scenes/right_gate.yaml"]:
    scene_path = Path(scene_path)
    scene_cfg = load_yaml(scene_path)
    fg = build_frame_graph(scene_cfg, base_path=scene_path.parent)

    # GSplatRenderer.from_scene_cfg resolves gsplat_config_yml /
    # gsplat_data_cwd, builds the FrameGraph, AND applies any declared
    # scene_edits. Always use this in preference to the bare constructor —
    # the bare form is easy to instantiate without scene_edits, which silently
    # renders the un-edited scene (e.g. the center-gate scene is a left-gate
    # gsplat with edits; without them you'd render the original gate pose).
    renderer = GSplatRenderer.from_scene_cfg(scene_cfg, scene_dir=scene_path.parent)
    exporter = TrainingDataExporter(
        scene_cfg=scene_cfg, frame_cfg=frame_cfg, frame_graph=fg,
        renderer=renderer.render, embodiment=embodiment,
    )
    out_root = Path(f"runs/datasets/{scene_cfg['scene_key']}")
    out_root.mkdir(parents=True, exist_ok=True)

    for npz in sorted(Path(f"runs/trajectories/{scene_cfg['scene_key']}").glob("*.npz")):
        traj = load_trajectory(npz)
        result = exporter.export_episode(
            traj, out_root / f"episode_{episode_counter:06d}",
            episode_index=episode_counter, index_offset=global_index,
        )
        global_index += result.n_frames
        episode_counter += 1
```

This is also the path to write a small driver script that combines all
the trajectory-generation skills you want into a single batch.

## Outputs

`<out>/episode_NNNNNN/episode_NNNNNN.parquet` for each input trajectory.
The `episode_index` and `index` columns are monotonic across the whole
batch so the parquets concatenate cleanly with `pyarrow.concat_tables` or
LeRobot's combine helpers.

## Sources you can mix

| Trajectory producer | Skill | Maturity |
|---|---|---|
| Recorded VLA run | `falsify-trajectory-from-replay` | done |
| Live VLA | `falsify-trajectory-from-vla` | done |
| Waypoint course (spline) | `falsify-trajectory-from-waypoints` | done |
| Mock / hand-authored | `falsify-trajectory-from-mock` | done |
| FiGS-MPC plan | `falsify-trajectory-from-mpc` | stub |
| SplatNav plan | `falsify-trajectory-from-splatnav` | stub |
| Falsified variant | `falsify-falsify-trajectory` | stub |

## Hands off to

- **`falsify-combine-datasets`** — stitch the per-episode parquets into
  a single LeRobot v2.1 dataset (regenerated meta, renumbered indices,
  optional bad-last filtering, per-range task assignment). This is
  almost always the next step before training.
- Downstream training in DroneVLA2.0 — point the trainer at the combined
  dataset.

## Gotchas

- **Don't reload the gsplat per episode.** Load once per scene; loop.
- Increment `index_offset` correctly across episodes (the CLI's
  `--trajectories-dir` does this for you within one invocation).
- If you mix scenes, write each scene's episodes to a different
  output dir so the merge step is explicit.
