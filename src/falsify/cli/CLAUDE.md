# `falsify.cli/` — entry points

- `smoke_test.py` — single-episode runner against `configs/falsification/smoke.yaml`.
  Supports `--stub-recovery` for env-less runs. Uses mock policies; never
  touches CUDA or the OpenPI server.
- `visualize_frames.py` — load a scene YAML, dump a deterministic helix
  trajectory + the scene's object PLYs (from `scene_objects:` in the YAML) in
  every configured frame. Runs a numerical round-trip check before writing.
  The intended sanity tool when adding/swapping a scene. Each emitted PLY
  records the frame name in its header.
- `inspect_scene_plotly.py` — interactive plotly HTML for picking
  waypoints. Renders scene point clouds + gate AABBs + plane-cut posts
  + the implied start. Prints MOCAP landmarks to stdout for direct
  copy-into-YAML. Optional `--course` overlays an authored course's
  waypoints and planned spline.
- `visualize_waypoints.py` — offline-PLY equivalent of the inspector;
  same content but as `.ply` files for MeshLab / open3d / blender.
  Use during waypoint authoring (see
  `.claude/skills/falsify-author-waypoints`).
- `plan_trajectory.py` — Course YAML → Trajectory NPZ via the
  spline planner (default; `--planner mpc/splatnav` stubs for future).
- `perturb_course.py` — generate variant course YAMLs (center / up /
  down / left / right of one waypoint). Used for corrective-maneuver
  dataset generation; see `.claude/skills/falsify-perturb-course`.
- `combine_lerobot.py` — merge multiple LeRobot v2.1 dataset directories
  into one (drop bad-last trajectories, renumber episode_index / index,
  reassign task_index per range, regenerate all four meta files).
  Schema-identical to DroneVLA2.0's `episode_000008.parquet`. See
  `.claude/skills/falsify-combine-datasets`.
- `preview_scene_nsviewer.py` — launches nerfstudio's `ns-viewer` against
  a falsify scene with any declared `scene_edits` applied to the loaded
  pipeline before the viewer starts. The live gsplat the viewer renders
  is exactly what `GSplatRenderer` would render in a rollout.
  **Requires the gsplat CUDA path**: if `~/.cache/torch_extensions/.../gsplat_cuda`
  fails to JIT-rebuild (pybind11 `cast.h:45` template error), the viewer
  starts and reports `Gaussians modified` correctly but dies on the first
  rendered frame. See the env caveat in the top-level `CLAUDE.md`. For a
  CUDA-free check that the move is correct, use `paint_gaussian_mask`
  (lives at means level) instead.
- `author_gaussian_mask.py` — static-HTML plotly tool that classifies
  Gaussian means by current mask membership (moved / stranded /
  in-exclude / outside). Read-only diagnostic; fast iterate-on-YAML loop.
- `paint_gaussian_mask.py` — Dash app companion to the above. 3D viewer
  with x/y/z + yaw sliders to frame a candidate box, paint Gaussians
  into the include set, and get both an axis-aligned AABB and an
  oriented box in paste-ready YAML. Reads the loaded scene's
  `scene_edits` and classifies every visible Gaussian by mask membership
  using the same `(broad \ exclude) ∪ precise` semantics the applier
  uses (precise inclusions override excludes). Because the classifier
  reads `pipeline.model.means` directly, it works even when the gsplat
  CUDA path is broken. See `.claude/skills/falsify-author-gaussian-mask`.
- `export_training_data.py` — render a `Trajectory` NPZ (or a recorded
  VLA run, or a directory of NPZs) to one LeRobot-style parquet per
  episode. Reuses the same `GSplatRenderer` + `FrameGraph` the rollout
  uses. See `src/falsify/training/CLAUDE.md` for the contract.
- `run_vla_episode.py` — full VLA-driven rollout against the OpenPI server.
  Smoke-imports `openpi_client`, `figs`, `nerfstudio`, asserts CUDA, opens
  + closes a websocket handshake, then runs one episode at `--hz` with
  chunks of `--actions-per-chunk` waypoints. Writes three bundles under
  `--out`: `frames/combined_<frame>.ply` (trajectory + scene clouds),
  `flythrough.mp4` (forward-camera renders along the flown path), and
  `vla_io/query_*` (per-query VLA inputs/outputs). Use `--skip-handshake`
  for renderer-only smoke runs.
- `run_falsification.py` (future) — full campaign driver.

## Running with the SousVide venv

The default `.venv/` is a symlink to `~/code/SousVide/.venv`. That venv has
working `openpi_client`, `nerfstudio`, `torch` (cu121), and the editable
`falsify` package is **not** registered there; pass `PYTHONPATH` instead:

```bash
PYTHONPATH=src:external/FiGS/src:external/splatnav \
  .venv/bin/python -m falsify.cli.run_vla_episode --help
```

The `external/FiGS/src` and `external/splatnav` entries are workarounds for
the symlinked venv's stale `__editable__.figs-0.1.0.pth` (points at a path
that doesn't exist in SousVide). If you ever run `uv sync` in this repo
you can drop the PYTHONPATH entries.
