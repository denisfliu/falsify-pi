# `falsify.cli/` — entry points

- `smoke_test.py` — single-episode runner against `configs/falsification/smoke.yaml`.
  Supports `--stub-recovery` for env-less runs. Uses mock policies; never
  touches CUDA or the OpenPI server.
- `visualize_frames.py` — load a scene YAML, dump a deterministic helix
  trajectory + the scene's object PLYs (from `scene_objects:` in the YAML) in
  every configured frame. Runs a numerical round-trip check before writing.
  The intended sanity tool when adding/swapping a scene. Each emitted PLY
  records the frame name in its header.
- `visualize_waypoints.py` — render a Course YAML (waypoints) against a
  scene; writes per-frame PLYs (markers + planned spline + scene
  objects). Use during waypoint authoring (see
  `.claude/skills/falsify-author-waypoints`).
- `plan_trajectory.py` — Course YAML → Trajectory NPZ via the
  spline planner (default; `--planner mpc/splatnav` stubs for future).
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
