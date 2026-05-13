# `falsify.io/` — config loading and episode persistence

**Status:** done for v0; episode parquet store is future work.

`config.py` exposes:
- `load_yaml(path) -> dict`
- `build_frame_graph(scene_cfg, *, base_path=cwd) -> FrameGraph` — dispatches each `transforms[i].type` through the loader registry in `falsify.geometry.loaders`. `base_path` is the directory of the scene YAML; loaders use it to resolve relative paths in `path:` fields. Unknown frame names fall back to the canonical defaults (`NED`, `MOCAP`, `NS`, `CAM_*`).

The scene YAML may also declare `scene_objects:` — a list of `(name, ply, frame, color)` entries that `falsify.cli.visualize_frames` and `run_vla_episode` read to overlay the scene's geometry on flown trajectories. The PLY paths are resolved against the scene YAML's directory.

`episode_store.py` (future) will persist `FalsificationEpisode` to parquet + JSON.
