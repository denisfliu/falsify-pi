# `falsify.io/` — config loading and episode persistence

**Status:** Phase 1 covers `build_frame_graph(scene_cfg)`; episode store is Phase 4+.

`config.py` exposes:
- `load_yaml(path) -> dict`
- `build_frame_graph(scene_cfg) -> FrameGraph` — dispatches each `transforms[i].type` through `falsify.geometry.loaders`.

`episode_store.py` (later) persists `FalsificationEpisode` to parquet + JSON.
