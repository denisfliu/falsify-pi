# `falsify.visualization/` — frame-aware pointcloud dumps + html replay

**Status:** done.

## Public API

- `write_ply(pc: PointCloud, path)` — ASCII PLY writer; **the active frame is recorded as a `comment` in the header** so a loader / human can tell which frame the file is in.
- `read_ply(path, frame)` — ASCII / binary-little-endian PLY reader; the caller supplies the frame (PLY files do not reliably encode frame metadata, so the scene YAML's `scene_objects` block is the source of truth).
- `subsample(pc, max_points)` — deterministic random subsample for keeping visualization file sizes reasonable.
- `trajectory_to_pointcloud(traj, color=None)` — view a `Trajectory` as a `PointCloud` (frame preserved).
- `stack_pointclouds(clouds)` — concatenate point clouds **in the same frame** into one.
- `dump_trajectory_in_frames(traj, frame_graph, out_dir, target_frames=(...))` — one `.ply` per requested frame.
- `dump_episode(episode, frame_graph, out_dir, target_frames=(...))` — nominal trajectory + recovery trajectory + start/goal/failure/last-safe markers, each emitted in every target frame.
- `html_replay(episode, frame_graph, out_path, view_frame="ned")` — interactive plotly html.

The `falsify.cli.visualize_frames` CLI composes these: it loads scene
PLYs declared under `scene_objects:` in the scene YAML, converts them
through the FrameGraph alongside a sample trajectory, and writes a
combined `.ply` per target frame so the trajectory and scene geometry
can be inspected together.

## Debugging workflow

When a trajectory looks wrong:

1. Run an episode, let `dump_episode` write per-frame `.ply`s.
2. Open the same scene in a 3D viewer (open3d / meshlab / blender) loading the per-frame files.
3. Flip between frames. Frame misalignments show up as obvious mis-projections — the trajectory ends up in the wrong room, axis-flipped, or scaled.

This is the intended debugging affordance — it's why every entity is
emitted in *every* requested frame, not just the one the user thinks is
canonical.

## Color convention

`DEFAULT_COLORS` maps entity names to RGB triplets (0–1):
- `nominal`: blue
- `recovery`: red
- `start`: green
- `goal`: yellow
- `failure`: orange
- `last_safe`: purple

These are also reused by the html replay.
