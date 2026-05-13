# `falsify.geometry/` — frame contract

This package is the **single source of truth** for coordinate-frame handling
in falsify. Every position/pose/trajectory/point cloud crossing a module
boundary carries a `Frame`, and every conversion goes through a `FrameGraph`.

## Why this exists

The previous project (SousVide) carried frame context only in variable names
(`pos_ned`, `pos_ns`). When the gsplat changed or a new frame appeared, the
team had to chase frame conversions sprinkled across `orchestrator.py`,
`splatnav_recovery.py`, etc. Falsify replaces that pattern with frame-tagged
types + a runtime transform registry, so:

- New scenes (different alignment, scaled differently, with different intermediate frames) can be added with a single YAML.
- New camera mounts (rear, wrist) are config edits, not library refactors.
- "Render frame" vs "collision frame" asymmetries are explicit frames in the graph, not duplicated transform matrices.

## Modules in this package

| File | Purpose |
|------|---------|
| `frames.py` | `Frame` dataclass + canonical defaults (`NED`, `MOCAP`, `COLMAP`, `NS`, `CAM_BODY`, `CAM_FORWARD`, `CAM_DOWNWARD`) |
| `types.py` | `Point`, `Pose`, `Trajectory`, `PointCloud` (all frame-tagged) + `assert_frame()` |
| `transforms.py` | `SE3`, `Sim3` with `@` overload; composition rules: SE3∘SE3=SE3, anything-with-Sim3=Sim3 |
| `frame_graph.py` | `FrameGraph` — runtime registry + BFS conversion + auto inverse |
| `loaders.py` | Plugin registry for YAML transform types: `permutation`, `se3_inline`, `sim3_inline`, `se3_file`, `sim3_file`, `sim3_matrix_file`, `dataparser` |
| `presets.py` | Named axis-permutation table (`perm0`…`perm7`, `ned_to_zup_xyflip`) compatible with SousVide |
| `conventions.py` | Documented constants (quaternion order `xyzw`, axis conventions) |

## Adding a new frame

1. Declare it in your scene YAML's `frames:` list:
   ```yaml
   frames:
     - { name: my_new_frame, notes: "what this is" }
   ```
2. Declare at least one transform to a frame already in the graph:
   ```yaml
   transforms:
     - { src: my_new_frame, dst: ned, type: se3_inline, R: [[...]], t: [...] }
   ```
3. Reload the scene. `FrameGraph.convert(Point(..., my_new_frame), to="ns")`
   now works — composition is computed by BFS.

You do **not** need to touch any Python file. If you need a totally new
transform source (say, an HDF5 file), register a loader:

```python
from falsify.geometry import register_loader
register_loader("my_hdf5_transform", my_loader_fn)
```

then reference it as `type: my_hdf5_transform` in the YAML.

## Adding a new transform type (loader)

`loaders.register_loader(name, fn)` accepts any callable matching
``fn(spec_dict, frame_lookup, base_path) -> SE3 | Sim3``. The built-in
loaders demonstrate the pattern; see `_load_dataparser` for the
Nerfstudio convention.

## Verifying a frame setup

After building the graph, dump it:

```python
graph = build_frame_graph(load_yaml("configs/scenes/left_gate.yaml"),
                          base_path=Path("configs/scenes"))
print(graph.describe())
```

Run the round-trip test:

```bash
PYTHONPATH=src pytest tests/test_frames.py -v
```

The test asserts that for every pair of registered frames, the composed
`A → B → A` transform recovers a random point to within 1e-6.

## Conventions

- **Quaternions**: x, y, z, w (scipy `Rotation` standard).
- **FiGS NED**: x = North, y = East, z = Down (gravity along +z).
- **Cameras**: optical frames are OpenCV-style (x = right, y = down, z = forward into the image).
- **Sim3 application**: `p_dst = s * (R @ p_src) + t`. The `dataparser` loader translates Nerfstudio's `transform` matrix into this convention by absorbing the scale into the translation.

## Static vs runtime frame composition

The `FrameGraph` is for **static** transforms — relationships that don't
change during an episode. World-aligned frames (`mocap`, `ned`, `colmap`,
`ns`) are static. **Body-relative** camera frames (`cam_body`, `cam_forward`,
`cam_downward`) are also static *with respect to each other* but their
relationship to the world frame is the runtime drone pose, which is **not**
in the graph.

So:

- `graph.convert(point, "cam_body" → "cam_forward")` works (static extrinsic).
- `graph.convert(point, "cam_body" → "mocap")` raises `KeyError` — there is
  no static edge; you must compose `T_world_from_body` (the drone state) with
  the graph's `T_body_from_cam` yourself. The `sim.cameras.CameraRig` does
  this composition behind the `render(camera_name, drone_state)` API.

This split is intentional: dynamic edges in a static registry would invite
the same staleness/copy bugs the rewrite is trying to avoid.

## Anti-patterns to avoid

- ❌ Returning a bare `np.ndarray` from a function. Wrap it.
- ❌ Hardcoding `Frame("ns")` in library code. Look up by name through `FrameGraph`.
- ❌ Storing two matrices `T_a_to_b_for_render` and `T_a_to_b_for_collision`. If they're truly different transforms, register a third frame and a separate edge.
- ❌ Skipping `assert_frame` at module boundaries. The cost is microseconds; the failure mode it prevents is silent wrong-frame propagation.
