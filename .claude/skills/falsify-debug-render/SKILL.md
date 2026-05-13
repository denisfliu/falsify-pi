---
name: falsify-debug-render
description: Diagnostic playbook for falsify rendering / frame-convention issues — gray PNGs, gaussians invisible in ns-viewer, wrong-direction camera, gsplat CUDA JIT failure. Captures the lessons from the gate-scene bring-up.
---

# falsify-debug-render

The "renders look wrong" playbook. Use it whenever:
- output PNGs are uniform / near-uniform gray
- the flythrough mp4 shows empty space
- ns-viewer shows camera frustums but no gaussians
- the gsplat CUDA build fails during JIT
- yaw / camera direction looks flipped or scaled

The diagnostic checklist runs cheapest-to-most-invasive.

## 0. Confirm the obvious

```bash
# GPU available?
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
.venv/bin/python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.device_count())"

# OpenPI VLA server reachable? (only relevant if running a VLA rollout)
timeout 3 bash -c 'cat </dev/tcp/moraband.stanford.edu/8000' \
  && echo "VLA up" || echo "VLA down"
```

If `torch.cuda.is_available()` is False but a 4090 is in `lspci`, the
driver isn't loaded — check `dmesg | tail` and reinstall
`linux-modules-nvidia-580-open-<kernel>` matching `uname -r`. See
`memory/today_<date>_kernel_module_recovery.md` if we wrote one.

## 1. Inspect the actual rendered PNGs

```bash
.venv/bin/python -c "
import numpy as np
from PIL import Image
for f in ['rgb_fwd.png', 'rgb_dwn.png']:
    p = 'runs/<run>/vla_io/query_0000_step_00000/' + f
    img = np.asarray(Image.open(p))
    print(f'{f}: shape={img.shape} std={img.std():.1f} unique={len(np.unique(img))}')
"
```

Decision tree on the result:
- **`std < 5`, `unique < 20`**: near-uniform background. Rasterizer
  produced no opacity in this view. Camera is outside the splat or the
  splat itself isn't loaded. → step 2.
- **`std > 30`, `unique == 256`**: the renderer IS producing content;
  problem is downstream (encoder, channel order, embodiment). → step 5.
- **`std` low but distinct between cameras**: forward camera is OK,
  downward is wrong (or vice versa). Camera extrinsic issue. → step 4.

## 2. Cross-check in ns-viewer

```bash
cd data/gate_scenes_export/<scene>
CC=gcc-11 CXX=g++-11 \
  PYTHONPATH=/home/dfliu/code/falsify/src:/home/dfliu/code/falsify/external/FiGS/src:/home/dfliu/code/falsify/external/splatnav \
  /home/dfliu/code/falsify/.venv/bin/ns-viewer \
  --load-config mocap_outputs/sagesplat_mocap/sagesplat/<timestamp>/config.yml \
  --viewer.websocket-host 0.0.0.0 --viewer.websocket-port 7007
```

Open `http://0.0.0.0:7007`. **In the panel set "Output type" to `rgb`** —
sagesplat defaults can land on `similarity` / `affordance` which need a
CLIP prompt and otherwise render blank. If `rgb` shows the gaussians,
the model is fine. If it doesn't, see step 3.

The viewer's default spawn point may be far from the splat; navigate
until you see content. Note the camera position shown in the panel —
that's the NS-frame coordinate the model considers "inside the scene".

## 3. Verify FiGS' Tw2g matches the FrameGraph

The falsify-pinned FiGS submodule does **not** load
`dataparser_transforms.json` on its own — its `Tw2g` ships as a bare
axis-flip. `GSplatRenderer` overrides this from the active `FrameGraph`
when a `frame_graph=` is passed. Confirm:

```python
from falsify.sim.renderer import GSplatRenderer
from falsify.io import build_frame_graph, load_yaml
scene = load_yaml("configs/scenes/left_gate.yaml")
fg = build_frame_graph(scene, base_path="configs/scenes")
renderer = GSplatRenderer(
    "data/gate_scenes_export/.../config.yml",
    world_frame="ned",
    data_cwd="data/gate_scenes_export/left_scene",
    frame_graph=fg,                # ← without this, Tw2g is wrong
)
import numpy as np
print(np.asarray(renderer._impl.Tw2g))
```

The diagonal should carry the dataparser scale (~0.126 for the gate
scenes), not be {±1}. If you see `diag(1, -1, -1, 1)` exactly, the
override didn't happen — check that you're passing `frame_graph=`.

## 4. Check camera extrinsics in the scene YAML

Conventions:
- Drone body frame is FRD (x=forward, y=right, z=down).
- Camera optical convention is OpenGL: x=right, y=up, z=backward (lens
  points to -z_cam). nerfstudio expects this.

Forward camera (lifted from SousVide carl.json, validated):
```yaml
- { src: cam_body, dst: cam_forward, type: se3_inline,
    R: [[0, 1, 0], [0, 0, -1], [-1, 0, 0]],
    t: [0.03, -0.01, 0.10] }
```

Downward camera (image-up = forward, image-right = body-right, lens
points body +z):
```yaml
- { src: cam_body, dst: cam_downward, type: se3_inline,
    R: [[0, 1, 0], [1, 0, 0], [0, 0, -1]],
    t: [0.0, 0.0, 0.05] }
```

If the downward render looks like the forward render: the rotation is
identity, lens is pointing along body +x not +z.

## 5. Channel order / encoding

FiGS returns RGB. DroneVLA2.0's collector uses cv2 → BGR PNGs.
`carl_dual_mocap` embodiment swaps RGB → BGR before encoding. If you
roll a custom embodiment, set `channel_order` deliberately.

```python
# Read a PNG; if a known-red scene shows the red value at channel 2 not 0,
# the file is BGR-encoded.
from PIL import Image; import numpy as np
img = np.asarray(Image.open("samples/.../front.png"))
print(img[0, 0])   # (B, G, R) for cv2-conventional PNG
```

## 6. NED ↔ MOCAP convention

`R_mocap_from_ned = diag(1, -1, -1)` (SousVide perm5). FiGS' baked-in
axis-flip uses the same matrix, so the renderer and the VLA policy agree
without glue *as long as the scene YAML matches*. A wrong sign here is
silent: state passes round-trips, but the rendered viewpoint is rotated.

Quick numeric sanity:
```python
from falsify.geometry import Point
p_mocap = Point.of(0.5, -0.3, 1.2, fg.frame("mocap"))
p_ned = fg.convert(p_mocap, to="ned")
print(p_ned.xyz)            # → (+0.5, +0.3, -1.2)  for perm5
```

If you see `(-0.5, +0.3, +1.2)` or similar, the scene YAML's `ned↔mocap`
edge is wrong.

## 7. Yaw / quaternion frames

Yaw conventions are different in NED vs MOCAP (axis flips sign).
`VLAPolicy.observe` should build quaternions in NED and attach them to
a *NED-frame* trajectory directly; routing them through
`frame_graph.convert(traj_mocap, to="ned")` applies perm5 to the quats
which is a 180° flip about x — not the yaw remapping you want. The fix
landed; if you re-introduce a MOCAP-frame Trajectory carrying
quaternions, you'll see the rendered drone rotate the wrong way.

## 8. gsplat CUDA JIT failure

If torch's JIT build of `gsplat_cuda` fails with pybind11 errors like
`expected template-name before '<' token`, the system gcc is too new
(Ubuntu 24.04's gcc-13 trips this). Force gcc-11 for the JIT:

```bash
CC=gcc-11 CXX=g++-11 <your command>
```

The cached `.so` lives at
`~/.cache/torch_extensions/py311_cu121/gsplat_cuda/gsplat_cuda.so` —
once built, subsequent runs skip the compile.

## Hands off to

- **`falsify-trajectory-from-vla`** / **`falsify-export-parquet`** —
  back to the happy path once the bug is found.
