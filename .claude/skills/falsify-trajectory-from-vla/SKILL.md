---
name: falsify-trajectory-from-vla
description: Run one VLA episode against a falsify scene and save the resulting trajectory as a canonical Trajectory NPZ ready for training-data export.
---

# falsify-trajectory-from-vla

Produces a canonical `Trajectory` NPZ from a live VLA rollout. The OpenPI
server (default `moraband:8000`) drives a chunked rollout through
`falsify.cli.run_vla_episode`; the resulting `runs/vla_<stamp>/` directory
is then converted to a Trajectory NPZ via `from_vla_run_dir`.

## Inputs

- scene YAML (e.g., `configs/scenes/left_gate.yaml`)
- drone-frame YAML (`configs/frames/carl_dual.yaml`)
- prompt string
- (optional) OpenPI host/port

## Procedure

1. **Smoke-check the VLA server is reachable.** A refused TCP connection
   on the OpenPI port causes the CLI to hang in retry forever; verify
   first:

   ```bash
   timeout 3 bash -c 'cat </dev/tcp/moraband.stanford.edu/8000' \
     && echo "VLA up" || echo "VLA down"
   ```

   If the server is down, fix that before continuing (see
   `falsify-debug-render` if you also see rendering issues).

2. **Run the episode.** The CLI handles smoke imports, websocket
   handshake, scene + gsplat load, chunked rollout, and the three output
   bundles (flythrough mp4, per-frame PLYs, VLA debug dump):

   ```bash
   CC=gcc-11 CXX=g++-11 \
     PYTHONPATH=src:external/FiGS/src:external/splatnav \
     .venv/bin/python -m falsify.cli.run_vla_episode \
       --scene configs/scenes/left_gate.yaml \
       --frame configs/frames/carl_dual.yaml \
       --prompt "go through the gate and hover over the stuffed animal" \
       --out runs/vla_$(date +%Y%m%d_%H%M%S) \
       --hz 10 --actions-per-chunk 50 --horizon-s 30
   ```

   The `CC=gcc-11 CXX=g++-11` env vars are required on this host until
   gsplat's CUDA JIT cache is warm (see `falsify-debug-render`).

3. **Pack the run dir as a Trajectory NPZ.** Once `runs/vla_<stamp>/`
   exists:

   ```bash
   .venv/bin/python -c "
   from pathlib import Path
   from falsify.training import from_vla_run_dir, save_trajectory
   run_dir = Path('runs/vla_<stamp>')
   traj = from_vla_run_dir(run_dir, chunk_steps=50, hz=10)
   save_trajectory(run_dir / 'trajectory.npz', traj)
   print('frames:', len(traj), 'duration_s:', traj.duration_s)
   "
   ```

## Outputs

- `runs/vla_<stamp>/` — full run dir (mp4 + plys + `vla_io/`)
- `runs/vla_<stamp>/trajectory.npz` — canonical Trajectory NPZ

## Hands off to

- **`falsify-export-parquet`** — turn `trajectory.npz` into training data.
- **`falsify-debug-render`** — when renders are gray, frames look wrong,
  or the gsplat CUDA build fails.

## Common gotchas

- The OpenPI server's first inference may take ~6 s; subsequent calls
  are sub-second.
- If `nvidia-smi` is missing or torch sees no CUDA, the renderer fails
  in the CLI's smoke check. Fix the driver (see `falsify-debug-render`).
- Frame conventions are pinned: `R_mocap_from_ned = diag(1, -1, -1)`
  (SousVide perm5). FiGS' `Tw2g` is overridden from the `FrameGraph`
  inside `GSplatRenderer` so the dataparser-load drift is invisible to
  callers. Do not touch.
