---
name: falsify-trajectory-from-replay
description: Build a canonical Trajectory NPZ from an already-recorded VLA run directory, without running the VLA again.
---

# falsify-trajectory-from-replay

Reconstructs a canonical `Trajectory` from a `runs/vla_<stamp>/`
directory's `vla_io/` chunks. Useful when the OpenPI server is offline,
when you want to re-export training data with a different embodiment,
or when you want to debug a previous rollout without burning more VLA
inference quota.

## Inputs

- a `runs/vla_<stamp>/` directory containing `vla_io/query_*/`

## Procedure

```bash
.venv/bin/python -c "
from pathlib import Path
from falsify.training import from_vla_run_dir, save_trajectory
run = Path('runs/vla_20260512_160932')   # adjust
traj = from_vla_run_dir(run, chunk_steps=50, hz=10)
out = save_trajectory(run / 'trajectory.npz', traj)
print('frames:', len(traj), 'duration_s:', traj.duration_s, '→', out)
"
```

The reconstructor walks the same chunk semantics the simulator did (see
`Simulator.rollout_with_policy` in `sim/simulator.py`): re-queries every
`chunk_steps` frames, integrating yaw with the SousVide-style sign flip
(`yaws_ned[i+1] = yaws_ned[i] - action[i, 3]`).

## Outputs

- `runs/vla_<stamp>/trajectory.npz` — canonical NPZ.

## Hands off to

- **`falsify-export-parquet`** — turn it into training data.
- **`falsify-orchestrate-batch`** — bulk-convert many run dirs at once.

## Gotchas

- `chunk_steps=50` must match the simulator's `actions_per_chunk` used
  for the original run. Mismatched values produce inconsistent yaw and
  position drift between chunks.
- The recorded yaw at chunk-N start (`data.txt:state_ned_yaw_rad`) is
  used as the per-chunk seed — this insulates the replay from a sign
  bug we had in `VLAPolicy` early on (fixed). If you ever doubt the
  recorded yaw, recompute from the action chain.
