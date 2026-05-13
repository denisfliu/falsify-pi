---
name: falsify-trajectory-from-mock
description: Hand-author or scripted-policy trajectories (straight lines, helices, splines) as canonical Trajectory NPZs — useful for renderer sanity checks and dataset bootstrapping without a VLA.
---

# falsify-trajectory-from-mock

Quick way to produce a Trajectory NPZ without invoking a VLA. Use it to:

- bootstrap a training dataset with simple sweeps
- exercise the renderer on a known path (sanity check)
- generate baseline data for ablations

## Procedure

Inline Python (the canonical pattern; package up into a script if you
find yourself running it repeatedly):

```python
import numpy as np
from pathlib import Path
from falsify.training import Trajectory, save_trajectory

hz = 10
horizon_s = 8.0
n = int(hz * horizon_s) + 1
times = np.arange(n) / hz

# Example: straight line from (-0.5, -0.7, -1.5) toward the gate in NED.
start = np.array([-0.5, -0.7, -1.5])
end   = np.array([ 1.5, -0.7, -1.5])
positions_ned = np.linspace(start, end, n)

# Identity orientation (facing +x_NED).
quats = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))

traj = Trajectory(
    times=times,
    positions_ned=positions_ned,
    quaternions_xyzw=quats,
    prompt="go through the gate and hover over the stuffed animal",
    source="mock_straight_line",
)
save_trajectory(Path("runs/mock_traj.npz"), traj)
```

Other shapes worth scripting:
- **Helix** through gate center (lift `_helix_in_mocap` from
  `src/falsify/cli/visualize_frames.py`; convert NED ↔ MOCAP through the
  scene's `FrameGraph`).
- **Spline through waypoints** — read `gate_position_mocap` from the
  scene YAML, build a cubic spline through start → gate → goal in MOCAP,
  convert to NED for the Trajectory.

## Hands off to

- **`falsify-export-parquet`** — render and emit training data.
- **`falsify-trajectory-from-mpc`** (future) — when waypoints should be
  physically realisable rather than analytically authored.

## Gotchas

- Quaternions must satisfy `||q|| = 1`. Bare identity `[0, 0, 0, 1]` is
  fine. If you yaw, use `[0, 0, sin(y/2), cos(y/2)]` (rotation about +z_NED).
- Positions are NED. Don't accidentally hand mocap coordinates here;
  convert through the active `FrameGraph` first.
