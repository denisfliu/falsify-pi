"""Canonical Trajectory NPZ — the single intermediate type between
trajectory generation and training-data export.

Schema
------
An NPZ archive with these arrays::

    times             (N,)        float64   seconds, monotonically increasing
    positions_ned     (N, 3)      float64   drone position in NED meters
    quaternions_xyzw  (N, 4)      float64   body→NED rotation, xyzw layout
    velocities_ned    (N, 3)      float64   optional — defaults to None
    prompt            (0-d)       <U…       optional task string
    source            (0-d)       <U…       optional provenance tag
                                            ("vla_rollout", "vla_replay",
                                             "mpc", "splatnav_recovery", ...)

The schema is intentionally minimal. Anything a producer wants to attach
(controller signals, perturbation manifest, etc.) goes in a *sibling* JSON
file in the same directory — the NPZ stays load-by-load cheap.

Producers in this module
------------------------
``from_episode_trace(ep)`` — convert a live ``EpisodeTrace`` into a
Trajectory. Used by VLA rollouts.

``from_vla_run_dir(run_dir, chunk_steps=50)`` — parse an existing
``runs/vla_<stamp>/vla_io/`` directory by stitching its recorded chunks
the way the simulator did. Generalises ``scripts/debug/replay_renders.py``.

Resampling
----------
``resample(traj, hz)`` returns a new Trajectory at the requested rate by
linear-interpolating positions/velocities and slerping quaternions. Useful
when the training pipeline wants a different rate than the source produced.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Trajectory dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Trajectory:
    """Frame-tagged trajectory in NED.

    Always NED — the exporter converts to MOCAP per the embodiment's
    state-layout spec. Keeping the canonical type in NED matches the rest of
    falsify's pipeline (simulator state, policy I/O boundary).
    """
    times: np.ndarray              # (N,) float64
    positions_ned: np.ndarray      # (N, 3) float64
    quaternions_xyzw: np.ndarray   # (N, 4) float64
    velocities_ned: Optional[np.ndarray] = None  # (N, 3) float64
    prompt: str = ""
    source: str = ""

    def __post_init__(self):
        t = np.asarray(self.times, dtype=np.float64)
        p = np.asarray(self.positions_ned, dtype=np.float64)
        q = np.asarray(self.quaternions_xyzw, dtype=np.float64)
        if t.ndim != 1:
            raise ValueError(f"times must be 1D, got shape {t.shape}")
        n = t.shape[0]
        if p.shape != (n, 3):
            raise ValueError(f"positions_ned must be ({n}, 3), got {p.shape}")
        if q.shape != (n, 4):
            raise ValueError(f"quaternions_xyzw must be ({n}, 4), got {q.shape}")
        if not np.all(np.diff(t) >= 0):
            raise ValueError("times must be non-decreasing")
        object.__setattr__(self, "times", t)
        object.__setattr__(self, "positions_ned", p)
        object.__setattr__(self, "quaternions_xyzw", q)
        if self.velocities_ned is not None:
            v = np.asarray(self.velocities_ned, dtype=np.float64)
            if v.shape != (n, 3):
                raise ValueError(f"velocities_ned must be ({n}, 3), got {v.shape}")
            object.__setattr__(self, "velocities_ned", v)

    def __len__(self) -> int:
        return self.times.shape[0]

    @property
    def duration_s(self) -> float:
        return float(self.times[-1] - self.times[0]) if len(self) > 0 else 0.0


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def save_trajectory(path: str | Path, traj: Trajectory) -> Path:
    """Save a Trajectory to ``.npz``. Returns the resolved path."""
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "times": traj.times,
        "positions_ned": traj.positions_ned,
        "quaternions_xyzw": traj.quaternions_xyzw,
    }
    if traj.velocities_ned is not None:
        payload["velocities_ned"] = traj.velocities_ned
    if traj.prompt:
        payload["prompt"] = np.array(traj.prompt)
    if traj.source:
        payload["source"] = np.array(traj.source)
    np.savez(path, **payload)
    return path


def load_trajectory(path: str | Path) -> Trajectory:
    """Load a Trajectory NPZ written by ``save_trajectory``."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        times = data["times"]
        positions = data["positions_ned"]
        quats = data["quaternions_xyzw"]
        velocities = data["velocities_ned"] if "velocities_ned" in data.files else None
        prompt = str(data["prompt"]) if "prompt" in data.files else ""
        source = str(data["source"]) if "source" in data.files else ""
    return Trajectory(
        times=times, positions_ned=positions, quaternions_xyzw=quats,
        velocities_ned=velocities, prompt=prompt, source=source,
    )


# ---------------------------------------------------------------------------
# Producers
# ---------------------------------------------------------------------------


def from_episode_trace(ep, *, prompt: str = "", source: str = "vla_rollout") -> Trajectory:
    """Convert a live ``EpisodeTrace`` into a Trajectory.

    Reads `ep.states` (or a `FalsificationEpisode.trace.states`) and packs
    them. All `DroneState`s must be in NED.
    """
    trace = getattr(ep, "trace", ep)
    states = trace.states
    if not states:
        raise ValueError("trace.states is empty")
    times = np.array([s.t for s in states], dtype=np.float64)
    positions = np.stack([s.pos.xyz for s in states], axis=0).astype(np.float64)
    quats = np.stack([s.quat_xyzw for s in states], axis=0).astype(np.float64)
    vels = np.stack([s.vel for s in states], axis=0).astype(np.float64)
    return Trajectory(
        times=times,
        positions_ned=positions,
        quaternions_xyzw=quats,
        velocities_ned=vels,
        prompt=prompt,
        source=source,
    )


# ---------------------------------------------------------------------------
# VLA-run-dir → Trajectory
# ---------------------------------------------------------------------------


_DATA_PAT = re.compile(r"^([^:]+):\s*(.*)$")


def _parse_kv(text: str) -> dict:
    out: dict = {}
    for line in text.splitlines():
        m = _DATA_PAT.match(line.strip())
        if not m:
            continue
        out[m.group(1).strip()] = m.group(2).strip()
    return out


def _parse_vec3(s: str) -> np.ndarray:
    return np.array(json.loads(s), dtype=np.float64)


def _yaw_to_quat_xyzw(yaw: float) -> np.ndarray:
    return np.array([0.0, 0.0, np.sin(0.5 * yaw), np.cos(0.5 * yaw)])


def from_vla_run_dir(
    run_dir: str | Path,
    *,
    chunk_steps: int = 50,
    hz: int = 10,
    prompt: Optional[str] = None,
) -> Trajectory:
    """Reconstruct a Trajectory from a recorded VLA run directory.

    Reads ``run_dir/vla_io/query_*/{data.txt,actions.npy,waypoints_ned.npy}``,
    stitches the chunks the way the simulator did, and packs the result.
    Yaw is integrated from action column 3 (sign-flipped per the VLA frame
    convention — see ``policy/vla.py`` for the exact rule).
    """
    run_dir = Path(run_dir)
    vla_io = run_dir / "vla_io"
    qdirs = sorted([p for p in vla_io.iterdir() if p.is_dir()])
    if not qdirs:
        raise FileNotFoundError(f"no query_* directories under {vla_io}")

    chunks = []
    inferred_prompt: Optional[str] = None
    for d in qdirs:
        kv = _parse_kv((d / "data.txt").read_text())
        chunks.append({
            "start_pos_ned": _parse_vec3(kv["state_ned_pos"]),
            "start_yaw_ned": float(kv["state_ned_yaw_rad"]),
            "actions": np.load(d / "actions.npy"),
            "waypoints_ned": np.load(d / "waypoints_ned.npy"),
        })
        if inferred_prompt is None:
            p = kv.get("prompt", "")
            if p.startswith("'") and p.endswith("'"):
                p = p[1:-1]
            inferred_prompt = p

    times_l: list[float] = []
    positions_l: list[np.ndarray] = []
    yaws_l: list[float] = []

    cur_pos = chunks[0]["start_pos_ned"].copy()
    cur_yaw = chunks[0]["start_yaw_ned"]
    step_global = 0

    for chunk in chunks:
        cur_pos = chunk["start_pos_ned"].copy()
        cur_yaw = chunk["start_yaw_ned"]
        wp = chunk["waypoints_ned"]
        actions = chunk["actions"]
        n_steps = min(chunk_steps, len(wp))
        has_yaw_action = actions.shape[1] >= 4
        for step_in_chunk in range(n_steps):
            times_l.append(step_global / hz)
            positions_l.append(cur_pos.copy())
            yaws_l.append(cur_yaw)
            cur_pos = wp[step_in_chunk]
            if has_yaw_action and step_in_chunk < len(actions):
                cur_yaw = cur_yaw - float(actions[step_in_chunk, 3])
            step_global += 1
    # final
    times_l.append(step_global / hz)
    positions_l.append(cur_pos.copy())
    yaws_l.append(cur_yaw)

    times = np.asarray(times_l, dtype=np.float64)
    positions = np.stack(positions_l, axis=0).astype(np.float64)
    quats = np.stack([_yaw_to_quat_xyzw(y) for y in yaws_l], axis=0)
    return Trajectory(
        times=times,
        positions_ned=positions,
        quaternions_xyzw=quats,
        prompt=prompt if prompt is not None else (inferred_prompt or ""),
        source="vla_replay",
    )


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


def resample(traj: Trajectory, hz: float) -> Trajectory:
    """Resample to a uniform rate. Linear interp positions/velocities; slerp quats."""
    if hz <= 0:
        raise ValueError(f"hz must be > 0, got {hz}")
    if len(traj) < 2:
        return traj
    t0 = float(traj.times[0])
    t1 = float(traj.times[-1])
    new_n = max(2, int(np.floor((t1 - t0) * hz)) + 1)
    new_times = t0 + np.arange(new_n) / hz
    new_times = np.clip(new_times, t0, t1)

    new_positions = np.stack([
        np.interp(new_times, traj.times, traj.positions_ned[:, i]) for i in range(3)
    ], axis=1)
    new_velocities = None
    if traj.velocities_ned is not None:
        new_velocities = np.stack([
            np.interp(new_times, traj.times, traj.velocities_ned[:, i]) for i in range(3)
        ], axis=1)

    from scipy.spatial.transform import Rotation as _R, Slerp
    rots = _R.from_quat(traj.quaternions_xyzw)
    slerp = Slerp(traj.times, rots)
    new_quats = slerp(new_times).as_quat()

    return Trajectory(
        times=new_times,
        positions_ned=new_positions,
        quaternions_xyzw=new_quats,
        velocities_ned=new_velocities,
        prompt=traj.prompt,
        source=traj.source,
    )
