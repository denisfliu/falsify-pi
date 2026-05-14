"""Course / Waypoint YAML schema.

A *course* declares a sequence of waypoints in a named frame (typically
``mocap``, because that's how humans think about the scene) plus the
total trajectory duration and a yaw policy. Planners
(``plan_spline``, future ``plan_mpc``) consume a Course and emit a
canonical ``falsify.training.Trajectory`` in NED.

YAML shape (see ``configs/courses/through_left_gate.yaml`` for a
working example)::

    name: through_left_gate
    scene: configs/scenes/left_gate.yaml         # informational; resolved by callers
    frame: mocap                                  # frame the waypoints live in
    fps: 10
    total_time_s: 8.0
    yaw_mode: tangent                             # fixed | interp | tangent

    waypoints:
      - { name: start,     p: [-0.5,  0.7, 1.5], yaw: 0.0, t: 0.0 }
      - { name: pre_gate,  p: [ 0.3,  0.7, 1.5] }
      - { name: gate,      p: [ 0.86, 0.69, 1.5] }
      - { name: post_gate, p: [ 1.5,  0.7, 1.5] }
      - { name: goal,      p: [ 1.5, -0.3, 1.5], yaw: -1.57, t: 8.0 }

    velocity_constraints:                         # optional
      max_speed_mps: 1.5

Rules:
- Each waypoint must have ``p: [x, y, z]``.
- ``yaw`` is optional; resolved per the course-level ``yaw_mode``.
- ``t`` is optional. If at least the first and last waypoints have ``t``,
  intermediate ones get them by even spacing along path length. If no
  ``t`` is given, the loader assigns evenly along ``total_time_s``.
- ``frame`` defaults to ``mocap``.
- ``yaw_mode`` defaults to ``tangent`` (yaw follows direction of motion).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import yaml


@dataclass(frozen=True)
class Waypoint:
    name: str
    p: np.ndarray                # (3,) position in `Course.frame`
    yaw: Optional[float] = None  # radians; None → resolved by yaw_mode
    t: Optional[float] = None    # seconds since course start; None → auto-assigned

    def __post_init__(self):
        p = np.asarray(self.p, dtype=np.float64)
        if p.shape != (3,):
            raise ValueError(f"waypoint {self.name!r} p must be (3,), got {p.shape}")
        object.__setattr__(self, "p", p)


@dataclass(frozen=True)
class Course:
    name: str
    frame: str
    fps: int
    total_time_s: float
    yaw_mode: Literal["fixed", "interp", "tangent"]
    waypoints: tuple[Waypoint, ...]
    scene_path: Optional[Path] = None       # informational; planners resolve themselves
    max_speed_mps: Optional[float] = None
    max_yaw_rate_rad_s: Optional[float] = None
    notes: str = ""

    def __post_init__(self):
        if len(self.waypoints) < 2:
            raise ValueError(f"course {self.name!r} needs at least 2 waypoints")
        if self.total_time_s <= 0:
            raise ValueError(f"course {self.name!r} total_time_s must be > 0")
        # Validate t-monotonicity if all set.
        ts = [wp.t for wp in self.waypoints if wp.t is not None]
        if ts and any(b <= a for a, b in zip(ts[:-1], ts[1:])):
            raise ValueError(
                f"course {self.name!r} waypoint ``t`` values must be strictly "
                f"increasing where present"
            )

    @property
    def positions(self) -> np.ndarray:
        """(N, 3) waypoint positions in self.frame."""
        return np.stack([wp.p for wp in self.waypoints], axis=0)

    def resolved_times(self) -> np.ndarray:
        """Per-waypoint ``t`` values, filling in any None entries.

        Rule: gaps are filled by path-length parameterisation between the
        nearest set ``t``s. If no waypoint has ``t``, evenly distribute
        across ``[0, total_time_s]``.
        """
        n = len(self.waypoints)
        ts: list[Optional[float]] = [wp.t for wp in self.waypoints]
        # If no t set, distribute evenly.
        if all(t is None for t in ts):
            return np.linspace(0.0, self.total_time_s, n)
        if ts[0] is None:
            ts[0] = 0.0
        if ts[-1] is None:
            ts[-1] = self.total_time_s
        # Fill internal Nones by path-length between bracketing set entries.
        positions = self.positions
        # Compute cumulative chord length.
        diffs = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(diffs)])
        # For each None entry, find bracketing set indices and interp by chord length.
        for i in range(n):
            if ts[i] is None:
                # find left set index, right set index
                left = max(j for j in range(i) if ts[j] is not None)
                right = min(j for j in range(i + 1, n) if ts[j] is not None)
                frac = (cum[i] - cum[left]) / max(1e-12, cum[right] - cum[left])
                ts[i] = float(ts[left] + frac * (ts[right] - ts[left]))
        return np.asarray(ts, dtype=np.float64)

    def resolved_yaws(self) -> np.ndarray:
        """Per-waypoint yaw, filling Nones per ``yaw_mode``.

        - "fixed": all None → 0.0; set values respected.
        - "interp": Nones linearly interpolated between bracketing set values.
        - "tangent": Nones computed from local tangent direction; set values
          override.
        """
        n = len(self.waypoints)
        yaws: list[Optional[float]] = [wp.yaw for wp in self.waypoints]

        if self.yaw_mode == "fixed":
            return np.array([y if y is not None else 0.0 for y in yaws], dtype=np.float64)

        if self.yaw_mode == "interp":
            if yaws[0] is None:
                yaws[0] = 0.0
            if yaws[-1] is None:
                yaws[-1] = yaws[0]
            ts = self.resolved_times()
            set_t = [ts[i] for i in range(n) if yaws[i] is not None]
            set_y = [yaws[i] for i in range(n) if yaws[i] is not None]
            return np.interp(ts, set_t, set_y)

        if self.yaw_mode == "tangent":
            positions = self.positions
            out = np.zeros(n)
            for i in range(n):
                if yaws[i] is not None:
                    out[i] = yaws[i]
                    continue
                # finite-diff tangent (forward, backward, or central)
                if i == 0:
                    d = positions[1] - positions[0]
                elif i == n - 1:
                    d = positions[-1] - positions[-2]
                else:
                    d = positions[i + 1] - positions[i - 1]
                out[i] = float(np.arctan2(d[1], d[0]))
            return out

        raise ValueError(f"unknown yaw_mode {self.yaw_mode!r}")


def save_course(course: Course, path: str | Path) -> Path:
    """Write a Course back out as YAML.

    Format mirrors ``load_course``'s input schema. Optional fields are
    only emitted when set, so a saved-then-loaded course is structurally
    equivalent to the original.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg: dict = {
        "name": course.name,
        "frame": course.frame,
        "fps": int(course.fps),
        "total_time_s": float(course.total_time_s),
        "yaw_mode": course.yaw_mode,
    }
    if course.scene_path is not None:
        cfg["scene"] = str(course.scene_path)
    if course.notes:
        cfg["notes"] = course.notes
    wps: list[dict] = []
    for wp in course.waypoints:
        entry: dict = {"name": wp.name, "p": [float(x) for x in wp.p.tolist()]}
        if wp.yaw is not None:
            entry["yaw"] = float(wp.yaw)
        if wp.t is not None:
            entry["t"] = float(wp.t)
        wps.append(entry)
    cfg["waypoints"] = wps
    vel: dict = {}
    if course.max_speed_mps is not None:
        vel["max_speed_mps"] = float(course.max_speed_mps)
    if course.max_yaw_rate_rad_s is not None:
        vel["max_yaw_rate_rad_s"] = float(course.max_yaw_rate_rad_s)
    if vel:
        cfg["velocity_constraints"] = vel
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def load_course(path: str | Path) -> Course:
    """Parse a course YAML into a :class:`Course`."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())

    wps = tuple(
        Waypoint(
            name=str(w.get("name", f"wp_{i}")),
            p=np.asarray(w["p"], dtype=np.float64),
            yaw=(float(w["yaw"]) if "yaw" in w else None),
            t=(float(w["t"]) if "t" in w else None),
        )
        for i, w in enumerate(cfg["waypoints"])
    )
    vel = cfg.get("velocity_constraints", {}) or {}
    return Course(
        name=cfg["name"],
        frame=cfg.get("frame", "mocap"),
        fps=int(cfg.get("fps", 10)),
        total_time_s=float(cfg["total_time_s"]),
        yaw_mode=cfg.get("yaw_mode", "tangent"),
        waypoints=wps,
        scene_path=Path(cfg["scene"]) if "scene" in cfg else None,
        max_speed_mps=vel.get("max_speed_mps"),
        max_yaw_rate_rad_s=vel.get("max_yaw_rate_rad_s"),
        notes=cfg.get("notes", ""),
    )
