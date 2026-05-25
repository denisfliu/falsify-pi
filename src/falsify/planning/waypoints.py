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
from typing import Literal, Optional, Union

import numpy as np
import yaml


# Phase identity: ("pre_gate", k) or ("post_gate", k). 1-based gate index.
# Used by the recovery pipeline to scope safe-state sampling and pick the
# recovery course's first anchor without per-task hardcoding.
PhaseLabel = tuple[Literal["pre_gate", "post_gate"], int]


@dataclass(frozen=True)
class GatePhase:
    """A gate phase declaration. Names existing waypoints by name; the
    loader validates that each referenced waypoint exists in the
    course's ``waypoints`` list."""
    index: int        # 1-based gate index, contiguous from 1
    pre: str          # approach waypoint name
    in_: str          # in-aperture waypoint name (Python `in` is reserved)
    post: str         # past-the-gate waypoint name


@dataclass(frozen=True)
class GoalPhase:
    """The terminal phase. Drone hovers / parks on ``waypoint``."""
    waypoint: str


Phase = Union[GatePhase, GoalPhase]


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
class CorrectivePerturbation:
    """Course-level configuration for *corrective* variants — discrete shifts
    of one waypoint along {up, down, left, right} that train the policy to
    recover from being off-axis at a specific point in the trajectory
    (typically `pre_gate`).

    Sampling per variant (stochastic strategy):
      - Bernoulli(``probability``) decides whether this variant is corrective.
      - If corrective: uniformly pick a mode from ``modes`` and a magnitude
        from ``magnitude_range_m``; apply to ``target_waypoint``.
      - Otherwise the variant is baseline (no corrective shift), distinguished
        from other baselines only by the trajectory-noise block.
    """
    target_waypoint: str
    modes: tuple[str, ...] = ("up", "down", "left", "right")
    magnitude_range_m: tuple[float, float] = (0.1, 0.3)
    probability: float = 1.0    # per-sample chance to apply a corrective shift
    # ``samples_per_mode`` / ``seed`` are legacy fields from the deterministic
    # enumeration sampler; ignored by the stochastic strategy used by
    # ``sample_stochastic_variants``. Kept in the dataclass so old YAMLs load.
    samples_per_mode: int = 1
    seed: int = 0


@dataclass(frozen=True)
class TrajectoryPerturbation:
    """Course-level configuration for *trajectory* noise — small spherical
    jitter applied to every waypoint to induce variance across samples
    without changing the qualitative shape of the trajectory.

    Sampling pattern (planned, not yet implemented in ``perturb_course``):
      for each new variant:
        - per waypoint not in ``exclude_waypoints``:
            displace by a vector drawn uniformly in a ball of radius
            ``radius_m`` (xyz Gaussian-ish or uniform-in-sphere depending
            on implementation choice).

    Combine with a CorrectivePerturbation by applying both in sequence:
    pick a corrective mode (or skip for the standard-flight baseline),
    then add the trajectory noise on top.
    """
    radius_m: float = 0.05
    exclude_waypoints: tuple[str, ...] = ()
    samples: int = 1                # variants to draw per nominal
    seed: int = 0


# Back-compat alias: older modules may still import ``Perturbation``.
Perturbation = CorrectivePerturbation


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
    corrective_perturbations: Optional[CorrectivePerturbation] = None
    trajectory_perturbations: Optional[TrajectoryPerturbation] = None
    # Phase metadata — the recovery pipeline reads this to classify the
    # drone's progress along the course (N-gate-generic) and to slice
    # the course's waypoint suffix for replan. Tuple of GatePhase /
    # GoalPhase. Empty tuple means "no phase info" — legacy courses that
    # the recovery pipeline can't use.
    phases: tuple = ()

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
        if self.phases:
            self._validate_phases()

    # ---- phase metadata helpers --------------------------------------

    def _validate_phases(self) -> None:
        """Loader-time invariants for the ``phases`` block.

        - Every referenced waypoint name must exist in ``waypoints``.
        - GatePhase indices must be contiguous from 1.
        - Exactly one GoalPhase, last in the list.
        """
        wp_names = {wp.name for wp in self.waypoints}
        gate_phases = [p for p in self.phases if isinstance(p, GatePhase)]
        goal_phases = [p for p in self.phases if isinstance(p, GoalPhase)]
        if len(goal_phases) != 1:
            raise ValueError(
                f"course {self.name!r} must declare exactly one GoalPhase; "
                f"got {len(goal_phases)}"
            )
        if not isinstance(self.phases[-1], GoalPhase):
            raise ValueError(
                f"course {self.name!r}: GoalPhase must be the last phase"
            )
        for i, gp in enumerate(gate_phases, start=1):
            if gp.index != i:
                raise ValueError(
                    f"course {self.name!r}: gate indices must be contiguous "
                    f"from 1; got phase #{i} with index={gp.index}"
                )
            for slot, name in (("pre", gp.pre), ("in", gp.in_), ("post", gp.post)):
                if name not in wp_names:
                    raise ValueError(
                        f"course {self.name!r}: gate_{gp.index}.{slot} references "
                        f"unknown waypoint {name!r}"
                    )
        if goal_phases[0].waypoint not in wp_names:
            raise ValueError(
                f"course {self.name!r}: GoalPhase.waypoint {goal_phases[0].waypoint!r} "
                f"not found in waypoints"
            )

    @property
    def n_gates(self) -> int:
        return sum(1 for p in self.phases if isinstance(p, GatePhase))

    def gate(self, idx: int) -> GatePhase:
        """1-based gate access. Raises if idx out of range."""
        for p in self.phases:
            if isinstance(p, GatePhase) and p.index == idx:
                return p
        raise IndexError(
            f"course {self.name!r}: no gate with index {idx} "
            f"(have {self.n_gates})"
        )

    def goal(self) -> "GoalPhase":
        for p in self.phases:
            if isinstance(p, GoalPhase):
                return p
        raise ValueError(f"course {self.name!r}: no GoalPhase declared")

    def waypoint_index(self, name: str) -> int:
        """Index of the waypoint by name. Raises KeyError if not found."""
        for i, wp in enumerate(self.waypoints):
            if wp.name == name:
                return i
        raise KeyError(
            f"course {self.name!r}: waypoint {name!r} not found"
        )

    def phase_label(self, n_crossings: int) -> PhaseLabel:
        """Map a count of correct-direction aperture crossings to the
        current phase. ``n_crossings`` clamped to ``[0, n_gates]``.

        - 0 crossings ⇒ ``("pre_gate", 1)``
        - k crossings, k < N ⇒ ``("post_gate", k)`` (= about to enter gate k+1)
        - N crossings ⇒ ``("post_gate", N)`` (terminal phase, only goal left)
        """
        N = self.n_gates
        if N == 0:
            raise ValueError(f"course {self.name!r}: no gates declared")
        k = max(0, min(int(n_crossings), N))
        if k == 0:
            return ("pre_gate", 1)
        return ("post_gate", k)

    def target_waypoint(self, post_phase: PhaseLabel, seed_kind: str) -> str:
        """Pick the recovery's first anchor (the ``target`` waypoint)
        given the current phase and the seed's nature.

        ``seed_kind`` ∈ {"in_gate", "pre_gate"}:
        - "in_gate" (collisions, or Case A trim-regressed-across-boundary)
            push aggressively to the in-aperture waypoint of the next
            gate the drone has to clear; in ``post_gate_N`` there's no
            next gate, so the target degrades to the goal waypoint.
        - "pre_gate" (Beta-sampled non-collision failures): restart at
            the approach waypoint of the next gate the drone has to
            clear; in ``post_gate_N`` similarly degrades to goal.
        """
        if seed_kind not in ("in_gate", "pre_gate"):
            raise ValueError(f"unknown seed_kind {seed_kind!r}")
        kind, k = post_phase
        N = self.n_gates
        if kind == "pre_gate":
            gp = self.gate(k)
            return gp.in_ if seed_kind == "in_gate" else gp.pre
        elif kind == "post_gate":
            if k >= N:
                return self.goal().waypoint
            gp = self.gate(k + 1)
            return gp.in_ if seed_kind == "in_gate" else gp.pre
        else:
            raise ValueError(f"unknown phase kind {kind!r}")

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
    if course.corrective_perturbations is not None:
        cp = course.corrective_perturbations
        cfg["corrective_perturbations"] = {
            "target_waypoint": cp.target_waypoint,
            "modes": list(cp.modes),
            "magnitude_range_m": list(cp.magnitude_range_m),
            "probability": float(cp.probability),
        }
    if course.trajectory_perturbations is not None:
        tp = course.trajectory_perturbations
        cfg["trajectory_perturbations"] = {
            "radius_m": float(tp.radius_m),
            "exclude_waypoints": list(tp.exclude_waypoints),
            "samples": int(tp.samples),
            "seed": int(tp.seed),
        }
    if course.phases:
        phases_out = []
        for p in course.phases:
            if isinstance(p, GatePhase):
                phases_out.append({
                    "kind": "gate", "index": int(p.index),
                    "pre": p.pre, "in": p.in_, "post": p.post,
                })
            elif isinstance(p, GoalPhase):
                phases_out.append({"kind": "goal", "waypoint": p.waypoint})
        cfg["phases"] = phases_out
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

    # corrective_perturbations is the canonical key; ``perturbation`` is
    # accepted as a synonym for one upgrade cycle so older YAMLs still load.
    cp_cfg = cfg.get("corrective_perturbations") or cfg.get("perturbation")
    corrective = None
    if cp_cfg is not None:
        corrective = CorrectivePerturbation(
            target_waypoint=str(cp_cfg["target_waypoint"]),
            modes=tuple(cp_cfg.get("modes", ("up", "down", "left", "right"))),
            magnitude_range_m=tuple(cp_cfg.get("magnitude_range_m", (0.1, 0.3))),
            probability=float(cp_cfg.get("probability", 1.0)),
            samples_per_mode=int(cp_cfg.get("samples_per_mode", 1)),
            seed=int(cp_cfg.get("seed", 0)),
        )
    tp_cfg = cfg.get("trajectory_perturbations")
    trajectory = None
    if tp_cfg is not None:
        trajectory = TrajectoryPerturbation(
            radius_m=float(tp_cfg.get("radius_m", 0.05)),
            exclude_waypoints=tuple(tp_cfg.get("exclude_waypoints", ())),
            samples=int(tp_cfg.get("samples", 1)),
            seed=int(tp_cfg.get("seed", 0)),
        )
    # Phase metadata is optional — courses authored before the recovery
    # refactor have no `phases:` block and the recovery pipeline rejects
    # them with a clear error rather than guessing.
    phases_cfg = cfg.get("phases") or ()
    phases: list = []
    for p in phases_cfg:
        kind = p.get("kind")
        if kind == "gate":
            phases.append(GatePhase(
                index=int(p["index"]),
                pre=str(p["pre"]),
                in_=str(p["in"]),
                post=str(p["post"]),
            ))
        elif kind == "goal":
            phases.append(GoalPhase(waypoint=str(p["waypoint"])))
        else:
            raise ValueError(
                f"course {cfg['name']!r}: unknown phase kind {kind!r} "
                "(expected 'gate' or 'goal')"
            )

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
        corrective_perturbations=corrective,
        trajectory_perturbations=trajectory,
        phases=tuple(phases),
    )
