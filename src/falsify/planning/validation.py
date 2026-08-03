"""Post-plan trajectory validation — never save a bad trajectory.

Runs every sample of a planned :class:`Trajectory` through the same safety
criteria the rollout detector uses (bounds, speed, tilt, drone-OBB vs scene
point clouds), built from a scene + safety YAML pair. Rollout-only latching
criteria (``miss_gate`` / ``ordered_miss_gate``) are stripped: validation is
about physical sanity of an open-loop plan, not task semantics.

Consumers:
- ``falsify.cli.plan_trajectory`` — refuses to write the NPZ on violation.
- ``scripts/recovery/collect_recovery_trajectories.py`` — refuses to harvest
  a recovery NPZ whose flight clips the (possibly perturbed) scene.

When the trial carries a gate perturbation, pass its deltas via
``gate_deltas`` so the collision clouds are rigidly moved to match the
perturbed gaussians — the same ``_gate_deltas`` hook the detector factory
already implements.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from falsify.geometry import FrameGraph, Point
from falsify.sim.dynamics_state import DroneState
from falsify.training.trajectory import Trajectory

# safety-YAML keys that participate in plan validation; everything else
# (miss_gate, ordered_miss_gate, …) is rollout-only and stripped
_VALIDATION_KEYS = (
    "bounds_frame", "bounds_lower", "bounds_upper",
    "max_speed", "max_tilt_rad",
    "drone_body", "collision",
)


@dataclass
class ValidationResult:
    ok: bool
    n_steps: int
    criteria: list[str]
    failure_step: Optional[int] = None
    failure_type: Optional[str] = None
    description: Optional[str] = None

    def summary(self) -> str:
        if self.ok:
            return (f"OK — {self.n_steps} steps checked against "
                    f"[{', '.join(self.criteria)}]")
        return (f"VIOLATION at step {self.failure_step}/{self.n_steps - 1}: "
                f"{self.failure_type} — {self.description}")


# collision criterion is built from these two keys; drop them to validate
# kinematics (bounds/speed/tilt) only — used when the gate collision geometry
# postdates the courses and is being temporarily ignored.
_COLLISION_KEYS = ("drone_body", "collision")


def validation_safety_cfg(safety_cfg: dict,
                          gate_deltas: Optional[dict] = None,
                          *, ignore_collision: bool = False) -> dict:
    keys = _VALIDATION_KEYS
    if ignore_collision:
        keys = tuple(k for k in keys if k not in _COLLISION_KEYS)
    cfg = {k: v for k, v in safety_cfg.items() if k in keys}
    if gate_deltas is not None and not ignore_collision:
        cfg["_gate_deltas"] = gate_deltas
    return cfg


def validate_trajectory(
    traj: Trajectory,
    frame_graph: FrameGraph,
    *,
    scene_cfg: dict,
    scene_dir: Path,
    safety_cfg: dict,
    gate_deltas: Optional[dict] = None,
    ignore_collision: bool = False,
) -> ValidationResult:
    """Step a fresh FailureDetector over every trajectory sample; the first
    violation fails the plan. Velocities default to zero when the trajectory
    doesn't carry them (the speed criterion then can't fire — bounds, tilt
    and collision still do).

    ``ignore_collision`` drops the drone-OBB collision criterion, validating
    kinematics (bounds/speed/tilt) only. Use when the scene's gate collision
    geometry postdates the courses being planned and clipping is a known,
    temporarily-tolerated artifact rather than a real crash."""
    # The detector factory lives with the CLI wiring; import lazily so
    # `falsify.planning` stays light to import.
    from falsify.cli.smoke_test import _build_detector_factory

    cfg = validation_safety_cfg(safety_cfg, gate_deltas,
                                ignore_collision=ignore_collision)
    detector = _build_detector_factory(scene_cfg, Path(scene_dir))(
        frame_graph, cfg)
    criteria_names = [type(c).__name__ for c in detector.criteria]

    ned = frame_graph.frame("ned")
    n = len(traj.times)
    vels = traj.velocities_ned
    for i in range(n):
        state = DroneState(
            pos=Point.of(*np.asarray(traj.positions_ned[i], dtype=float), ned),
            vel=(np.asarray(vels[i], dtype=float) if vels is not None
                 else np.zeros(3)),
            quat_xyzw=np.asarray(traj.quaternions_xyzw[i], dtype=float),
            t=float(traj.times[i]),
        )
        record = detector.update(state, i)
        if record is not None:
            return ValidationResult(
                ok=False, n_steps=n, criteria=criteria_names,
                failure_step=record.failure_step,
                failure_type=str(record.failure_type),
                description=record.description,
            )
    return ValidationResult(ok=True, n_steps=n, criteria=criteria_names)
