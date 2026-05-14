"""Waypoint perturbations for generating corrective-maneuver datasets.

A *waypoint perturbation* nudges one named waypoint of a base Course
along a sampled direction with a sampled magnitude, producing a new
Course. The downstream pipeline (plan_trajectory → export_training_data)
then turns each variant into one episode's training parquet — giving the
policy demonstrations of "I'm off to the right; correct left".

Directions
----------
- ``center``   — no perturbation (baseline / standard example)
- ``up``       — +z_mocap (world up)
- ``down``     — -z_mocap
- ``left``     — perpendicular to local flight direction, in xy.
                 Body-relative: a positive perturbation is the same physical
                 side regardless of which gate the drone is approaching.
- ``right``    — opposite of ``left``

Body-relative left/right at waypoint ``i`` uses the local heading
estimated by the chord (waypoint i-1) → (waypoint i+1) in xy. This
avoids needing a planned spline before perturbing — we work directly on
the course definition.

API
---
- ``perturb_waypoint(course, name, direction, magnitude) -> Course``
  Single-shot perturbation; returns a new Course (the base is not mutated).
- ``sample_variants(course, waypoint_name, modes, magnitude_range_m,
  n_per_mode, seed) -> list[CourseVariant]``
  Reproducible batch generator. ``CourseVariant.label`` is a slug like
  ``left_003`` suitable for filenames.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Literal, Sequence

import numpy as np

from .waypoints import Course, Waypoint


Direction = Literal["center", "up", "down", "left", "right"]


@dataclass(frozen=True)
class CourseVariant:
    """One sampled perturbation. ``label`` is filename-safe."""
    course: Course
    label: str           # e.g. "left_003"
    direction: Direction
    magnitude_m: float
    waypoint_name: str


# ---------------------------------------------------------------------------
# Direction → unit vector
# ---------------------------------------------------------------------------


def _heading_unit_xy(course: Course, waypoint_index: int) -> np.ndarray:
    """Estimate the local flight heading at a waypoint, in the course frame's xy.

    Uses the chord (i-1) → (i+1); falls back to forward/backward differences
    at the endpoints. Returns a unit 2-vector. If the local chord has zero
    xy length, returns ``[1, 0]`` (an arbitrary but stable default).
    """
    wps = course.waypoints
    n = len(wps)
    i = waypoint_index
    if i == 0:
        delta = wps[1].p - wps[0].p
    elif i == n - 1:
        delta = wps[-1].p - wps[-2].p
    else:
        delta = wps[i + 1].p - wps[i - 1].p
    d2 = np.array([delta[0], delta[1]], dtype=np.float64)
    norm = float(np.linalg.norm(d2))
    if norm < 1e-9:
        return np.array([1.0, 0.0])
    return d2 / norm


def _direction_to_unit_vec(
    direction: Direction,
    heading_unit_xy: np.ndarray,
) -> np.ndarray:
    """3-vector in the course frame for a body-relative ``direction``."""
    if direction == "center":
        return np.zeros(3)
    if direction == "up":
        return np.array([0.0, 0.0, 1.0])
    if direction == "down":
        return np.array([0.0, 0.0, -1.0])
    # Body-relative right = heading rotated 90° clockwise (positive x → positive y in screen coords)
    # In MOCAP xy: heading (hx, hy) → right (hy, -hx).
    right_xy = np.array([heading_unit_xy[1], -heading_unit_xy[0]])
    if direction == "right":
        return np.array([right_xy[0], right_xy[1], 0.0])
    if direction == "left":
        return np.array([-right_xy[0], -right_xy[1], 0.0])
    raise ValueError(f"unknown direction {direction!r}")


# ---------------------------------------------------------------------------
# Perturb
# ---------------------------------------------------------------------------


def _find_waypoint_index(course: Course, name: str) -> int:
    for i, wp in enumerate(course.waypoints):
        if wp.name == name:
            return i
    raise KeyError(
        f"waypoint {name!r} not in course {course.name!r}; "
        f"have: {[w.name for w in course.waypoints]}"
    )


def perturb_waypoint(
    course: Course,
    name: str,
    direction: Direction,
    magnitude_m: float,
) -> Course:
    """Return a new Course with one waypoint nudged.

    The course's other fields (yaw_mode, total_time_s, etc.) and other
    waypoints are preserved by-value. ``direction == "center"`` returns
    a structurally-equivalent copy.
    """
    i = _find_waypoint_index(course, name)
    heading = _heading_unit_xy(course, i)
    unit = _direction_to_unit_vec(direction, heading)
    delta = unit * float(magnitude_m)
    new_p = course.waypoints[i].p + delta
    new_wp = replace(course.waypoints[i], p=new_p)
    new_wps = tuple(
        new_wp if j == i else wp for j, wp in enumerate(course.waypoints)
    )
    return replace(course, waypoints=new_wps)


# ---------------------------------------------------------------------------
# Batch sampling
# ---------------------------------------------------------------------------


def sample_variants(
    course: Course,
    waypoint_name: str,
    *,
    modes: Sequence[Direction] = ("center", "up", "down", "left", "right"),
    magnitude_range_m: tuple[float, float] = (0.2, 0.5),
    n_per_mode: int = 1,
    seed: int = 0,
) -> list[CourseVariant]:
    """Reproducibly sample N variants per direction.

    Magnitudes are uniformly drawn from ``magnitude_range_m`` per-sample.
    ``center`` always uses magnitude 0 regardless of the range.

    Returns one ``CourseVariant`` per (mode, sample). Variant labels are
    of the form ``"<mode>_<NNN>"`` (zero-padded so a sorted listing
    preserves order).
    """
    rng = np.random.default_rng(seed)
    lo, hi = float(magnitude_range_m[0]), float(magnitude_range_m[1])
    if lo > hi:
        raise ValueError(f"magnitude_range_m must be (lo, hi) with lo <= hi; got {magnitude_range_m}")
    if n_per_mode < 1:
        raise ValueError(f"n_per_mode must be >= 1; got {n_per_mode}")

    out: list[CourseVariant] = []
    width = max(2, len(str(n_per_mode - 1)))
    for mode in modes:
        for k in range(n_per_mode):
            if mode == "center":
                mag = 0.0
            else:
                mag = float(rng.uniform(lo, hi))
            variant = perturb_waypoint(course, waypoint_name, mode, mag)
            # Rename the variant's `course.name` so downstream artifacts
            # (manifest, parquet filenames) are distinguishable.
            label = f"{mode}_{k:0{width}d}"
            variant = replace(variant, name=f"{course.name}__{label}")
            out.append(CourseVariant(
                course=variant, label=label, direction=mode,
                magnitude_m=mag, waypoint_name=waypoint_name,
            ))
    return out
