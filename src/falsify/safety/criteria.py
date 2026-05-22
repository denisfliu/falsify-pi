"""Pluggable safety criteria.

Each criterion votes on whether a `DroneState` is safe and emits a small
"violation" description if not. The detector fuses votes (any failure
flips the verdict to unsafe).

Frame-aware design
------------------
Bounds are declared in a named frame (typically ``"ned"`` for the simulator's
native frame, but anything in the active `FrameGraph` works). The criterion
converts the incoming state into its frame via the graph and operates there.
Adding a new criterion = subclassing `SafetyCriterion` and declaring its
``operates_in_frame``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as _R

from falsify.geometry import FrameGraph, Point, assert_frame
from falsify.sim.dynamics_state import DroneState

if TYPE_CHECKING:
    from .records import FailureType


@dataclass
class Violation:
    description: str
    value: float
    threshold: float
    # Optional override of the failure-type lookup the detector would otherwise
    # do via `criterion.name`. Set this when a single criterion can produce
    # multiple `FailureType`s (e.g. PointCloudCollisionCriterion classifying
    # the hit as gate-vs-other after the fact).
    failure_type: Optional["FailureType"] = None
    extra: dict = field(default_factory=dict)


class SafetyCriterion(ABC):
    """Base class for safety checks.

    Subclasses declare the frame they operate in and implement `check`.
    """

    operates_in_frame: str = "ned"
    name: str = ""

    @abstractmethod
    def check(self, state_in_frame: DroneState) -> Optional[Violation]:
        """Return a `Violation` if the state is unsafe, else None."""
        ...

    def reset(self) -> None:
        """Clear any per-episode internal state.

        Default no-op. Criteria that track history across `check` calls
        (e.g. `MissGateCriterion` watching for plane crossings) override this.
        """

    def check_with_graph(
        self, state: DroneState, frame_graph: FrameGraph
    ) -> Optional[Violation]:
        """Convert `state` into this criterion's frame, then `check`.

        Velocity/quaternion don't currently get re-rotated when the state
        crosses frames — for v0 the rule is that criteria pick frames where
        these are interpretable directly (e.g. velocity criterion in NED).
        Specialized criteria that need transformed velocity should override
        this method.
        """
        if state.pos.frame.name == self.operates_in_frame:
            return self.check(state)
        # Convert position only; velocity stays in its original frame and is
        # only used by criteria that don't care about cross-frame velocities.
        new_pos = frame_graph.convert(state.pos, to=self.operates_in_frame)
        moved = DroneState(
            pos=new_pos, vel=state.vel, quat_xyzw=state.quat_xyzw, t=state.t,
        )
        return self.check(moved)


class BoundsCriterion(SafetyCriterion):
    """Axis-aligned-box safety bounds in a named frame.

    Lower and upper corners are passed as `Point`s — the **frame is read
    from the points themselves**, eliminating the possibility of accidentally
    specifying bounds in the wrong frame. Both points must share a frame.
    """
    name: str = "bounds"

    def __init__(self, lower: Point, upper: Point):
        if lower.frame.name != upper.frame.name:
            raise ValueError(
                f"BoundsCriterion: lower/upper frames disagree "
                f"({lower.frame.name!r} vs {upper.frame.name!r})"
            )
        self.lower_pt = lower
        self.upper_pt = upper
        self.operates_in_frame = lower.frame.name

    def check(self, state_in_frame: DroneState) -> Optional[Violation]:
        p = state_in_frame.pos.xyz
        lo = self.lower_pt.xyz
        hi = self.upper_pt.xyz
        if np.any(p < lo) or np.any(p > hi):
            worst = float(max(np.max(lo - p), np.max(p - hi)))
            return Violation(
                description=f"position {p.tolist()} outside [{lo.tolist()}, {hi.tolist()}] in frame {self.operates_in_frame!r}",
                value=worst,
                threshold=0.0,
            )
        return None


@dataclass
class VelocityCriterion(SafetyCriterion):
    """Speed (||v||) exceeds threshold."""
    max_speed: float = 5.0
    operates_in_frame: str = "ned"
    name: str = "velocity"

    def check(self, state_in_frame: DroneState) -> Optional[Violation]:
        speed = float(np.linalg.norm(state_in_frame.vel))
        if speed > self.max_speed:
            return Violation(
                description=f"speed {speed:.3f} m/s exceeds {self.max_speed:.3f}",
                value=speed,
                threshold=self.max_speed,
            )
        return None


@dataclass
class DroneBody:
    """Rectangular-prism collision shell of the drone, expressed in body (FRD) frame.

    The body frame is FRD: +x forward, +y right, +z down (matches the drone
    state convention — see `falsify.sim.dynamics_state`). ``half_extents`` is
    [hx, hy, hz] in metres so the physical footprint is
    ``2 * half_extents`` along each axis. ``center_offset_body`` shifts the
    box centre away from `state.pos` (useful if the drone's body origin is
    not at the geometric centre); defaults to zero.
    """
    half_extents: np.ndarray
    center_offset_body: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )

    def __post_init__(self) -> None:
        he = np.asarray(self.half_extents, dtype=np.float64)
        if he.shape != (3,) or np.any(he <= 0):
            raise ValueError(
                f"half_extents must be (3,) positive, got {he!r}"
            )
        co = np.asarray(self.center_offset_body, dtype=np.float64)
        if co.shape != (3,):
            raise ValueError(
                f"center_offset_body must be (3,), got {co.shape}"
            )
        # __post_init__ on a non-frozen dataclass — direct assignment is fine.
        self.half_extents = he
        self.center_offset_body = co

    @property
    def bounding_radius(self) -> float:
        """Radius of the smallest sphere centred at the box centre that
        contains the box. Used as the coarse cull radius for collision
        queries. ``|center_offset|`` is added so the cull includes the
        offset even when the body origin is not the box centre.
        """
        return float(
            np.linalg.norm(self.half_extents)
            + np.linalg.norm(self.center_offset_body)
        )


def _obb_contains(
    points_world: np.ndarray,
    *,
    centre_world: np.ndarray,
    R_world_from_body: np.ndarray,
    half_extents: np.ndarray,
    centre_offset_body: np.ndarray,
) -> np.ndarray:
    """Vectorised point-in-OBB test.

    Returns a boolean array, one entry per input point, ``True`` where the
    point lies inside the oriented box. The box centre in world coords is
    ``centre_world + R_world_from_body @ centre_offset_body``; the box's
    axes in world coords are the columns of ``R_world_from_body``.
    """
    centre = centre_world + R_world_from_body @ centre_offset_body
    rel = points_world - centre                       # (N, 3) in world
    # Express in body frame by left-multiplying by R^T.
    rel_body = rel @ R_world_from_body                # rel @ R == (R^T @ rel^T)^T
    return np.all(np.abs(rel_body) <= half_extents, axis=1)


class PointCloudCollisionCriterion(SafetyCriterion):
    """Rectangular-prism (drone body) collision against labeled point clouds.

    The criterion is constructed with one or more labeled point clouds in
    a single frame (typically ``"ned"``) — for example the gate vertices
    under label ``"gate"`` and table / floor vertices under ``"other"``.
    At every step the drone's OBB is built from ``state.pos`` and
    ``state.quat_xyzw`` and tested against each labeled cloud.

    Classification rule: if any ``"gate"`` point is inside the box, the
    violation is reported as `FailureType.COLLISION_GATE`. Otherwise, if
    any other-labelled point is inside, it is reported as
    `FailureType.COLLISION_OTHER`. ``"gate"`` wins ties — this matches the
    user's failure taxonomy where "drone hit the gate" is the explicit
    primary failure mode and "hit anything else" is the catch-all.

    Coarse-cull optimisation: each labelled cloud carries a per-point KD-
    style bound via numpy's broadcast — we first restrict the candidate
    set with a sphere of radius ``drone_body.bounding_radius`` around the
    OBB centre, then apply the exact OBB test only to the survivors.
    """

    name: str = "collision"

    def __init__(
        self,
        drone_body: DroneBody,
        labeled_clouds: dict[str, np.ndarray],
        *,
        operates_in_frame: str = "ned",
        gate_label: str = "gate",
    ) -> None:
        if not labeled_clouds:
            raise ValueError("labeled_clouds must contain at least one entry")
        self.drone_body = drone_body
        self.operates_in_frame = operates_in_frame
        self.gate_label = gate_label
        # Concat once for the fast sphere prefilter, but remember the slices
        # so we can recover the original label per point on a hit.
        chunks: list[np.ndarray] = []
        labels: list[str] = []
        offsets: dict[str, tuple[int, int]] = {}
        cursor = 0
        for label, pts in labeled_clouds.items():
            arr = np.ascontiguousarray(np.asarray(pts, dtype=np.float64))
            if arr.ndim != 2 or arr.shape[1] != 3:
                raise ValueError(
                    f"label {label!r}: expected (N, 3) points, got {arr.shape}"
                )
            chunks.append(arr)
            labels.append(label)
            offsets[label] = (cursor, cursor + arr.shape[0])
            cursor += arr.shape[0]
        self._all_points = np.vstack(chunks) if chunks else np.empty((0, 3))
        self._label_slices = offsets
        self._labels_in_order = labels

    # ---- frame plumbing ------------------------------------------------

    def check(self, state_in_frame: DroneState) -> Optional[Violation]:
        from .records import FailureType  # local import — avoid circular

        if self._all_points.size == 0:
            return None
        centre = np.asarray(state_in_frame.pos.xyz, dtype=np.float64)
        radius = self.drone_body.bounding_radius
        # Coarse sphere cull: keep only points whose squared distance to the
        # box-centre estimate is within the bounding radius. We use the body
        # *origin* as the centre estimate to skip the rotation here — the
        # error is at most |center_offset| which is already absorbed in
        # bounding_radius.
        deltas = self._all_points - centre
        within = (deltas * deltas).sum(axis=1) <= radius * radius
        if not np.any(within):
            return None

        R = _R.from_quat(state_in_frame.quat_xyzw).as_matrix()
        candidates = self._all_points[within]
        hits = _obb_contains(
            candidates,
            centre_world=centre,
            R_world_from_body=R,
            half_extents=self.drone_body.half_extents,
            centre_offset_body=self.drone_body.center_offset_body,
        )
        if not np.any(hits):
            return None

        # Map the hit-positions in the concatenated array back to labels.
        hit_global_idxs = np.where(within)[0][hits]
        labels_hit: set[str] = set()
        for label in self._labels_in_order:
            lo, hi = self._label_slices[label]
            if np.any((hit_global_idxs >= lo) & (hit_global_idxs < hi)):
                labels_hit.add(label)
        if self.gate_label in labels_hit:
            ftype = FailureType.COLLISION_GATE
            picked = self.gate_label
        else:
            ftype = FailureType.COLLISION_OTHER
            # Prefer a deterministic label for the description; pick the
            # first label-in-order that we hit.
            picked = next(
                (l for l in self._labels_in_order if l in labels_hit),
                "other",
            )
        n_hits = int(np.count_nonzero(hits))
        return Violation(
            description=(
                f"drone OBB contains {n_hits} point(s) labelled {picked!r} "
                f"at position {centre.tolist()} in frame "
                f"{self.operates_in_frame!r}"
            ),
            value=float(n_hits),
            threshold=0.0,
            failure_type=ftype,
            extra={"hit_label": picked, "n_hits": n_hits},
        )


@dataclass
class TiltCriterion(SafetyCriterion):
    """Body roll/pitch magnitude exceeds threshold (radians).

    The check is frame-independent (it's an internal body orientation), but
    we still declare ``operates_in_frame == "ned"`` for consistency.
    """
    max_tilt_rad: float = 1.2   # ~70 degrees
    operates_in_frame: str = "ned"
    name: str = "tilt"

    def check(self, state_in_frame: DroneState) -> Optional[Violation]:
        rot = _R.from_quat(state_in_frame.quat_xyzw)
        # World-z axis expressed in body frame after rotation; tilt is its
        # angle from world-z.
        world_z = np.array([0.0, 0.0, 1.0])
        body_z = rot.apply(world_z, inverse=True)
        cos_tilt = float(np.clip(np.dot(body_z, world_z), -1.0, 1.0))
        tilt = float(np.arccos(cos_tilt))
        if tilt > self.max_tilt_rad:
            return Violation(
                description=f"tilt {tilt:.3f} rad exceeds {self.max_tilt_rad:.3f}",
                value=tilt,
                threshold=self.max_tilt_rad,
            )
        return None


def _aperture_geometry(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Derive (centre, u, v, n, hu, hv) from a (4, 3) aperture-corner list.

    Enforces planarity and orthogonality of the two adjacent edges:
    ``corners[1] - corners[0]`` and ``corners[3] - corners[0]``. Shared by
    `MissGateCriterion` and `OrderedMissGateCriterion`.
    """
    if corners.shape != (4, 3):
        raise ValueError(
            f"corners must be (4, 3), got {corners.shape}; pass them in the "
            "order they trace the aperture rectangle"
        )
    centre = corners.mean(axis=0)
    e1 = corners[1] - corners[0]
    e2 = corners[3] - corners[0]
    if abs(float(np.dot(e1, e2))) > 1e-4 * (np.linalg.norm(e1) * np.linalg.norm(e2)):
        raise ValueError(
            "corner order must trace a rectangle: corners[1]-corners[0] "
            "and corners[3]-corners[0] should be orthogonal adjacent edges"
        )
    u = e1 / np.linalg.norm(e1)
    v = e2 / np.linalg.norm(e2)
    n = np.cross(u, v)
    n /= np.linalg.norm(n)
    if abs(float(np.dot(corners[2] - corners[0], n))) > 1e-4:
        raise ValueError("aperture corners are not coplanar")
    hu = 0.5 * float(np.linalg.norm(e1))
    hv = 0.5 * float(np.linalg.norm(e2))
    return centre, u, v, n, hu, hv


class MissGateCriterion(SafetyCriterion):
    """Catch every way the drone fails to navigate the gate.

    Emits ``MISS_GATE`` on any of:

    - **(a) Plane-crossing outside the aperture** — the drone crossed the
      gate plane but the crossing point is outside the rectangle defined
      by ``corners``. Classic geometric miss.
    - **(b) Reached goal proximity without transiting** — drone is within
      ``goal_tolerance_m`` of ``goal_position`` but never crossed the
      aperture plane inside the rectangle. The policy short-circuited to
      the goal without going through the gate.
    - **(c) Stuck before transit** — distance to the aperture centre has
      not decreased by at least ``min_progress_m`` in the trailing
      ``min_progress_window_s`` seconds. The drone is hovering, drifting,
      or otherwise failing to make headway toward the gate.

    Emits ``GOAL_NOT_REACHED`` when the drone *did* transit successfully
    but then fails to reach the goal:

    - **(d) Stuck after transit** — same no-progress check as (c) but
      using distance to ``goal_position`` instead of the aperture centre.

    Modes (b)/(c)/(d) only activate when ``goal_position`` /
    ``min_progress_window_s`` are configured. Without them, the criterion
    degrades to the original geometric-only miss check (a).

    Aperture geometry:

      - plane normal ``n``         = unit normal through the four corners,
      - in-plane axes ``u, v``     = orthonormal axes along adjacent edges,
      - half-widths ``hu, hv``     = aperture extents along ``u`` / ``v``,
      - centre                     = mean of the four corners.
    """

    name: str = "miss_gate"

    def __init__(
        self,
        corners: np.ndarray,
        *,
        frame_name: str = "mocap",
        margin_m: float = 0.0,
        goal_position: Optional[np.ndarray] = None,
        goal_tolerance_m: float = 0.30,
        goal_tolerance_half_extents: Optional[np.ndarray] = None,
        min_progress_window_s: Optional[float] = None,
        min_progress_m: float = 0.05,
        eval_stop_mode: bool = False,
        transit_aabb_min: Optional[np.ndarray] = None,
        transit_aabb_max: Optional[np.ndarray] = None,
    ) -> None:
        corners = np.asarray(corners, dtype=np.float64)
        if corners.shape != (4, 3):
            raise ValueError(
                f"corners must be (4, 3), got {corners.shape}; pass them in "
                "the order they trace the aperture rectangle"
            )
        # Validate planarity / orthogonality and derive frame.
        centre = corners.mean(axis=0)
        e1 = corners[1] - corners[0]
        e2 = corners[3] - corners[0]   # adjacent edge (must be the *short* side w.r.t. e1)
        if abs(float(np.dot(e1, e2))) > 1e-4 * (np.linalg.norm(e1) * np.linalg.norm(e2)):
            raise ValueError(
                "corner order must trace a rectangle: corners[1]-corners[0] "
                "and corners[3]-corners[0] should be orthogonal adjacent edges"
            )
        u = e1 / np.linalg.norm(e1)
        v = e2 / np.linalg.norm(e2)
        n = np.cross(u, v)
        n /= np.linalg.norm(n)
        # Coplanarity check on corner 2.
        if abs(float(np.dot(corners[2] - corners[0], n))) > 1e-4:
            raise ValueError("aperture corners are not coplanar")
        hu = 0.5 * float(np.linalg.norm(e1))
        hv = 0.5 * float(np.linalg.norm(e2))
        self.operates_in_frame = frame_name
        self.corners = corners
        self.centre = centre
        self.u = u
        self.v = v
        self.n = n
        self.hu = hu
        self.hv = hv
        self.margin_m = float(margin_m)
        # Optional goal + progress fields (modes b/c/d).
        self.goal_xyz: Optional[np.ndarray] = (
            np.asarray(goal_position, dtype=np.float64)
            if goal_position is not None else None
        )
        self.goal_tolerance_m = float(goal_tolerance_m)
        # Optional axis-aligned box tolerance in the same frame as
        # `goal_position`. When set, the GOAL_REACHED check uses
        # `all(|xyz - goal| <= half_extents)` instead of the Euclidean
        # `||xyz - goal|| <= goal_tolerance_m` sphere. The sphere config
        # is still honoured by the no-progress check (mode (b)/(d)) so
        # the legacy field's meaning doesn't silently change.
        self.goal_tolerance_half_extents: Optional[np.ndarray] = (
            np.asarray(goal_tolerance_half_extents, dtype=np.float64)
            if goal_tolerance_half_extents is not None else None
        )
        if self.goal_tolerance_half_extents is not None:
            if self.goal_tolerance_half_extents.shape != (3,):
                raise ValueError(
                    "goal_tolerance_half_extents must be a length-3 vector "
                    f"(got shape {self.goal_tolerance_half_extents.shape})"
                )
            if np.any(self.goal_tolerance_half_extents <= 0):
                raise ValueError(
                    "goal_tolerance_half_extents components must be > 0"
                )
        self.min_progress_window_s: Optional[float] = (
            float(min_progress_window_s)
            if min_progress_window_s is not None else None
        )
        self.min_progress_m = float(min_progress_m)
        # When eval_stop_mode is True, this criterion stops being a
        # MISS_GATE *classifier* and becomes a pure stop-signal:
        #   - skip mode (a) plane-crossing-outside-aperture (the bug)
        #   - on goal proximity, fire FailureType.GOAL_REACHED (success-stop)
        #     regardless of whether the drone "transited" by the plane-cross
        #     heuristic. Final MISS_GATE / SUCCESS classification is decided
        #     post-hoc by `falsify.safety.posthoc.classify_trajectory_posthoc`
        #     using the gate's MOCAP AABB.
        #   - stuck-pre-transit still fires MISS_GATE (stop signal); post-hoc
        #     reclassifies based on whether any state was inside the gate AABB.
        self.eval_stop_mode = bool(eval_stop_mode)
        # When `transit_aabb_*` are provided, the criterion tracks
        # AABB-containment of the current state per step. In
        # `eval_stop_mode`, `GOAL_REACHED` only fires *after* the drone has
        # entered the AABB at least once. Without that gate, an early
        # graze of the 10cm goal sphere before the gate crossing would cut
        # off a still-valid approach. With AABB transit tracked here at
        # runtime, post-hoc on `GOAL_REACHED` always implies SUCCESS.
        self._transit_aabb_min: Optional[np.ndarray] = (
            np.asarray(transit_aabb_min, dtype=np.float64)
            if transit_aabb_min is not None else None
        )
        self._transit_aabb_max: Optional[np.ndarray] = (
            np.asarray(transit_aabb_max, dtype=np.float64)
            if transit_aabb_max is not None else None
        )
        if (self._transit_aabb_min is None) != (self._transit_aabb_max is None):
            raise ValueError(
                "transit_aabb_min and transit_aabb_max must both be provided "
                "or both omitted"
            )
        # State.
        self._prev_xyz: Optional[np.ndarray] = None
        self._transited: bool = False
        # Flipped True the first step the drone's MOCAP position falls
        # inside `transit_aabb_*`. Used by eval_stop_mode to gate
        # GOAL_REACHED so we don't stop a pre-gate approach.
        self._ever_inside_aabb: bool = False
        # Time at which the drone first transited the aperture. Used by
        # the recovery seed sampler to scope its draw to post-transit
        # states when failure mode is GOAL_NOT_REACHED.
        self._transit_t: Optional[float] = None
        # Sliding window of (time, distance-to-current-target) for the
        # no-progress check. Target swaps from aperture centre → goal at
        # transit.
        from collections import deque
        self._progress: "deque[tuple[float, float]]" = deque()
        # Auto-inferred expected direction of aperture transit. Set on
        # the FIRST state to `-sign(initial s)` where s = (xyz - centre)·n.
        # See OrderedMissGateCriterion for the same pattern + caveats
        # (detour scenes need an explicit override which we don't plumb
        # here yet).
        self._expected_dir_sign: Optional[int] = None

    # ---- overrides -----------------------------------------------------

    def reset(self) -> None:
        self._prev_xyz = None
        self._transited = False
        self._transit_t = None
        self._ever_inside_aabb = False
        self._expected_dir_sign = None
        self._progress.clear()

    def check(self, state_in_frame: DroneState) -> Optional[Violation]:
        # Logic lives in `check_with_graph` so we can read prev_xyz
        # against the aperture's frame regardless of state frame.
        return None

    def check_with_graph(
        self, state: DroneState, frame_graph: FrameGraph
    ) -> Optional[Violation]:
        from .records import FailureType  # local import — avoid circular

        if state.pos.frame.name == self.operates_in_frame:
            xyz = np.asarray(state.pos.xyz, dtype=np.float64)
        else:
            xyz = np.asarray(
                frame_graph.convert(state.pos, to=self.operates_in_frame).xyz,
                dtype=np.float64,
            )

        prev = self._prev_xyz
        self._prev_xyz = xyz

        # ---- AABB containment latch -----------------------------------
        # If a transit AABB is configured, set `_ever_inside_aabb` the
        # first step the drone is inside it. `eval_stop_mode` reads this
        # flag (plus the current-step `inside_now` below) to decide
        # whether GOAL_REACHED is allowed to fire.
        inside_now = False
        if self._transit_aabb_min is not None:
            inside_now = bool(
                (xyz >= self._transit_aabb_min).all()
                and (xyz <= self._transit_aabb_max).all()
            )
            if inside_now and not self._ever_inside_aabb:
                self._ever_inside_aabb = True

        # ---- Auto-infer expected aperture-transit direction --------------
        # First-state-anchored. See OrderedMissGateCriterion for the
        # same logic + caveat about detour scenes.
        if self._expected_dir_sign is None:
            s0 = float(np.dot(xyz - self.centre, self.n))
            self._expected_dir_sign = (
                -int(np.sign(s0)) if s0 != 0.0 else +1
            )

        # ---- (a) plane-crossing check (run in BOTH eval and legacy mode) -
        # In eval mode we use it only for: (i) latching `_transited`
        # correctly (so the GOAL_REACHED gate above demands aperture
        # transit, not just AABB entry), and (ii) firing
        # wrong-direction MISS_GATE immediately. The
        # plane-cross-OUTSIDE-aperture MISS_GATE stays gated on
        # `not eval_stop_mode` (its known false-positive case —
        # approach arcs can clip the strict rectangle).
        if prev is not None and not self._transited:
            s0 = float(np.dot(prev - self.centre, self.n))
            s1 = float(np.dot(xyz - self.centre, self.n))
            if s0 != s1 and s0 * s1 <= 0.0:
                # Parametric crossing point.
                t = s0 / (s0 - s1)
                cross = prev + t * (xyz - prev)
                rel = cross - self.centre
                cu = float(np.dot(rel, self.u))
                cv = float(np.dot(rel, self.v))
                inside_u = abs(cu) <= self.hu - self.margin_m
                inside_v = abs(cv) <= self.hv - self.margin_m
                dir_sign = int(np.sign(s1 - s0))
                if inside_u and inside_v and dir_sign == self._expected_dir_sign:
                    self._transited = True
                    self._transit_t = float(state.t)
                    # Clear the progress window — distance metric is about
                    # to switch from "aperture centre" to "goal".
                    self._progress.clear()
                elif inside_u and inside_v:
                    # Inside the aperture but wrong direction — drone is
                    # backtracking through the gate. Always fires (no
                    # false-positive concern; direction is unambiguous).
                    return Violation(
                        description=(
                            f"drone crossed gate aperture in the WRONG "
                            f"direction (dir_sign={dir_sign:+d}, expected "
                            f"{self._expected_dir_sign:+d})"
                        ),
                        value=float(dir_sign),
                        threshold=float(self._expected_dir_sign),
                        failure_type=FailureType.MISS_GATE,
                        extra={"mode": "wrong_direction_aperture",
                               "dir_sign": dir_sign,
                               "expected_dir_sign": self._expected_dir_sign,
                               "cu": cu, "cv": cv},
                    )
                elif not self.eval_stop_mode:
                    worst = max(abs(cu) - self.hu, abs(cv) - self.hv)
                    return Violation(
                        description=(
                            f"drone crossed gate plane at "
                            f"(u={cu:+.3f}, v={cv:+.3f}) outside aperture "
                            f"half-widths (hu={self.hu:.3f}, hv={self.hv:.3f}) "
                            f"in frame {self.operates_in_frame!r}"
                        ),
                        value=float(worst),
                        threshold=0.0,
                        failure_type=FailureType.MISS_GATE,
                        extra={"mode": "plane_outside_aperture",
                               "cu": cu, "cv": cv,
                               "hu": self.hu, "hv": self.hv},
                    )

        # ---- (b/c/d) goal-aware checks — only when goal is configured ---
        if self.goal_xyz is None:
            return None

        dist_to_goal = float(np.linalg.norm(xyz - self.goal_xyz))

        # Box-tolerance has priority when configured: the drone must lie
        # inside the axis-aligned box `goal ± goal_tolerance_half_extents`
        # for the proximity stop to fire. Otherwise we use the legacy
        # Euclidean sphere `dist_to_goal <= goal_tolerance_m`.
        if self.goal_tolerance_half_extents is not None:
            delta = np.abs(xyz - self.goal_xyz)
            goal_prox_ok = bool(np.all(delta <= self.goal_tolerance_half_extents))
            goal_prox_desc = (
                f"drone inside goal box (|Δ|={delta.tolist()} ≤ "
                f"half_extents={self.goal_tolerance_half_extents.tolist()})"
            )
            goal_prox_value = float(np.max(delta - self.goal_tolerance_half_extents))
            goal_prox_threshold = 0.0
        else:
            goal_prox_ok = dist_to_goal <= self.goal_tolerance_m
            goal_prox_desc = (
                f"drone reached goal proximity (d={dist_to_goal:.3f} m "
                f"≤ {self.goal_tolerance_m:.3f} m)"
            )
            goal_prox_value = dist_to_goal
            goal_prox_threshold = self.goal_tolerance_m

        # ---- goal-proximity stop ---------------------------------------
        # In `eval_stop_mode`, fire GOAL_REACHED on goal proximity ONLY
        # when ALL three conditions hold:
        #   (i)  drone has APERTURE-TRANSITED the gate at least once
        #        (`_transited`) — was previously a loose AABB-entry
        #        latch, tightened to aperture-transit so a drone that
        #        clips an AABB edge without threading the rectangle
        #        can no longer satisfy SUCCESS.
        #   (ii) drone is NOT currently inside the AABB (post-transit
        #        hover, not mid-passage about to collide with a post).
        #   (iii) drone is in the goal-tolerance region (box or sphere).
        # If no transit AABB is configured we fall back to the legacy
        # "any goal-proximity stops" behavior (purely goal-distance based).
        gate_aabb_required = self._transit_aabb_min is not None
        transit_ok = (not gate_aabb_required) or self._transited
        outside_aabb = (not gate_aabb_required) or (not inside_now)
        if (self.eval_stop_mode
                and transit_ok
                and outside_aabb
                and goal_prox_ok):
            return Violation(
                description=(
                    f"{goal_prox_desc}; stopping rollout — "
                    f"post-hoc decides SUCCESS vs SKIPPED_GATE"
                ),
                value=goal_prox_value,
                threshold=goal_prox_threshold,
                failure_type=FailureType.GOAL_REACHED,
                extra={"mode": "goal_reached",
                       "dist_to_goal": dist_to_goal,
                       "transit_time": self._transit_t},
            )

        # (b) Legacy: reached goal proximity without transiting → MISS_GATE.
        # Disabled in eval_stop_mode — the new pre-transit goal-proximity
        # logic above lets the rollout keep running, and post-hoc decides
        # the final outcome. Leaving mode (b) on while loosening
        # `goal_tolerance_m` would cause it to fire from start because the
        # drone may begin inside the now-large tolerance sphere.
        if (not self.eval_stop_mode
                and not self._transited
                and dist_to_goal <= self.goal_tolerance_m):
            return Violation(
                description=(
                    f"drone reached goal proximity (d={dist_to_goal:.3f} m "
                    f"≤ {self.goal_tolerance_m:.3f} m) without crossing the "
                    f"gate aperture — skipped the gate"
                ),
                value=dist_to_goal,
                threshold=self.goal_tolerance_m,
                failure_type=FailureType.MISS_GATE,
                extra={"mode": "goal_without_transit",
                       "dist_to_goal": dist_to_goal},
            )

        # (f) Post-transit success — drone has transited AND is at goal.
        if self._transited and dist_to_goal <= self.goal_tolerance_m:
            # Nothing to report; mark progress window irrelevant.
            self._progress.clear()
            return None

        # (c)/(d) Stuck check. Track distance to the *relevant* target.
        # In eval_stop_mode the target is always the goal (no transit-based
        # target switching — transit detection is post-hoc).
        if self.min_progress_window_s is None:
            return None
        if self.eval_stop_mode:
            target_xyz = self.goal_xyz
        else:
            target_xyz = self.centre if not self._transited else self.goal_xyz
        dist_to_target = float(np.linalg.norm(xyz - target_xyz))
        t_now = float(state.t)
        self._progress.append((t_now, dist_to_target))
        # Trim entries older than the window.
        while (self._progress
               and t_now - self._progress[0][0] > self.min_progress_window_s):
            self._progress.popleft()
        # We need at least the full window of history before judging.
        if not self._progress:
            return None
        t_oldest, d_oldest = self._progress[0]
        if t_now - t_oldest < self.min_progress_window_s:
            return None
        progress = d_oldest - dist_to_target
        if progress < self.min_progress_m:
            if not self._transited:
                return Violation(
                    description=(
                        f"drone hasn't reduced distance to aperture by "
                        f"≥{self.min_progress_m:.3f} m in the last "
                        f"{self.min_progress_window_s:.1f} s "
                        f"(d_oldest={d_oldest:.3f} → d_now={dist_to_target:.3f}); "
                        f"stuck before the gate"
                    ),
                    value=float(progress),
                    threshold=self.min_progress_m,
                    failure_type=FailureType.MISS_GATE,
                    extra={"mode": "stuck_before_transit",
                           "dist_to_aperture": dist_to_target,
                           "dist_to_goal": dist_to_goal},
                )
            else:
                return Violation(
                    description=(
                        f"drone transited the gate but hasn't reduced "
                        f"distance to goal by ≥{self.min_progress_m:.3f} m in "
                        f"the last {self.min_progress_window_s:.1f} s "
                        f"(d_oldest={d_oldest:.3f} → d_now={dist_to_target:.3f}); "
                        f"post-gate hover failed"
                    ),
                    value=float(progress),
                    threshold=self.min_progress_m,
                    failure_type=FailureType.GOAL_NOT_REACHED,
                    extra={"mode": "stuck_after_transit",
                           "dist_to_goal": dist_to_target,
                           "transit_time": self._transit_t},
                )
        return None


class OrderedMissGateCriterion(SafetyCriterion):
    """Two-gate compositional aperture check with strict ordering.

    The drone must transit ``corners_1`` BEFORE any progress toward
    ``corners_2`` counts. Emits the same failure taxonomy as
    `MissGateCriterion` but with explicit gate identity in the violation
    extras (``which_gate``: ``"gate_1"`` / ``"gate_2"``).

    State progression:

      ``pre_gate_1`` → (transit gate 1) → ``pre_gate_2`` → (transit gate 2)
      → ``post_gate_2``.

    Failure modes:

    - ``MISS_GATE`` (``which_gate=gate_1``): plane crossed outside gate-1
      aperture, OR no progress to gate-1 centre, OR reached goal proximity
      before transiting gate 1 (skipped gate 1 entirely).
    - ``MISS_GATE`` (``which_gate=gate_2``): same three sub-modes for gate
      2 after gate-1 transit. Reached-goal-without-gate-2 also fires
      ``MISS_GATE`` (not ``GOAL_NOT_REACHED``) — skipping a gate is a
      gate-miss, not a goal failure.
    - ``GOAL_NOT_REACHED``: both gates transited but the drone stalled
      before reaching ``goal_position`` (stuck-after-gate-2).

    The progress window's distance metric tracks the *current* target,
    which switches from gate-1 centre → gate-2 centre → goal as transits
    happen. Each switch clears the progress window so a slow approach to
    one target doesn't trigger a false stuck-report for the next.

    `eval_stop_mode` and `transit_aabb_*` are intentionally NOT yet
    supported here — compositional eval-mode success classification is a
    later concern (Phase 3.2).
    """

    name: str = "ordered_miss_gate"

    def __init__(
        self,
        corners_1: np.ndarray,
        corners_2: np.ndarray,
        *,
        frame_name: str = "mocap",
        margin_m: float = 0.0,
        goal_position: Optional[np.ndarray] = None,
        goal_tolerance_m: float = 0.30,
        min_progress_window_s: Optional[float] = None,
        min_progress_m: float = 0.05,
        eval_stop_mode: bool = False,
        transit_aabb_1_min: Optional[np.ndarray] = None,
        transit_aabb_1_max: Optional[np.ndarray] = None,
        transit_aabb_2_min: Optional[np.ndarray] = None,
        transit_aabb_2_max: Optional[np.ndarray] = None,
    ) -> None:
        c1 = np.asarray(corners_1, dtype=np.float64)
        c2 = np.asarray(corners_2, dtype=np.float64)
        self.corners_1 = c1
        self.corners_2 = c2
        (self.centre_1, self.u_1, self.v_1, self.n_1,
         self.hu_1, self.hv_1) = _aperture_geometry(c1)
        (self.centre_2, self.u_2, self.v_2, self.n_2,
         self.hu_2, self.hv_2) = _aperture_geometry(c2)
        self.operates_in_frame = frame_name
        self.margin_m = float(margin_m)
        self.goal_xyz: Optional[np.ndarray] = (
            np.asarray(goal_position, dtype=np.float64)
            if goal_position is not None else None
        )
        self.goal_tolerance_m = float(goal_tolerance_m)
        self.min_progress_window_s: Optional[float] = (
            float(min_progress_window_s)
            if min_progress_window_s is not None else None
        )
        self.min_progress_m = float(min_progress_m)
        # eval_stop_mode mirrors MissGateCriterion's design — disables
        # the in-flight plane-cross-outside-aperture check (which was
        # killing compositional trials whose natural arc clipped either
        # gate's plane outside the strict rectangle), and instead uses
        # per-gate AABB transit latches to drive (a) goal-proximity stop
        # gating and (b) the stuck-target switcher. See the module
        # docstring on eval-mode semantics; configure per scene via
        # `safety_cfg.ordered_miss_gate.eval_stop_mode`.
        self.eval_stop_mode = bool(eval_stop_mode)
        self._transit_aabb_1_min: Optional[np.ndarray] = (
            np.asarray(transit_aabb_1_min, dtype=np.float64)
            if transit_aabb_1_min is not None else None
        )
        self._transit_aabb_1_max: Optional[np.ndarray] = (
            np.asarray(transit_aabb_1_max, dtype=np.float64)
            if transit_aabb_1_max is not None else None
        )
        self._transit_aabb_2_min: Optional[np.ndarray] = (
            np.asarray(transit_aabb_2_min, dtype=np.float64)
            if transit_aabb_2_min is not None else None
        )
        self._transit_aabb_2_max: Optional[np.ndarray] = (
            np.asarray(transit_aabb_2_max, dtype=np.float64)
            if transit_aabb_2_max is not None else None
        )
        if (self._transit_aabb_1_min is None) != (self._transit_aabb_1_max is None):
            raise ValueError("transit_aabb_1_min/max must both be provided or both omitted")
        if (self._transit_aabb_2_min is None) != (self._transit_aabb_2_max is None):
            raise ValueError("transit_aabb_2_min/max must both be provided or both omitted")
        # State.
        self._prev_xyz: Optional[np.ndarray] = None
        self._transited_1: bool = False
        self._transited_2: bool = False
        self._transit_t_1: Optional[float] = None
        self._transit_t_2: Optional[float] = None
        # First time the drone EXITED each AABB after first entering it.
        # Used by recovery-seed scoping: a between_gates failure should
        # only sample safe states from states with t ≥ _transit_t_1_exit
        # (i.e., the drone has cleared the gate, not just entered it).
        # Without the exit guard, the sampler can pick a mid-transit
        # state inside the AABB and the recovery trim jumps to
        # pre_gate_2 from there — visually "skipping" the gate.
        self._transit_t_1_exit: Optional[float] = None
        self._transit_t_2_exit: Optional[float] = None
        # AABB latches (used by eval_stop_mode; harmless when off).
        self._ever_inside_aabb_1: bool = False
        self._ever_inside_aabb_2: bool = False
        # Expected direction of aperture transit per gate. Auto-
        # inferred from the first state's signed distance to each
        # gate plane — drone is expected to cross AWAY from its
        # initial side. ``expected_dir_sign_X`` is the expected
        # sign of (s_after - s_before) at an inside crossing.
        # +1 means drone goes from -s side to +s side; -1 means the
        # reverse. Used to detect wrong-direction crossings (drone
        # going backwards through aperture) and fire MISS_GATE
        # immediately. Detour scenes that don't start on the
        # "wrong side" of the gate will trip this — for those, an
        # explicit override should be plumbed via constructor (TODO).
        self._expected_dir_sign_1: Optional[int] = None
        self._expected_dir_sign_2: Optional[int] = None
        # First time the drone crossed each gate's PLANE in the expected
        # direction (regardless of whether it was inside the aperture
        # rectangle). Stamped on any forward sign-flip — so a drone
        # that clips OUTSIDE the aperture but still ends up on the far
        # side of the plane gets recorded. Used by the recovery-seed
        # sampler to scope a pre-gate failure's seed to states BEFORE
        # the bypass: otherwise the planner has to first reverse back
        # north of the gate, then re-approach south through it (the
        # "doubles back" pathology).
        self._first_plane_cross_t_1: Optional[float] = None
        self._first_plane_cross_t_2: Optional[float] = None
        from collections import deque
        self._progress: "deque[tuple[float, float]]" = deque()

    # ---- overrides -----------------------------------------------------

    def reset(self) -> None:
        self._prev_xyz = None
        self._transited_1 = False
        self._transited_2 = False
        self._transit_t_1 = None
        self._transit_t_2 = None
        self._transit_t_1_exit = None
        self._transit_t_2_exit = None
        self._ever_inside_aabb_1 = False
        self._ever_inside_aabb_2 = False
        self._expected_dir_sign_1 = None
        self._expected_dir_sign_2 = None
        self._first_plane_cross_t_1 = None
        self._first_plane_cross_t_2 = None
        self._progress.clear()

    def _phase(self) -> str:
        """Compositional rollout phase, derived from APERTURE-TRANSIT
        latches (`_transited_*`) in both eval and legacy modes.

        Using the AABB latches instead (the prior behavior) misclassified
        gate-1 collisions: a drone clipping a post of gate_1 was inside
        the AABB at the moment of impact, so `_ever_inside_aabb_1=True`
        → phase=between_gates → the recovery planner trimmed all
        pre-gate-1 waypoints and replanned straight toward gate_2,
        skipping gate_1 entirely. Aperture transit is the right signal:
        only a true aperture crossing should move the phase forward.
        """
        t1 = self._transited_1
        t2 = self._transited_2
        if not t1:
            return "pre_gate_1"
        if not t2:
            return "between_gates"
        return "post_gate_2"

    def check(self, state_in_frame: DroneState) -> Optional[Violation]:
        # All the logic needs `prev_xyz`; see `check_with_graph`.
        return None

    def phase_snapshot(self) -> dict:
        """Snapshot of the criterion's compositional progress at the
        current step. The ``FailureDetector`` merges this into every
        ``FailureRecord.extra`` (without overwriting keys already on
        the firing violation), so failures from OTHER criteria — a
        ``PointCloudCollisionCriterion`` clip post-gate-1, for
        instance — still carry the gate-1 transit info the recovery
        sampler needs to scope post-gate-1 safe-history correctly.

        Keys are namespaced to avoid colliding with other criteria.
        """
        return {
            "phase": self._phase(),
            "transit_time_1": self._transit_t_1,
            "transit_time_2": self._transit_t_2,
            "transit_time_1_exit": self._transit_t_1_exit,
            "transit_time_2_exit": self._transit_t_2_exit,
            "ever_inside_aabb_1": self._ever_inside_aabb_1,
            "ever_inside_aabb_2": self._ever_inside_aabb_2,
            "first_plane_cross_t_1": self._first_plane_cross_t_1,
            "first_plane_cross_t_2": self._first_plane_cross_t_2,
        }

    def check_with_graph(
        self, state: DroneState, frame_graph: FrameGraph
    ) -> Optional[Violation]:
        from .records import FailureType

        if state.pos.frame.name == self.operates_in_frame:
            xyz = np.asarray(state.pos.xyz, dtype=np.float64)
        else:
            xyz = np.asarray(
                frame_graph.convert(state.pos, to=self.operates_in_frame).xyz,
                dtype=np.float64,
            )

        prev = self._prev_xyz
        self._prev_xyz = xyz

        # ---- Auto-infer expected aperture-transit direction --------------
        # On the FIRST state, compute the signed distance to each gate's
        # plane. The drone is expected to cross AWAY from that initial
        # side — so `expected_dir_sign = -sign(initial_s)`. Caveat: this
        # only works for trajectories whose initial position is on the
        # correct entry side; scenes that DETOUR around the gate first
        # (e.g. center_from_right starts at +y but should approach from
        # -y) will register wrong-direction on the eventual crossing.
        # For those, an explicit override is the right fix (TODO);
        # today the existing post-hoc `expected_dy_sign` demotion
        # handles them.
        if self._expected_dir_sign_1 is None:
            s0_1 = float(np.dot(xyz - self.centre_1, self.n_1))
            self._expected_dir_sign_1 = (
                -int(np.sign(s0_1)) if s0_1 != 0.0 else +1
            )
        if self._expected_dir_sign_2 is None:
            s0_2 = float(np.dot(xyz - self.centre_2, self.n_2))
            self._expected_dir_sign_2 = (
                -int(np.sign(s0_2)) if s0_2 != 0.0 else +1
            )

        # ---- AABB containment latches (eval_stop_mode) -------------------
        # Order is enforced: gate-2 only latches *after* gate-1 has latched,
        # so a drone that enters the center AABB before the original gate
        # AABB doesn't accidentally satisfy gate-2 first. The latches
        # exist independent of eval_stop_mode (cheap to compute) but only
        # the eval_stop_mode branches read them.
        inside_now_1 = False
        inside_now_2 = False
        if self._transit_aabb_1_min is not None:
            inside_now_1 = bool(
                (xyz >= self._transit_aabb_1_min).all()
                and (xyz <= self._transit_aabb_1_max).all()
            )
            if inside_now_1 and not self._ever_inside_aabb_1:
                self._ever_inside_aabb_1 = True
                if self.eval_stop_mode:
                    self._progress.clear()
            # Track FIRST exit from gate_1 AABB AFTER a real aperture
            # transit. Gating on `_transited_1` prevents AABB-clip-only
            # trajectories (drone hit a post without threading the
            # aperture) from stamping a fake exit time that the
            # recovery-seed sampler would then scope on.
            if (self._transited_1 and not inside_now_1
                    and self._transit_t_1_exit is None):
                self._transit_t_1_exit = float(state.t)
        if self._transit_aabb_2_min is not None:
            inside_now_2 = bool(
                (xyz >= self._transit_aabb_2_min).all()
                and (xyz <= self._transit_aabb_2_max).all()
            )
            if (inside_now_2 and not self._ever_inside_aabb_2
                    and self._ever_inside_aabb_1):
                self._ever_inside_aabb_2 = True
                if self.eval_stop_mode:
                    self._progress.clear()
            if (self._transited_2 and not inside_now_2
                    and self._transit_t_2_exit is None):
                self._transit_t_2_exit = float(state.t)

        # ---- Plane-crossing check for the currently-active gate ----------
        # The plane-cross-OUTSIDE-aperture MISS_GATE violation is
        # disabled in eval_stop_mode (same reasoning as
        # MissGateCriterion: the natural arc of an approach can clip
        # the plane outside the strict rectangle without indicating
        # policy failure). But the aperture-transit LATCH itself runs
        # in BOTH modes — the eval-mode GOAL_REACHED gate below uses it
        # to demand the drone actually threaded each gate, not just
        # clipped an AABB edge.
        def _try_transit(prev_xyz, cur_xyz, centre, u, v, n, hu, hv):
            """Return (crossed, inside, cu, cv, dir_sign) for the plane
            swept between ``prev → cur``. ``crossed`` is True iff signed
            distance changed sign; ``inside`` is True iff the crossing
            point lies inside the (margin-shrunk) aperture rectangle;
            ``dir_sign`` is +1 if the drone moved from -s side → +s side
            (i.e. with the normal), -1 if reverse. 0 only when not
            crossed."""
            s0 = float(np.dot(prev_xyz - centre, n))
            s1 = float(np.dot(cur_xyz - centre, n))
            if s0 == s1 or s0 * s1 > 0.0:
                return False, False, 0.0, 0.0, 0
            t = s0 / (s0 - s1)
            cross = prev_xyz + t * (cur_xyz - prev_xyz)
            rel = cross - centre
            cu = float(np.dot(rel, u))
            cv = float(np.dot(rel, v))
            inside_u = abs(cu) <= hu - self.margin_m
            inside_v = abs(cv) <= hv - self.margin_m
            dir_sign = int(np.sign(s1 - s0))
            return True, (inside_u and inside_v), cu, cv, dir_sign

        # Check BOTH gate planes unconditionally (until each is
        # transited). Out-of-order — drone crosses gate_2's aperture
        # before transiting gate_1 — fires immediate MISS_GATE rather
        # than being silently ignored.
        if prev is not None and not self._transited_1:
            crossed, inside, cu, cv, dir_sign = _try_transit(
                prev, xyz, self.centre_1, self.u_1, self.v_1, self.n_1,
                self.hu_1, self.hv_1,
            )
            if crossed:
                # Stamp the first forward plane-crossing time (in OR
                # out of aperture) so the recovery seed sampler can
                # avoid sampling from past gate-1 when phase=pre_gate_1.
                if (self._first_plane_cross_t_1 is None
                        and dir_sign == self._expected_dir_sign_1):
                    self._first_plane_cross_t_1 = float(state.t)
                if inside and dir_sign == self._expected_dir_sign_1:
                    self._transited_1 = True
                    self._transit_t_1 = float(state.t)
                    self._progress.clear()   # distance target switches to gate 2
                elif inside:
                    # Inside the aperture but wrong direction — drone
                    # is going back through the gate. Fires regardless
                    # of eval_stop_mode (unambiguous failure, no
                    # false-positive concern).
                    return Violation(
                        description=(
                            f"drone crossed gate-1 aperture in the WRONG "
                            f"direction (dir_sign={dir_sign:+d}, expected "
                            f"{self._expected_dir_sign_1:+d})"
                        ),
                        value=float(dir_sign),
                        threshold=float(self._expected_dir_sign_1),
                        failure_type=FailureType.MISS_GATE,
                        extra={"which_gate": "gate_1",
                               "phase": self._phase(),
                               "mode": "wrong_direction_aperture",
                               "dir_sign": dir_sign,
                               "expected_dir_sign": self._expected_dir_sign_1,
                               "cu": cu, "cv": cv},
                    )
                elif not self.eval_stop_mode:
                    worst = max(abs(cu) - self.hu_1, abs(cv) - self.hv_1)
                    return Violation(
                        description=(
                            f"drone crossed gate-1 plane at (u={cu:+.3f}, "
                            f"v={cv:+.3f}) outside aperture half-widths "
                            f"(hu={self.hu_1:.3f}, hv={self.hv_1:.3f})"
                        ),
                        value=float(worst),
                        threshold=0.0,
                        failure_type=FailureType.MISS_GATE,
                        extra={"which_gate": "gate_1",
                               "phase": self._phase(),
                               "mode": "plane_outside_aperture",
                               "cu": cu, "cv": cv,
                               "hu": self.hu_1, "hv": self.hv_1},
                    )

        if prev is not None and not self._transited_2:
            crossed, inside, cu, cv, dir_sign = _try_transit(
                prev, xyz, self.centre_2, self.u_2, self.v_2, self.n_2,
                self.hu_2, self.hv_2,
            )
            if crossed:
                if (self._first_plane_cross_t_2 is None
                        and dir_sign == self._expected_dir_sign_2):
                    self._first_plane_cross_t_2 = float(state.t)
                if inside and not self._transited_1:
                    # Out-of-order: drone is threading gate_2 before
                    # gate_1. Treat as a MISS — the policy went to the
                    # wrong gate / wrong order.
                    return Violation(
                        description=(
                            "drone crossed gate-2 aperture BEFORE "
                            "transiting gate-1 (out of order)"
                        ),
                        value=0.0,
                        threshold=0.0,
                        failure_type=FailureType.MISS_GATE,
                        extra={"which_gate": "gate_2",
                               "phase": self._phase(),
                               "mode": "out_of_order_aperture",
                               "dir_sign": dir_sign,
                               "cu": cu, "cv": cv},
                    )
                if inside and dir_sign == self._expected_dir_sign_2:
                    self._transited_2 = True
                    self._transit_t_2 = float(state.t)
                    self._progress.clear()   # distance target switches to goal
                elif inside:
                    return Violation(
                        description=(
                            f"drone crossed gate-2 aperture in the WRONG "
                            f"direction (dir_sign={dir_sign:+d}, expected "
                            f"{self._expected_dir_sign_2:+d})"
                        ),
                        value=float(dir_sign),
                        threshold=float(self._expected_dir_sign_2),
                        failure_type=FailureType.MISS_GATE,
                        extra={"which_gate": "gate_2",
                               "phase": self._phase(),
                               "mode": "wrong_direction_aperture",
                               "dir_sign": dir_sign,
                               "expected_dir_sign": self._expected_dir_sign_2,
                               "cu": cu, "cv": cv},
                    )
                elif not self.eval_stop_mode:
                    worst = max(abs(cu) - self.hu_2, abs(cv) - self.hv_2)
                    return Violation(
                        description=(
                            f"drone crossed gate-2 plane at (u={cu:+.3f}, "
                            f"v={cv:+.3f}) outside aperture half-widths "
                            f"(hu={self.hu_2:.3f}, hv={self.hv_2:.3f})"
                        ),
                        value=float(worst),
                        threshold=0.0,
                        failure_type=FailureType.MISS_GATE,
                        extra={"which_gate": "gate_2",
                               "phase": self._phase(),
                               "mode": "plane_outside_aperture",
                               "cu": cu, "cv": cv,
                               "hu": self.hu_2, "hv": self.hv_2},
                    )

        # ---- Goal-aware checks — only when goal_position is configured ----
        if self.goal_xyz is None:
            return None
        dist_to_goal = float(np.linalg.norm(xyz - self.goal_xyz))

        # ---- eval_stop_mode: GOAL_REACHED only when the drone has
        # actually transited BOTH apertures (not just clipped the AABB
        # edges), is currently outside both AABBs, and within goal
        # tolerance. The aperture-transit latches `_transited_1/2` are
        # now maintained in eval_stop_mode as well (see the plane-cross
        # check above), so a path that touches both AABBs without
        # threading the apertures won't satisfy this gate.
        if self.eval_stop_mode:
            both_transited = self._transited_1 and self._transited_2
            outside_now = (not inside_now_1) and (not inside_now_2)
            if (both_transited and outside_now
                    and dist_to_goal <= self.goal_tolerance_m):
                return Violation(
                    description=(
                        f"drone reached goal proximity (d={dist_to_goal:.3f} m "
                        f"≤ {self.goal_tolerance_m:.3f} m) after both gate "
                        "aperture transits; stopping rollout"
                    ),
                    value=dist_to_goal,
                    threshold=self.goal_tolerance_m,
                    failure_type=FailureType.GOAL_REACHED,
                    extra={"phase": "post_gate_2",
                           "mode": "goal_reached_both_gates",
                           "dist_to_goal": dist_to_goal,
                           "both_aperture_transited": True,
                           "transit_time_1": self._transit_t_1,
                           "transit_time_2": self._transit_t_2,
                           "transit_time_1_exit": self._transit_t_1_exit,
                           "transit_time_2_exit": self._transit_t_2_exit},
                )
        else:
            # ---- Legacy skip check: reached goal proximity before
            # transiting both → MISS_GATE. Disabled in eval_stop_mode
            # for the same reason as MissGateCriterion mode (b): with
            # loosened tolerance and a start point near goal it would
            # fire spuriously; in eval mode we let the rollout run to
            # the natural stop conditions and decide post-hoc.
            if dist_to_goal <= self.goal_tolerance_m:
                if not self._transited_1:
                    return Violation(
                        description=(
                            f"drone reached goal proximity (d={dist_to_goal:.3f} m "
                            f"≤ {self.goal_tolerance_m:.3f} m) without crossing "
                            f"gate 1 — skipped both gates"
                        ),
                        value=dist_to_goal,
                        threshold=self.goal_tolerance_m,
                        failure_type=FailureType.MISS_GATE,
                        extra={"which_gate": "gate_1",
                               "phase": self._phase(),
                               "mode": "goal_without_transit",
                               "dist_to_goal": dist_to_goal},
                    )
                if not self._transited_2:
                    return Violation(
                        description=(
                            f"drone reached goal proximity (d={dist_to_goal:.3f} m "
                            f"≤ {self.goal_tolerance_m:.3f} m) after gate 1 but "
                            f"without crossing gate 2 — skipped gate 2"
                        ),
                        value=dist_to_goal,
                        threshold=self.goal_tolerance_m,
                        failure_type=FailureType.MISS_GATE,
                        extra={"which_gate": "gate_2",
                               "phase": self._phase(),
                               "mode": "goal_without_transit",
                               "dist_to_goal": dist_to_goal,
                               "transit_time": self._transit_t_1},
                    )
                # Both transited → success; nothing to emit.
                self._progress.clear()
                return None

        # ---- Stuck check against the current target ---------------------
        # In eval_stop_mode the AABB latches drive the target switch;
        # otherwise the legacy plane-cross transit flags do. AABB-based
        # target switching is intentional: a drone that clipped past
        # gate_1 (touched the AABB but didn't thread the aperture) IS
        # geographically past gate_1, so the next target is gate_2 —
        # the failure tag stays "between_gates". The recovery-seed
        # sampler handles the "don't seed from before the bypass"
        # concern via `first_plane_cross_t_1` (see orchestrator).
        if self.min_progress_window_s is None:
            return None
        if self.eval_stop_mode and self._transit_aabb_1_min is not None:
            t1, t2 = self._ever_inside_aabb_1, self._ever_inside_aabb_2
        else:
            t1, t2 = self._transited_1, self._transited_2
        if not t1:
            target_xyz = self.centre_1
            target_label = "gate_1_centre"
        elif not t2:
            target_xyz = self.centre_2
            target_label = "gate_2_centre"
        else:
            target_xyz = self.goal_xyz
            target_label = "goal"
        dist_to_target = float(np.linalg.norm(xyz - target_xyz))
        t_now = float(state.t)
        self._progress.append((t_now, dist_to_target))
        while (self._progress
               and t_now - self._progress[0][0] > self.min_progress_window_s):
            self._progress.popleft()
        if not self._progress:
            return None
        t_oldest, d_oldest = self._progress[0]
        if t_now - t_oldest < self.min_progress_window_s:
            return None
        progress = d_oldest - dist_to_target
        if progress < self.min_progress_m:
            if not t1:
                ftype = FailureType.MISS_GATE
                which = "gate_1"
                phase = "pre_gate_1"
                desc_tail = "stuck before gate 1"
            elif not t2:
                ftype = FailureType.MISS_GATE
                which = "gate_2"
                phase = "between_gates"
                desc_tail = "stuck between gates 1 and 2"
            else:
                ftype = FailureType.GOAL_NOT_REACHED
                which = "post_gate_2"
                phase = "post_gate_2"
                desc_tail = "stuck after gate 2; goal not reached"
            return Violation(
                description=(
                    f"drone hasn't reduced distance to {target_label} by "
                    f"≥{self.min_progress_m:.3f} m in the last "
                    f"{self.min_progress_window_s:.1f} s "
                    f"(d_oldest={d_oldest:.3f} → d_now={dist_to_target:.3f}); "
                    f"{desc_tail}"
                ),
                value=float(progress),
                threshold=self.min_progress_m,
                failure_type=ftype,
                extra={"which_gate": which,
                       "phase": phase,
                       "mode": "stuck",
                       "target": target_label,
                       "dist_to_target": dist_to_target,
                       "dist_to_goal": dist_to_goal,
                       "transit_time_1": self._transit_t_1,
                       "transit_time_2": self._transit_t_2,
                       "transit_time_1_exit": self._transit_t_1_exit,
                       "transit_time_2_exit": self._transit_t_2_exit},
            )
        return None
