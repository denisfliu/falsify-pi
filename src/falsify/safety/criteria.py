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


class MissGateCriterion(SafetyCriterion):
    """Fire when the drone crosses the gate plane outside the aperture.

    The aperture is a 3D rectangle defined by four corners in some named
    frame. Internally we compute:

      - plane normal ``n``           = unit normal of the plane through the corners,
      - plane offset ``d``           = ``n · centre`` (any corner works),
      - two in-plane axes ``u, v``   = orthonormal axes along adjacent edges,
      - half-widths ``hu, hv``       = aperture extents along ``u`` and ``v``,
      - centre                       = mean of the four corners.

    Each call records ``state.pos`` as the previous-state cache. Between
    consecutive states the segment ``p0 → p1`` is intersected with the
    plane (signed distance changes sign). If a crossing is found, the
    intersection point is projected into ``(u, v)`` coords; if either is
    outside its half-width, fire `FailureType.MISS_GATE`. Inside ⇒ the
    drone went through the gate; mark `_transited` so we don't fire on
    subsequent re-crossings (e.g. recovery loops back through).

    The criterion needs ``check_with_graph`` to convert *both* the prev
    and current state into the aperture's frame, so we override that.
    """

    name: str = "miss_gate"

    def __init__(
        self,
        corners: np.ndarray,
        *,
        frame_name: str = "mocap",
        margin_m: float = 0.0,
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
        self._prev_xyz: Optional[np.ndarray] = None
        self._transited: bool = False

    # ---- overrides -----------------------------------------------------

    def reset(self) -> None:
        self._prev_xyz = None
        self._transited = False

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
        if prev is None or self._transited:
            return None

        s0 = float(np.dot(prev - self.centre, self.n))
        s1 = float(np.dot(xyz - self.centre, self.n))
        if s0 * s1 > 0.0:
            return None  # same side, no crossing
        if s0 == s1:
            return None  # both exactly on the plane — no clean crossing

        # Parametric crossing point: prev + t * (xyz - prev), t = s0 / (s0 - s1)
        t = s0 / (s0 - s1)
        cross = prev + t * (xyz - prev)
        rel = cross - self.centre
        cu = float(np.dot(rel, self.u))
        cv = float(np.dot(rel, self.v))
        inside_u = abs(cu) <= self.hu - self.margin_m
        inside_v = abs(cv) <= self.hv - self.margin_m
        if inside_u and inside_v:
            self._transited = True
            return None

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
            extra={"cu": cu, "cv": cv, "hu": self.hu, "hv": self.hv},
        )
