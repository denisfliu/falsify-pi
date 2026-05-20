"""Scene-time Gaussian edits.

A *scene edit* is a declarative description of a Gaussian-level
modification applied to a loaded gsplat: select a subset of Gaussians
by region, then transform their `means` and `quats` in place. The edit
lives in the scene YAML (data, not code) and runs once after the
``GSplatRenderer`` finishes loading — subsequent renders pick up the
mutated model automatically.

Why this layer exists
---------------------
Splat-MOVER demonstrates that mutating ``pipeline.model.means[mask]``
and ``pipeline.model.quats[mask]`` is the entire mechanic of
"moving an object" in a trained gsplat. We don't need their CLIP-based
object selection because we already have clean MOCAP-frame AABBs for
the gates from ``objects_summary.json``. So selection here is just
"Gaussians whose mean (in NS) lies inside the AABB lifted to NS"; the
transform is a rigid one composed in MOCAP, also lifted to NS via the
active `FrameGraph`.

Frame contract
--------------
The user authors edits in MOCAP (the human-readable frame). The
applier converts to NS once per edit, then writes into the gsplat's
native NS-frame ``means`` / ``quats``. The composition is:

  T_edit_ns = T_mocap→ns ∘ T_edit_mocap ∘ T_ns→mocap

The `T_mocap→ns` Sim3's scale cancels the `T_ns→mocap` Sim3's inverse
scale, so the resulting NS transform's scale is unity (within float
precision) — semantically an SE3, applied to NS-frame Gaussians.

The same MOCAP-frame transform is also exposed so visualization tools
(`inspect_scene_plotly`, `visualize_waypoints`) can apply it to the
scene_objects PLYs and keep the rendered view consistent with the
inspector view.

YAML schema (see ``configs/scenes/center_gate.yaml`` for a worked example)::

  scene_edits:
    - name: move_gate
      type: rigid_transform_aabb
      target_aabb_frame: mocap
      target_aabb_min: [0.51, 0.27, 0.07]
      target_aabb_max: [1.21, 1.12, 1.97]
      transform:
        source_anchor: [0.86, 0.69, 0.07]    # where the AABB region currently sits
        target_anchor: [2.5, -0.25, 0.0]     # where it should end up
        source_normal: [0.749, 0.663, 0.0]   # "forward" direction of the object now
        target_normal: [0.0, -1.0, 0.0]      # ... and where it should point
      # Optional: scene_objects (by name) that the visualizer should also
      # transform so the inspector matches what the renderer will produce.
      applies_to_scene_objects: [gate]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Protocol, Sequence

import numpy as np

from falsify.geometry import FrameGraph, SE3, Sim3
from falsify.geometry.frames import Frame


# ---------------------------------------------------------------------------
# Public interfaces
# ---------------------------------------------------------------------------


class SceneEdit(Protocol):
    """Declarative scene edit. Implementations must expose the MOCAP-frame
    transform and the AABB mask, so the applier and the visualizer can both
    consume the same spec."""

    name: str
    type: str
    applies_to_scene_objects: tuple[str, ...]

    def transform_in(self, frame_name: str, frame_graph: FrameGraph) -> SE3 | Sim3:
        """Return the transform from the *source* pose to the *target* pose,
        expressed as an operator on points in ``frame_name``. Composed with
        the FrameGraph if the edit was authored in a different frame.
        """

    def aabb_corners_in(self, frame_name: str, frame_graph: FrameGraph) -> np.ndarray:
        """Return the 8 AABB corner positions in ``frame_name``. The applier
        takes their per-axis min/max to mask Gaussians."""


# ---------------------------------------------------------------------------
# rigid_transform_aabb
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Box:
    """Axis-aligned bounding box in the edit's authored frame."""
    min: np.ndarray  # (3,)
    max: np.ndarray  # (3,)

    def __post_init__(self):
        mn = np.asarray(self.min, dtype=np.float64)
        mx = np.asarray(self.max, dtype=np.float64)
        if mn.shape != (3,) or mx.shape != (3,):
            raise ValueError(f"box min/max must be (3,); got {mn.shape}/{mx.shape}")
        object.__setattr__(self, "min", mn)
        object.__setattr__(self, "max", mx)

    def corners(self) -> np.ndarray:
        mn, mx = self.min, self.max
        return np.array([
            [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
            [mn[0], mx[1], mn[2]], [mx[0], mx[1], mn[2]],
            [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
            [mn[0], mx[1], mx[2]], [mx[0], mx[1], mx[2]],
        ])


@dataclass(frozen=True)
class _OrientedBox:
    """Box with center + half-extents + yaw about the authored frame's +z.

    Used for selecting Gaussians along narrow non-axis-aligned regions
    (the gate base running diagonally between the two posts is the
    motivating example). The yaw is applied about the box's centre in
    the xy-plane only — gates and similar objects are vertical, so we
    don't bother with pitch/roll.
    """
    center: np.ndarray        # (3,) — xyz centre in the authored frame
    half_extents: np.ndarray  # (3,) — positive
    yaw: float = 0.0          # radians, CCW about +z

    def __post_init__(self):
        c = np.asarray(self.center, dtype=np.float64)
        h = np.asarray(self.half_extents, dtype=np.float64)
        if c.shape != (3,) or h.shape != (3,):
            raise ValueError(f"oriented box center/half_extents must be (3,)")
        if (h <= 0).any():
            raise ValueError(f"oriented box half_extents must be positive; got {h.tolist()}")
        object.__setattr__(self, "center", c)
        object.__setattr__(self, "half_extents", h)
        object.__setattr__(self, "yaw", float(self.yaw))

    def _R(self) -> np.ndarray:
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def corners(self) -> np.ndarray:
        """8 corners of the rotated box in the authored frame."""
        hx, hy, hz = self.half_extents
        local = np.array([
            [-hx, -hy, -hz], [+hx, -hy, -hz],
            [-hx, +hy, -hz], [+hx, +hy, -hz],
            [-hx, -hy, +hz], [+hx, -hy, +hz],
            [-hx, +hy, +hz], [+hx, +hy, +hz],
        ])
        return (self._R() @ local.T).T + self.center

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Boolean mask of which points (in the authored frame) lie inside."""
        c, s = np.cos(-self.yaw), np.sin(-self.yaw)
        R_inv = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        local = (R_inv @ (points - self.center).T).T
        return (np.abs(local) <= self.half_extents).all(axis=1)


@dataclass(frozen=True)
class RigidTransformAABB:
    """A rigid (yaw-only by default) transform applied to Gaussians selected
    by AABB membership.

    The selection mask is::

        broad   = target_aabb  \ (exclude_aabbs ∪ oriented_exclude_aabbs)
        precise = include_aabbs ∪ oriented_include_aabbs
        move    = broad ∪ precise

    ``target_aabb_*`` is a BROAD, exclude-subject bracket — useful for a
    loose "anything inside this box that isn't on the table" selection.
    ``include_aabbs`` (and ``oriented_include_aabbs``) are PRECISE,
    hand-curated regions that override excludes — they're treated as
    ground truth and always move. Containment is evaluated in the
    ``target_aabb_frame`` directly, so the authored bounds are respected
    exactly (no corner-rebracketing approximation).
    """
    name: str
    target_aabb_frame: str
    target_aabb_min: np.ndarray  # (3,)
    target_aabb_max: np.ndarray  # (3,)
    source_anchor: np.ndarray    # (3,)
    target_anchor: np.ndarray    # (3,)
    source_normal: np.ndarray    # (3,) — projected to xy for yaw-only solve
    target_normal: np.ndarray    # (3,)
    transform_frame: str = "mocap"
    applies_to_scene_objects: tuple[str, ...] = ()
    include_aabbs: tuple[_Box, ...] = ()
    exclude_aabbs: tuple[_Box, ...] = ()
    oriented_include_aabbs: tuple[_OrientedBox, ...] = ()
    oriented_exclude_aabbs: tuple[_OrientedBox, ...] = ()
    type: str = "rigid_transform_aabb"

    def __post_init__(self):
        for fld in (
            "target_aabb_min", "target_aabb_max",
            "source_anchor", "target_anchor",
            "source_normal", "target_normal",
        ):
            v = np.asarray(getattr(self, fld), dtype=np.float64)
            if v.shape != (3,):
                raise ValueError(f"{self.name}: {fld} must be (3,), got {v.shape}")
            object.__setattr__(self, fld, v)

    # ---- transform solver ----------------------------------------------

    def transform_in_authored_frame(self) -> SE3:
        """SE3 in the authored ``transform_frame`` that maps the source pose
        to the target pose. Solved as a yaw rotation about z (gates are
        vertical) plus a translation, so we don't fight rotations that would
        tilt a fundamentally-vertical structure."""
        # Yaw about z that aligns source_normal's xy with target_normal's xy.
        s_xy = self.source_normal[:2]
        t_xy = self.target_normal[:2]
        s_norm = np.linalg.norm(s_xy)
        t_norm = np.linalg.norm(t_xy)
        if s_norm < 1e-9 or t_norm < 1e-9:
            raise ValueError(
                f"{self.name}: source/target normals must have non-zero xy "
                f"component (got source={self.source_normal}, target={self.target_normal})"
            )
        theta = math.atan2(t_xy[1], t_xy[0]) - math.atan2(s_xy[1], s_xy[0])
        c, s = math.cos(theta), math.sin(theta)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        t = self.target_anchor - R @ self.source_anchor
        # Build SE3 with src==dst==transform_frame (an in-frame nudge).
        f = Frame(self.transform_frame)
        return SE3(R=R, t=t, src=f, dst=f)

    def transform_in(self, frame_name: str, frame_graph: FrameGraph) -> SE3 | Sim3:
        """Return the same rigid edit, expressed as a transform on points in
        ``frame_name``. Composition: ``T_to ∘ T_edit_authored ∘ T_from``.
        """
        T_edit = self.transform_in_authored_frame()
        if frame_name == self.transform_frame:
            return T_edit
        T_from = frame_graph.transform(frame_name, self.transform_frame)
        T_to = frame_graph.transform(self.transform_frame, frame_name)
        return T_to @ T_edit @ T_from

    # ---- AABB ----------------------------------------------------------

    def _corners_authored(self) -> np.ndarray:
        mn, mx = self.target_aabb_min, self.target_aabb_max
        return np.array([
            [mn[0], mn[1], mn[2]],
            [mx[0], mn[1], mn[2]],
            [mn[0], mx[1], mn[2]],
            [mx[0], mx[1], mn[2]],
            [mn[0], mn[1], mx[2]],
            [mx[0], mn[1], mx[2]],
            [mn[0], mx[1], mx[2]],
            [mx[0], mx[1], mx[2]],
        ])

    def aabb_corners_in(self, frame_name: str, frame_graph: FrameGraph) -> np.ndarray:
        corners = self._corners_authored()
        if frame_name == self.target_aabb_frame:
            return corners
        T = frame_graph.transform(self.target_aabb_frame, frame_name)
        # Apply Sim3 / SE3 to each corner.
        s = getattr(T, "s", 1.0)
        return (s * (T.R @ corners.T) + T.t[:, None]).T

    def aabb_min_max_in(self, frame_name: str, frame_graph: FrameGraph) -> tuple[np.ndarray, np.ndarray]:
        c = self.aabb_corners_in(frame_name, frame_graph)
        return c.min(axis=0), c.max(axis=0)

    def include_aabbs_in(
        self, frame_name: str, frame_graph: FrameGraph,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """All inclusion AABBs (main + ``include_aabbs``) bracketed in
        ``frame_name``. Each authored AABB is lifted by its 8 corners and
        re-axis-aligned."""
        boxes = [_Box(min=self.target_aabb_min, max=self.target_aabb_max),
                 *self.include_aabbs]
        out: list[tuple[np.ndarray, np.ndarray]] = []
        for box in boxes:
            c = box.corners()
            if frame_name != self.target_aabb_frame:
                T = frame_graph.transform(self.target_aabb_frame, frame_name)
                s = getattr(T, "s", 1.0)
                c = (s * (T.R @ c.T) + T.t[:, None]).T
            out.append((c.min(axis=0), c.max(axis=0)))
        return out

    def exclude_aabbs_in(
        self, frame_name: str, frame_graph: FrameGraph,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return each exclusion AABB as an axis-aligned min/max pair in
        ``frame_name``. Each authored AABB is lifted by its 8 corners and
        re-axis-aligned in the target frame."""
        out: list[tuple[np.ndarray, np.ndarray]] = []
        for box in self.exclude_aabbs:
            c = box.corners()
            if frame_name != self.target_aabb_frame:
                T = frame_graph.transform(self.target_aabb_frame, frame_name)
                s = getattr(T, "s", 1.0)
                c = (s * (T.R @ c.T) + T.t[:, None]).T
            out.append((c.min(axis=0), c.max(axis=0)))
        return out


# ---------------------------------------------------------------------------
# Loader registry
# ---------------------------------------------------------------------------


_LOADER_REGISTRY: dict[str, Callable[[dict], SceneEdit]] = {}


def register_edit_loader(type_name: str, loader: Callable[[dict], SceneEdit]) -> None:
    if type_name in _LOADER_REGISTRY:
        raise ValueError(f"scene-edit loader {type_name!r} already registered")
    _LOADER_REGISTRY[type_name] = loader


def _load_rigid_transform_aabb(spec: dict) -> RigidTransformAABB:
    tr = spec.get("transform", {})
    includes = tuple(
        _Box(min=e["min"], max=e["max"])
        for e in (spec.get("include_aabbs") or [])
    )
    excludes = tuple(
        _Box(min=e["min"], max=e["max"])
        for e in (spec.get("exclude_aabbs") or [])
    )
    ori_inc = tuple(
        _OrientedBox(
            center=np.asarray(e["center"], dtype=np.float64),
            half_extents=np.asarray(e["half_extents"], dtype=np.float64),
            yaw=float(e.get("yaw", 0.0)),
        )
        for e in (spec.get("oriented_include_aabbs") or [])
    )
    ori_exc = tuple(
        _OrientedBox(
            center=np.asarray(e["center"], dtype=np.float64),
            half_extents=np.asarray(e["half_extents"], dtype=np.float64),
            yaw=float(e.get("yaw", 0.0)),
        )
        for e in (spec.get("oriented_exclude_aabbs") or [])
    )
    return RigidTransformAABB(
        name=spec["name"],
        target_aabb_frame=spec.get("target_aabb_frame", "mocap"),
        target_aabb_min=np.asarray(spec["target_aabb_min"], dtype=np.float64),
        target_aabb_max=np.asarray(spec["target_aabb_max"], dtype=np.float64),
        source_anchor=np.asarray(tr["source_anchor"], dtype=np.float64),
        target_anchor=np.asarray(tr["target_anchor"], dtype=np.float64),
        source_normal=np.asarray(tr["source_normal"], dtype=np.float64),
        target_normal=np.asarray(tr["target_normal"], dtype=np.float64),
        transform_frame=spec.get("transform_frame", "mocap"),
        applies_to_scene_objects=tuple(spec.get("applies_to_scene_objects", []) or []),
        include_aabbs=includes,
        exclude_aabbs=excludes,
        oriented_include_aabbs=ori_inc,
        oriented_exclude_aabbs=ori_exc,
    )


register_edit_loader("rigid_transform_aabb", _load_rigid_transform_aabb)


# ---------------------------------------------------------------------------
# duplicate_aabb
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DuplicateAABB(RigidTransformAABB):
    """Copy a region of Gaussians and place the copy at a new pose.

    Identical field shape to :class:`RigidTransformAABB` (same selection
    semantics: ``broad ∪ precise − exclude``; same rigid-transform solve
    via ``source_anchor / target_anchor / source_normal / target_normal``).
    Only the **write-back** differs: instead of mutating the masked
    Gaussians in place, the appliers append a transformed copy of the
    masked subset so both the original *and* the moved copy end up in
    the scene.

    Used by the compositional eval scenes (``left_and_center``,
    ``right_and_center``) to keep the original gate where the trained
    checkpoint expects it *and* add a second gate at the center anchor —
    without authoring a new gsplat asset.

    YAML::

      scene_edits:
        - name: duplicate_left_gate_to_center
          type: duplicate_aabb
          target_aabb_frame: mocap
          target_aabb_min: [0.51, 0.27, 0.07]
          target_aabb_max: [1.21, 1.12, 1.97]
          transform:
            source_anchor: [0.86, 0.69, 0.07]
            target_anchor: [2.5, -0.25, 0.0]
            source_normal: [0.749, 0.663, 0.0]
            target_normal: [0.0, -1.0, 0.0]
          applies_to_scene_objects: [gate]

    Caveats
    -------
    - Spherical-harmonic coefficients (``features_dc`` / ``features_rest``)
      are copied without rotation when the pipeline applier expands the
      gsplat. For a true rigid transform of the *lighting environment*
      they would be rotated, but for our compositional scenes (drone is
      the only thing in motion and the gate's view-dependent appearance
      is shallow) this is an acceptable approximation — the same one
      ``RigidTransformAABB`` already makes for the moved gate in
      ``center_gate.yaml``.
    - Output of the appliers is **longer** than the input. Callers that
      assume preserved length (none in falsify today) must be updated.
    """
    type: str = "duplicate_aabb"


def _load_duplicate_aabb(spec: dict) -> DuplicateAABB:
    tr = spec.get("transform", {})
    includes = tuple(
        _Box(min=e["min"], max=e["max"])
        for e in (spec.get("include_aabbs") or [])
    )
    excludes = tuple(
        _Box(min=e["min"], max=e["max"])
        for e in (spec.get("exclude_aabbs") or [])
    )
    ori_inc = tuple(
        _OrientedBox(
            center=np.asarray(e["center"], dtype=np.float64),
            half_extents=np.asarray(e["half_extents"], dtype=np.float64),
            yaw=float(e.get("yaw", 0.0)),
        )
        for e in (spec.get("oriented_include_aabbs") or [])
    )
    ori_exc = tuple(
        _OrientedBox(
            center=np.asarray(e["center"], dtype=np.float64),
            half_extents=np.asarray(e["half_extents"], dtype=np.float64),
            yaw=float(e.get("yaw", 0.0)),
        )
        for e in (spec.get("oriented_exclude_aabbs") or [])
    )
    return DuplicateAABB(
        name=spec["name"],
        target_aabb_frame=spec.get("target_aabb_frame", "mocap"),
        target_aabb_min=np.asarray(spec["target_aabb_min"], dtype=np.float64),
        target_aabb_max=np.asarray(spec["target_aabb_max"], dtype=np.float64),
        source_anchor=np.asarray(tr["source_anchor"], dtype=np.float64),
        target_anchor=np.asarray(tr["target_anchor"], dtype=np.float64),
        source_normal=np.asarray(tr["source_normal"], dtype=np.float64),
        target_normal=np.asarray(tr["target_normal"], dtype=np.float64),
        transform_frame=spec.get("transform_frame", "mocap"),
        applies_to_scene_objects=tuple(spec.get("applies_to_scene_objects", []) or []),
        include_aabbs=includes,
        exclude_aabbs=excludes,
        oriented_include_aabbs=ori_inc,
        oriented_exclude_aabbs=ori_exc,
    )


register_edit_loader("duplicate_aabb", _load_duplicate_aabb)


def load_scene_edits(scene_cfg: dict) -> list[SceneEdit]:
    """Parse ``scene_cfg["scene_edits"]`` into typed SceneEdit objects."""
    out: list[SceneEdit] = []
    for spec in scene_cfg.get("scene_edits") or []:
        loader = _LOADER_REGISTRY.get(spec["type"])
        if loader is None:
            raise ValueError(
                f"unknown scene-edit type {spec['type']!r}; "
                f"available: {sorted(_LOADER_REGISTRY)}"
            )
        out.append(loader(spec))
    return out


# ---------------------------------------------------------------------------
# Gaussian appliers
# ---------------------------------------------------------------------------


def _mask_inside_aabb(points: np.ndarray, mn: np.ndarray, mx: np.ndarray) -> np.ndarray:
    return ((points >= mn) & (points <= mx)).all(axis=1)


def _rotation_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → unit quaternion (xyzw). Numerically-stable shortcut
    for det(R)=+1 inputs (we only build those above)."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return np.array([qx, qy, qz, qw])


def _quat_product_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 ⊗ q2 with q in wxyz layout. q can be (4,) or (N, 4)."""
    if q1.ndim == 1:
        q1 = q1[None, :]
    if q2.ndim == 1:
        q2 = q2[None, :]
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=1)


def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    out = np.empty_like(q)
    out[0] = q[3]
    out[1:] = q[:3]
    return out


def apply_edits_to_arrays(
    means_ns: np.ndarray,         # (N, 3) float
    quats_wxyz: Optional[np.ndarray],  # (N, 4) float in WXYZ — splatfacto convention
    edits: Sequence[SceneEdit],
    frame_graph: FrameGraph,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Pure-numpy applier: returns new (means, quats) with edits applied.

    No torch / no gsplat dependency — used by the visualizer to keep
    inspector PLYs in sync with what the rendered model will show.
    """
    means_ns = np.asarray(means_ns, dtype=np.float64).copy()
    new_quats = None if quats_wxyz is None else np.asarray(quats_wxyz, dtype=np.float64).copy()

    means_authored_cache: dict[str, np.ndarray] = {}

    def _means_in(frame_name: str) -> np.ndarray:
        if frame_name == "ns":
            return means_ns
        if frame_name in means_authored_cache:
            return means_authored_cache[frame_name]
        T = frame_graph.transform("ns", frame_name)
        s = getattr(T, "s", 1.0)
        out = (s * (T.R @ means_ns.T)).T + T.t
        means_authored_cache[frame_name] = out
        return out

    for edit in edits:
        # Mask semantics: the BROAD main AABB is loose-fit and is subject to
        # ``exclude_aabbs`` / ``oriented_exclude_aabbs``. The PRECISE inclusions
        # (``include_aabbs`` + ``oriented_include_aabbs``) are treated as
        # hand-curated ground truth — they always win over excludes.
        # Containment is evaluated in the edit's authored frame so the user's
        # box bounds are respected exactly (no re-bracketing approximation).
        authored = _means_in(edit.target_aabb_frame)

        broad = _mask_inside_aabb(authored, edit.target_aabb_min, edit.target_aabb_max)
        for ex in edit.exclude_aabbs:
            broad &= ~_mask_inside_aabb(authored, ex.min, ex.max)
        for box in edit.oriented_exclude_aabbs:
            broad &= ~box.contains(authored)

        precise = np.zeros(means_ns.shape[0], dtype=bool)
        for inc in edit.include_aabbs:
            precise |= _mask_inside_aabb(authored, inc.min, inc.max)
        for box in edit.oriented_include_aabbs:
            precise |= box.contains(authored)

        mask = broad | precise
        if not mask.any():
            continue
        T_ns = edit.transform_in("ns", frame_graph)
        # T_ns is Sim3 (with scale ~1 after cancellation) or SE3.
        R = T_ns.R
        s = getattr(T_ns, "s", 1.0)
        t = T_ns.t
        transformed_means = (s * (R @ means_ns[mask].T)).T + t
        if new_quats is not None:
            q_R_xyzw = _rotation_to_quat_xyzw(R)
            q_R_wxyz = _xyzw_to_wxyz(q_R_xyzw)
            transformed_quats = _quat_product_wxyz(q_R_wxyz, new_quats[mask])
        if isinstance(edit, DuplicateAABB):
            # Copy semantics: original Gaussians stay; transformed copies
            # are appended.
            means_ns = np.concatenate([means_ns, transformed_means], axis=0)
            if new_quats is not None:
                new_quats = np.concatenate([new_quats, transformed_quats], axis=0)
        else:
            # Move semantics: original Gaussians are overwritten in place.
            means_ns[mask] = transformed_means
            if new_quats is not None:
                new_quats[mask] = transformed_quats
        # Invalidate cache — means moved or appended.
        means_authored_cache.clear()

    return means_ns, new_quats


def apply_edits_to_scene_object(
    object_name: str,
    points_in_authored_frame: np.ndarray,
    edits: Sequence[SceneEdit],
    frame_graph: FrameGraph,
) -> np.ndarray:
    """Apply every edit whose ``applies_to_scene_objects`` includes
    ``object_name`` to a point cloud in the edit's authored frame.

    This is the visualizer-side counterpart to ``apply_edits_to_pipeline``
    — it keeps the inspector / waypoint visualizer in sync with what the
    rendered model will produce. AABB masking is **not** applied here; if
    the user named a scene object in ``applies_to_scene_objects``, the
    whole cloud is treated as that object.

    Length semantics
    ----------------
    - ``RigidTransformAABB`` (move): output has the same length as input.
    - ``DuplicateAABB`` (copy): output grows by the input length each time
      a matching duplicate fires — the transformed copy is appended so the
      cloud reflects both the original and the moved gate.
    """
    pts = np.asarray(points_in_authored_frame, dtype=np.float64).copy()
    for edit in edits:
        if object_name not in edit.applies_to_scene_objects:
            continue
        # PLY-level: the user is asserting that the entire named cloud IS
        # this object. Whole-cloud rigid transform; no AABB filtering, no
        # exclusion — those are Gaussian-level concerns handled in
        # ``apply_edits_to_pipeline`` / ``apply_edits_to_arrays``.
        T = edit.transform_in_authored_frame()
        transformed = (T.R @ pts.T).T + T.t
        if isinstance(edit, DuplicateAABB):
            # Append a transformed copy; originals stay.
            pts = np.concatenate([pts, transformed], axis=0)
        else:
            pts = transformed
    return pts


def apply_edits_to_pipeline(pipeline, edits: Sequence[SceneEdit], frame_graph: FrameGraph) -> int:
    """Apply edits in place to ``pipeline.model``.

    For ``RigidTransformAABB`` (move) the Gaussian count stays constant —
    selected rows of ``means`` / ``quats`` are rewritten in place.

    For ``DuplicateAABB`` (copy) the Gaussian count grows. The selected
    rows of ``means`` / ``quats`` are *appended* (with the rigid transform
    applied), and every other per-Gaussian field on the model
    (``scales``, ``opacities``, ``features_dc``, ``features_rest``) is
    expanded by appending the same source rows verbatim. The new tensors
    are wrapped in fresh ``nn.Parameter`` instances and assigned back to
    the model — same mechanism splatfacto already uses for density
    refinement, so the renderer keeps working.

    Returns the total number of Gaussians touched: moves + duplications.
    Uses torch when the model fields are tensors; otherwise falls back to
    numpy.
    """
    n_touched = 0
    try:
        import torch
        from torch import nn
    except ImportError:
        torch = None  # type: ignore
        nn = None  # type: ignore

    model = pipeline.model
    means = model.means
    quats = getattr(model, "quats", None)
    use_torch = torch is not None and hasattr(means, "device")

    if use_torch:
        means_np = means.detach().cpu().numpy().astype(np.float64)
        quats_np = quats.detach().cpu().numpy().astype(np.float64) if quats is not None else None
    else:
        means_np = np.asarray(means, dtype=np.float64).copy()
        quats_np = None if quats is None else np.asarray(quats, dtype=np.float64).copy()

    # ORIGINAL Gaussian indices that need their per-Gaussian fields
    # (scales / opacities / features) duplicated, in submission order.
    # ``means_np`` and ``quats_np`` grow inside the loop; for the other
    # fields we apply all expansions in one pass at the end.
    duplicate_source_indices: list[np.ndarray] = []

    authored_cache: dict[str, np.ndarray] = {}

    def _means_in_pipeline(frame_name: str) -> np.ndarray:
        if frame_name == "ns":
            return means_np
        if frame_name in authored_cache:
            return authored_cache[frame_name]
        T = frame_graph.transform("ns", frame_name)
        s = getattr(T, "s", 1.0)
        out = (s * (T.R @ means_np.T)).T + T.t
        authored_cache[frame_name] = out
        return out

    for edit in edits:
        # See apply_edits_to_arrays for the semantics: broad main AABB is
        # exclude-subject; precise include_aabbs / oriented_include_aabbs
        # override excludes. Containment is evaluated in the authored frame
        # so the user's box bounds are respected exactly.
        authored = _means_in_pipeline(edit.target_aabb_frame)

        broad = _mask_inside_aabb(authored, edit.target_aabb_min, edit.target_aabb_max)
        for ex in edit.exclude_aabbs:
            broad &= ~_mask_inside_aabb(authored, ex.min, ex.max)
        for box in edit.oriented_exclude_aabbs:
            broad &= ~box.contains(authored)

        precise = np.zeros(means_np.shape[0], dtype=bool)
        for inc in edit.include_aabbs:
            precise |= _mask_inside_aabb(authored, inc.min, inc.max)
        for box in edit.oriented_include_aabbs:
            precise |= box.contains(authored)

        mask = broad | precise
        if not mask.any():
            continue
        T_ns = edit.transform_in("ns", frame_graph)
        R, s, t = T_ns.R, getattr(T_ns, "s", 1.0), T_ns.t
        transformed_means = (s * (R @ means_np[mask].T)).T + t
        if quats_np is not None:
            q_R_xyzw = _rotation_to_quat_xyzw(R)
            q_R_wxyz = _xyzw_to_wxyz(q_R_xyzw)
            transformed_quats = _quat_product_wxyz(q_R_wxyz, quats_np[mask])
        if isinstance(edit, DuplicateAABB):
            # Append transformed copies. Originals stay in place.
            idx = np.where(mask)[0].astype(np.int64)
            duplicate_source_indices.append(idx)
            means_np = np.concatenate([means_np, transformed_means], axis=0)
            if quats_np is not None:
                quats_np = np.concatenate([quats_np, transformed_quats], axis=0)
        else:
            # In-place move.
            means_np[mask] = transformed_means
            if quats_np is not None:
                quats_np[mask] = transformed_quats
        n_touched += int(mask.sum())
        authored_cache.clear()

    grew = len(duplicate_source_indices) > 0

    if use_torch:
        device = means.device
        dtype = means.dtype
        # nerfstudio splatfacto v2 / sagesplat keep all per-Gaussian
        # tensors inside ``model.gauss_params`` (a ``nn.ParameterDict``);
        # ``model.means`` etc. are properties that delegate to it. When the
        # container is present we MUST mutate via the dict — overwriting
        # ``model.means`` directly clashes with the property setter. For
        # legacy splatfacto layouts that don't have gauss_params we fall
        # back to direct attribute assignment.
        gp = getattr(model, "gauss_params", None)
        if grew:
            # Replace nn.Parameters wholesale — same mechanism splatfacto
            # uses for density refinement, so the render path picks up the
            # new count automatically.
            new_means_t = torch.tensor(means_np, device=device, dtype=dtype)
            new_quats_t = None
            if quats_np is not None and quats is not None:
                new_quats_t = torch.tensor(
                    quats_np, device=quats.device, dtype=quats.dtype,
                )
            full_idx = np.concatenate(duplicate_source_indices).astype(np.int64)
            full_idx_t = torch.as_tensor(full_idx, dtype=torch.long, device=device)
            if gp is not None:
                # Write every per-Gaussian tensor (incl. sage extras like
                # ``affordance`` / ``clip_embeds``) via the ParameterDict.
                # Features etc. are NOT rotated — the same approximation
                # RigidTransformAABB already makes for moved Gaussians.
                gp["means"] = nn.Parameter(
                    new_means_t, requires_grad=gp["means"].requires_grad,
                )
                if new_quats_t is not None:
                    gp["quats"] = nn.Parameter(
                        new_quats_t, requires_grad=gp["quats"].requires_grad,
                    )
                for fname, orig in list(gp.items()):
                    if fname in ("means", "quats"):
                        continue
                    appended = orig.detach().index_select(
                        0, full_idx_t.to(orig.device),
                    )
                    new_t = torch.cat([orig.detach(), appended], dim=0)
                    gp[fname] = nn.Parameter(
                        new_t, requires_grad=orig.requires_grad,
                    )
            else:
                # Legacy layout: attributes directly on the model.
                setattr(model, "means", nn.Parameter(
                    new_means_t, requires_grad=means.requires_grad,
                ))
                if new_quats_t is not None:
                    setattr(model, "quats", nn.Parameter(
                        new_quats_t, requires_grad=quats.requires_grad,
                    ))
                for fname in ("scales", "opacities",
                              "features_dc", "features_rest"):
                    orig = getattr(model, fname, None)
                    if orig is None:
                        continue
                    appended = orig.detach().index_select(
                        0, full_idx_t.to(orig.device),
                    )
                    new_t = torch.cat([orig.detach(), appended], dim=0)
                    setattr(model, fname, nn.Parameter(
                        new_t, requires_grad=orig.requires_grad,
                    ))
        else:
            # Same-size in-place copy — preserves any optimizer / cache
            # state that's keyed by tensor identity. Works the same for
            # both ParameterDict-backed and direct-attribute layouts
            # because ``model.means`` resolves to the underlying tensor.
            new_means = torch.tensor(means_np, device=device, dtype=dtype)
            with torch.no_grad():
                means.data.copy_(new_means)
            if quats_np is not None and quats is not None:
                new_quats = torch.tensor(quats_np, device=quats.device, dtype=quats.dtype)
                with torch.no_grad():
                    quats.data.copy_(new_quats)
    else:
        # Best-effort write-back for non-torch models (mostly for tests).
        try:
            if grew:
                # NumPy fallback: replace the arrays outright; the test
                # models don't have features/scales/opacities to worry about.
                model.means = means_np
                if quats_np is not None:
                    model.quats = quats_np
                # Expand auxiliary per-Gaussian arrays if present.
                full_idx = np.concatenate(duplicate_source_indices).astype(np.int64)
                for fname in ("scales", "opacities", "features_dc", "features_rest"):
                    orig = getattr(model, fname, None)
                    if orig is None:
                        continue
                    orig_np = np.asarray(orig)
                    setattr(model, fname, np.concatenate(
                        [orig_np, orig_np[full_idx]], axis=0,
                    ))
            else:
                model.means[:] = means_np
                if quats_np is not None:
                    model.quats[:] = quats_np
        except Exception:
            pass

    return n_touched
