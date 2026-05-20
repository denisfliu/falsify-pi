"""Per-trial cost functions for CEM.

Each cost is a **continuous** scalar so CEM elite-selection has gradient
information even on trials that didn't actually fail. Higher cost = more
failure-like; CEM picks the top-K trials by cost.

The four supported failure types map to costs as follows:

| target              | cost (higher = worse, i.e. more failure-like)                          |
|---------------------|------------------------------------------------------------------------|
| ``COLLISION_GATE``  | ``− min_t signed_dist(drone_OBB(t), gate_cloud)``                      |
| ``COLLISION_OTHER`` | ``− min_t signed_dist(drone_OBB(t), other_cloud)`` − big-penalty if    |
|                     | a gate collision also happened (we don't want to reward gate-as-table) |
| ``MISS_GATE``       | per-plane-crossing offset beyond the aperture rectangle; 0 if the      |
|                     | trajectory never crossed the gate plane                                |
| ``GOAL_NOT_REACHED``| ``‖final_position − goal‖`` in NED                                     |

Signed-distance convention: positive outside the OBB, negative inside —
exactly the same sign convention SDFs use. So a collision contributes a
negative ``min_t``; flipping the sign in the cost lifts it above any
non-collision trial.

The scorer is intentionally decoupled from the failure detector: it reads
the rollout NPZ persisted by ``run_eval_campaign.py`` and re-derives the
metric. The detector's binary verdict isn't enough — we need the
continuous signal for non-failing trials too.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as _R

from falsify.geometry import PointCloud
from falsify.io import build_frame_graph, load_yaml


# ---------------------------------------------------------------------------
# OBB distance — the only non-trivial geometric primitive in this module.
# ---------------------------------------------------------------------------


def obb_to_points_signed_distance(
    centre_world: np.ndarray,
    R_world_from_body: np.ndarray,
    half_extents: np.ndarray,
    centre_offset_body: np.ndarray,
    points_world: np.ndarray,
) -> np.ndarray:
    """Signed Euclidean distance from each point to the OBB surface.

    Positive outside, negative inside. The signed-distance convention
    matches typical SDFs: ``distance(point, box) > 0`` ⇔ point is exterior.

    Implementation follows Inigo Quilez's classic box SDF:

    - Express points in body coordinates.
    - Per-axis residual ``q = |p_body| − half_extents``.
    - Exterior distance: ``‖max(q, 0)‖₂`` (the standard Euclidean distance
      to the nearest face).
    - Interior penetration: ``min(max(q), 0)`` (negative, deepest face).
    - Combined SDF: ``‖max(q, 0)‖₂ + min(max(q), 0)``.

    Parameters
    ----------
    centre_world
        Drone body-origin position in world coords, shape (3,).
    R_world_from_body
        Rotation matrix taking body-frame vectors into world, shape (3, 3).
    half_extents
        Body-FRD half-widths, shape (3,).
    centre_offset_body
        Shift from body origin to OBB centre, in body coords, shape (3,).
    points_world
        Points to evaluate, shape (N, 3).

    Returns
    -------
    np.ndarray
        Signed distances, shape (N,).
    """
    points_world = np.asarray(points_world, dtype=np.float64)
    if points_world.size == 0:
        return np.empty((0,), dtype=np.float64)
    obb_centre_world = centre_world + R_world_from_body @ centre_offset_body
    rel = points_world - obb_centre_world         # (N, 3) world
    rel_body = rel @ R_world_from_body            # equivalent to (R.T @ rel.T).T
    q = np.abs(rel_body) - half_extents           # (N, 3)
    exterior = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    interior = np.minimum(np.max(q, axis=1), 0.0)
    return exterior + interior


# ---------------------------------------------------------------------------
# Scene context — derived once per scene-key, replayed across trials.
# ---------------------------------------------------------------------------


@dataclass
class SceneContext:
    """Pre-loaded scene data needed to score any trial against this scene.

    Gate point cloud + aperture corners + (optional) goal are kept in
    MOCAP so we can apply per-trial gate-rigid-transform deltas before
    converting to NED. The other-objects point cloud doesn't move with
    the gate, so it's cached in NED directly.
    """

    drone_body_half_extents: np.ndarray
    drone_body_center_offset: np.ndarray
    drone_bounding_radius: float
    gate_points_mocap: np.ndarray          # (Ng, 3)
    other_points_ned: np.ndarray           # (No, 3); may be empty
    aperture_corners_mocap: np.ndarray     # (4, 3)
    goal_mocap: Optional[np.ndarray]       # (3,) or None
    gate_anchor_mocap: np.ndarray          # for the rigid-transform helper
    _fg: object                            # FrameGraph (kept for mocap→ned conversions)

    @classmethod
    def from_yamls(cls, scene_yaml: Path, safety_yaml: Path) -> "SceneContext":
        scene_yaml = Path(scene_yaml)
        safety_yaml = Path(safety_yaml)
        scene_cfg = load_yaml(scene_yaml)
        safety_cfg = load_yaml(safety_yaml)
        fg = build_frame_graph(scene_cfg, base_path=scene_yaml.parent)

        # Drone body (already validated by smoke_test's loader; we just
        # need the raw shapes).
        body_cfg = safety_cfg["drone_body"]
        half_extents = np.asarray(body_cfg["half_extents"], dtype=np.float64)
        center_offset = np.asarray(
            body_cfg.get("center_offset_body", [0.0, 0.0, 0.0]),
            dtype=np.float64,
        )
        bounding_radius = float(
            np.linalg.norm(half_extents) + np.linalg.norm(center_offset)
        )

        # Gate cloud (mocap) + other cloud (ned). We re-implement a slim
        # version of smoke_test._build_collision_criterion's loader here
        # because we don't want to instantiate a SafetyCriterion just to
        # get at its points.
        coll_cfg = safety_cfg.get("collision") or {}
        gate_names = set(coll_cfg.get("gate_objects", ["gate"]))
        other_names = set(coll_cfg.get("other_objects", []))
        scene_objects = scene_cfg.get("scene_objects") or []
        by_name = {entry["name"]: entry for entry in scene_objects}

        from falsify.visualization import read_ply
        gate_pts_mocap: list[np.ndarray] = []
        other_pts_ned: list[np.ndarray] = []
        for name, entry in by_name.items():
            if name not in (gate_names | other_names):
                continue
            ply_path = Path(entry["ply"])
            if not ply_path.is_absolute():
                ply_path = (scene_yaml.parent / ply_path).resolve()
            entry_frame = fg.frame(entry["frame"])
            pc = read_ply(ply_path, entry_frame)
            if name in gate_names:
                if entry["frame"] != "mocap":
                    raise ValueError(
                        f"CEM scorer requires gate scene_object {name!r} to "
                        f"live in 'mocap'; got {entry['frame']!r}"
                    )
                gate_pts_mocap.append(np.asarray(pc.points, dtype=np.float64))
            else:
                pc_ned = fg.convert(pc, to="ned")
                other_pts_ned.append(np.asarray(pc_ned.points, dtype=np.float64))

        if not gate_pts_mocap:
            raise ValueError(
                "no gate scene_object found — CEM gate-collision cost needs at "
                "least one PLY in safety.collision.gate_objects"
            )
        gate_points_mocap = np.vstack(gate_pts_mocap)
        other_points_ned = (
            np.vstack(other_pts_ned) if other_pts_ned else np.empty((0, 3))
        )

        # Aperture corners + goal (mocap).
        miss_cfg = safety_cfg.get("miss_gate") or {}
        corners = miss_cfg.get("corners")
        if corners is None:
            raise ValueError(
                "safety.miss_gate.corners is required by CEM scorer"
            )
        if miss_cfg.get("corners_frame", "mocap") != "mocap":
            raise ValueError(
                "CEM scorer requires miss_gate.corners_frame=='mocap'"
            )
        aperture_corners_mocap = np.asarray(corners, dtype=np.float64)
        goal_mocap = miss_cfg.get("goal_position")
        goal_mocap = (
            np.asarray(goal_mocap, dtype=np.float64) if goal_mocap is not None else None
        )

        # Gate anchor (for gate rigid transform). The perturbation reads
        # this from scene.gate_region.anchor; we mirror it.
        region = scene_cfg.get("gate_region") or {}
        anchor = region.get("anchor")
        if anchor is None:
            raise ValueError(
                "scene.gate_region.anchor required (used as rotation centre "
                "for gate perturbations)"
            )
        gate_anchor_mocap = np.asarray(anchor, dtype=np.float64)

        return cls(
            drone_body_half_extents=half_extents,
            drone_body_center_offset=center_offset,
            drone_bounding_radius=bounding_radius,
            gate_points_mocap=gate_points_mocap,
            other_points_ned=other_points_ned,
            aperture_corners_mocap=aperture_corners_mocap,
            goal_mocap=goal_mocap,
            gate_anchor_mocap=gate_anchor_mocap,
            _fg=fg,
        )

    # ---- per-trial transforms ----------------------------------------

    def _mocap_to_ned(self, pts_mocap: np.ndarray) -> np.ndarray:
        pc = PointCloud(points=np.asarray(pts_mocap, dtype=np.float64),
                        frame=self._fg.frame("mocap"))
        pc_ned = self._fg.convert(pc, to="ned")
        return np.asarray(pc_ned.points, dtype=np.float64)

    def gate_cloud_ned(self, gate_deltas_mocap: Optional[dict]) -> np.ndarray:
        moved = _apply_gate_rigid_transform_mocap(
            self.gate_points_mocap, self.gate_anchor_mocap, gate_deltas_mocap,
        )
        return self._mocap_to_ned(moved)

    def aperture_corners_ned(
        self, gate_deltas_mocap: Optional[dict],
    ) -> np.ndarray:
        moved = _apply_gate_rigid_transform_mocap(
            self.aperture_corners_mocap, self.gate_anchor_mocap, gate_deltas_mocap,
        )
        return self._mocap_to_ned(moved)

    def goal_ned(self) -> Optional[np.ndarray]:
        # Goal does NOT move with the gate (it's a task-level point).
        if self.goal_mocap is None:
            return None
        return self._mocap_to_ned(self.goal_mocap.reshape(1, 3))[0]


def _apply_gate_rigid_transform_mocap(
    points_mocap: np.ndarray,
    anchor_mocap: np.ndarray,
    gate_deltas_mocap: Optional[dict],
) -> np.ndarray:
    """Translate+yaw rigid transform in MOCAP about ``anchor_mocap``.

    Mirrors ``smoke_test._apply_gate_rigid_transform`` so the scorer's
    gate-perturbation handling exactly matches the safety / runtime
    paths. ``gate_deltas_mocap`` may be ``None`` (no perturbation), in
    which case ``points_mocap`` is returned unchanged.
    """
    if gate_deltas_mocap is None:
        return points_mocap
    dxyz = np.asarray(gate_deltas_mocap["delta_xyz"], dtype=np.float64)
    dyaw = float(gate_deltas_mocap["delta_yaw_rad"])
    c, s = np.cos(dyaw), np.sin(dyaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return (points_mocap - anchor_mocap) @ Rz.T + anchor_mocap + dxyz


# ---------------------------------------------------------------------------
# Cost functions — one per target FailureType.
# ---------------------------------------------------------------------------


GATE_COLLISION_PENALTY = 100.0   # subtracted from COLLISION_OTHER cost when
                                  # gate-collision also occurred. Should
                                  # dominate cm-scale distance differences.


def _min_obb_to_cloud_signed_distance(
    positions_ned: np.ndarray,
    quaternions_xyzw: np.ndarray,
    cloud_ned: np.ndarray,
    half_extents: np.ndarray,
    centre_offset: np.ndarray,
    bounding_radius: float,
) -> float:
    """Minimum signed distance over the trajectory.

    Sphere-cull per step (same trick `PointCloudCollisionCriterion` uses)
    before invoking the exact SDF. With clouds in the ~10k-points range
    and trajectories ~600 steps this is the difference between a
    sub-second and a multi-second scorer.
    """
    if cloud_ned.size == 0:
        return np.inf
    best = np.inf
    radius_sq = (bounding_radius + np.linalg.norm(half_extents)) ** 2
    for pos, quat in zip(positions_ned, quaternions_xyzw):
        # Coarse sphere cull around the body origin.
        deltas = cloud_ned - pos
        within = (deltas * deltas).sum(axis=1) <= radius_sq
        if not np.any(within):
            continue
        R = _R.from_quat(quat).as_matrix()
        d = obb_to_points_signed_distance(
            centre_world=pos,
            R_world_from_body=R,
            half_extents=half_extents,
            centre_offset_body=centre_offset,
            points_world=cloud_ned[within],
        )
        if d.size:
            mn = float(d.min())
            if mn < best:
                best = mn
                if best <= -bounding_radius:
                    # Already deep inside the cloud; can't beat this much.
                    break
    return best


def cost_collision_gate(
    rollout: dict,
    ctx: SceneContext,
    gate_deltas: Optional[dict],
) -> dict:
    """Cost = − (closest signed distance from drone OBB to gate cloud).

    A collision contributes a negative signed distance → flipping sign
    yields a strongly positive cost. Misses contribute small positive
    distances → small negative costs.
    """
    cloud = ctx.gate_cloud_ned(gate_deltas)
    min_d = _min_obb_to_cloud_signed_distance(
        rollout["positions_ned"], rollout["quaternions_xyzw"], cloud,
        ctx.drone_body_half_extents, ctx.drone_body_center_offset,
        ctx.drone_bounding_radius,
    )
    return {
        "cost": -min_d,
        "min_signed_distance": float(min_d),
        "collision": bool(min_d < 0.0),
    }


def cost_collision_other(
    rollout: dict,
    ctx: SceneContext,
    gate_deltas: Optional[dict],
) -> dict:
    if ctx.other_points_ned.size == 0:
        return {"cost": 0.0, "min_signed_distance": float("inf"), "collision": False,
                "note": "no other_objects declared"}
    min_d_other = _min_obb_to_cloud_signed_distance(
        rollout["positions_ned"], rollout["quaternions_xyzw"], ctx.other_points_ned,
        ctx.drone_body_half_extents, ctx.drone_body_center_offset,
        ctx.drone_bounding_radius,
    )
    # Penalise gate collisions so CEM doesn't tunnel through the gate to
    # get to the table behind it.
    gate_cloud = ctx.gate_cloud_ned(gate_deltas)
    min_d_gate = _min_obb_to_cloud_signed_distance(
        rollout["positions_ned"], rollout["quaternions_xyzw"], gate_cloud,
        ctx.drone_body_half_extents, ctx.drone_body_center_offset,
        ctx.drone_bounding_radius,
    )
    gate_collided = min_d_gate < 0.0
    cost = -min_d_other - (GATE_COLLISION_PENALTY if gate_collided else 0.0)
    return {
        "cost": cost,
        "min_signed_distance_other": float(min_d_other),
        "min_signed_distance_gate":  float(min_d_gate),
        "collision_other": bool(min_d_other < 0.0),
        "collision_gate":  bool(gate_collided),
    }


def cost_miss_gate(
    rollout: dict,
    ctx: SceneContext,
    gate_deltas: Optional[dict],
) -> dict:
    """Cost = furthest-outside-aperture offset across all plane crossings.

    The MissGateCriterion fires when the segment (prev, current) crosses
    the aperture plane outside the rectangle. We replicate the same
    geometric test here as a continuous metric: at each crossing,
    ``offset = max(|u| − hu, 0) + max(|v| − hv, 0)``. Cost is the max
    offset across all crossings; 0 if the trajectory never crossed.

    This is mode-(a) of the three MissGate sub-modes. Modes (b/c) — no-
    transit-but-near-goal and stuck — are noisier to score continuously
    and fall out of scope for v0. CEM driving the (a) cost up should
    also surface (b/c) failures incidentally.
    """
    corners = ctx.aperture_corners_ned(gate_deltas)   # (4, 3) in NED
    p0, p1, _p2, p3 = corners
    u_axis = p1 - p0
    v_axis = p3 - p0
    hu = float(np.linalg.norm(u_axis)) / 2.0
    hv = float(np.linalg.norm(v_axis)) / 2.0
    u_hat = u_axis / (2.0 * hu)
    v_hat = v_axis / (2.0 * hv)
    n_hat = np.cross(u_hat, v_hat)
    n_hat = n_hat / np.linalg.norm(n_hat)
    centre = corners.mean(axis=0)

    positions = rollout["positions_ned"]
    signed = (positions - centre) @ n_hat            # (T,)
    max_offset = 0.0
    for i in range(1, len(positions)):
        if signed[i - 1] == 0.0 and signed[i] == 0.0:
            continue
        if signed[i - 1] * signed[i] > 0.0:
            continue
        # Linear interpolation along the segment to the plane.
        denom = signed[i] - signed[i - 1]
        alpha = -signed[i - 1] / denom if denom != 0.0 else 0.0
        cross = positions[i - 1] + alpha * (positions[i] - positions[i - 1])
        rel = cross - centre
        u = float(rel @ u_hat)
        v = float(rel @ v_hat)
        offset = max(abs(u) - hu, 0.0) + max(abs(v) - hv, 0.0)
        if offset > max_offset:
            max_offset = offset

    return {
        "cost": float(max_offset),
        "any_crossing": bool(np.any(signed[:-1] * signed[1:] <= 0)),
        "aperture_hu": hu,
        "aperture_hv": hv,
    }


def cost_goal_not_reached(
    rollout: dict,
    ctx: SceneContext,
    gate_deltas: Optional[dict],
) -> dict:
    """Cost = distance from final rollout position to goal."""
    goal = ctx.goal_ned()
    if goal is None:
        return {"cost": 0.0, "note": "no goal_position in safety YAML"}
    final = rollout["positions_ned"][-1]
    dist = float(np.linalg.norm(final - goal))
    return {"cost": dist, "final_distance_to_goal": dist}


COST_FUNCTIONS = {
    "COLLISION_GATE":   cost_collision_gate,
    "COLLISION_OTHER":  cost_collision_other,
    "MISS_GATE":        cost_miss_gate,
    "GOAL_NOT_REACHED": cost_goal_not_reached,
}


# ---------------------------------------------------------------------------
# Top-level: score one trial.
# ---------------------------------------------------------------------------


def load_rollout_npz(path: Path) -> dict:
    """Read the rollout_states.npz format produced by run_eval_campaign.py."""
    data = np.load(path, allow_pickle=True)
    return {
        "times":             np.asarray(data["times"], dtype=np.float64),
        "positions_ned":     np.asarray(data["positions_ned"], dtype=np.float64),
        "quaternions_xyzw":  np.asarray(data["quaternions_xyzw"], dtype=np.float64),
        "velocities":        np.asarray(data["velocities"], dtype=np.float64),
        "failure_step":      int(data["failure_step"]),
        "failure_type":      str(data["failure_type"]),
    }


def score_trial(
    rollout_npz: Path,
    ctx: SceneContext,
    target_failure_type: str,
    gate_deltas: Optional[dict] = None,
) -> dict:
    """Score one trial against the chosen target failure type.

    ``gate_deltas`` may be ``None`` (no gate perturbation) or a dict with
    keys ``delta_xyz`` (3 floats, mocap) and ``delta_yaw_rad`` (float).
    """
    if target_failure_type not in COST_FUNCTIONS:
        raise ValueError(
            f"unsupported target_failure_type {target_failure_type!r}; "
            f"must be one of {sorted(COST_FUNCTIONS)}"
        )
    rollout = load_rollout_npz(Path(rollout_npz))
    out = COST_FUNCTIONS[target_failure_type](rollout, ctx, gate_deltas)
    out["target_failure_type"] = target_failure_type
    out["actual_failure_type"] = rollout["failure_type"]
    out["target_matched"] = bool(rollout["failure_type"] == target_failure_type)
    return out
