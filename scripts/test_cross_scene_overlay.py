"""One-off experiment: overlay Gaussians from the right_scene gsplat onto
the left_scene gsplat to fill in the bare floor patch left behind after
the gate was moved by ``center_gate.yaml``.

Pipeline
--------
1. Load the left_scene gsplat. Apply the center_gate ``scene_edits``
   (moves the gate Gaussians out of the old location).
2. Load the right_scene gsplat as an independent ``pipeline``.
3. Select right_scene Gaussians whose mean lies inside an AABB authored
   in joint mocap (the region of the original left-gate footprint).
4. Transform those Gaussians through ``right_ns → joint_mocap → left_ns``
   — similarity composition via the two scene FrameGraphs.
   - Means: ordinary point transform.
   - Quats (wxyz): pre-multiplied by the rotation part of the composed
     similarity.
   - Scales (log): shifted by ``log(s_left / s_right)`` so the world-
     space Gaussian size is preserved (per-axis log scales add the
     uniform log of the scale-factor ratio).
   - features_dc / features_rest / opacities / sagesplat extras
     (``clip_embeds``, ``affordance``): copied untouched.
5. Append the transformed Gaussians to ``left_pipeline.model.gauss_params``
   (creates new ``nn.Parameter`` tensors of the concatenated size).
6. Launch the viser viewer with the combined model.

The script is intentionally a script (not a CLI / library function) — if
the visual result is promising we'll lift the merge into a
``scene_edits`` type with a clean YAML schema.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from falsify.io import build_frame_graph, load_yaml
from falsify.sim.scene_edits import apply_edits_to_pipeline, load_scene_edits

from nerfstudio.scripts.viewer.run_viewer import _start_viewer
from nerfstudio.utils.eval_utils import eval_setup


LEFT_SCENE_YAML  = Path("configs/scenes/center_gate.yaml")
RIGHT_SCENE_YAML = Path("configs/scenes/right_gate.yaml")

# Region in JOINT MOCAP to pull Gaussians from. Cover the original left-
# gate footprint plus a generous floor buffer, but cap z below the gate
# top so we don't drag down ceiling artefacts from the right_scene.
SOURCE_AABB_MOCAP_MIN = np.array([0.30, 0.05, 0.0])
SOURCE_AABB_MOCAP_MAX = np.array([1.40, 1.30, 1.40])


def _resolve(rel: str, base: Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (base / p).resolve()


def _load_pipeline(gsplat_yml: Path, data_cwd: Path | None):
    prev = os.getcwd()
    if data_cwd is not None:
        os.chdir(data_cwd)
    try:
        config, pipeline, _, step = eval_setup(
            gsplat_yml, eval_num_rays_per_chunk=None, test_mode="test",
        )
    finally:
        os.chdir(prev)
    return config, pipeline, step


def _transform_corners(corners_src: np.ndarray, T) -> np.ndarray:
    """Lift 8 AABB corners through Sim3/SE3 ``T`` and return the new
    axis-aligned min/max bracket in T's destination frame."""
    s = getattr(T, "s", 1.0)
    return (s * (T.R @ corners_src.T) + T.t[:, None]).T


def _aabb_min_max_in(min_xyz, max_xyz, T) -> tuple[np.ndarray, np.ndarray]:
    mn, mx = min_xyz, max_xyz
    corners = np.array([
        [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
        [mn[0], mx[1], mn[2]], [mx[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
        [mn[0], mx[1], mx[2]], [mx[0], mx[1], mx[2]],
    ])
    out = _transform_corners(corners, T)
    return out.min(axis=0), out.max(axis=0)


def _rotation_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → unit quaternion in wxyz layout."""
    import math
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return np.array([0.25 * s,
                         (R[2, 1] - R[1, 2]) / s,
                         (R[0, 2] - R[2, 0]) / s,
                         (R[1, 0] - R[0, 1]) / s])
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return np.array([(R[2, 1] - R[1, 2]) / s,
                         0.25 * s,
                         (R[0, 1] + R[1, 0]) / s,
                         (R[0, 2] + R[2, 0]) / s])
    if R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return np.array([(R[0, 2] - R[2, 0]) / s,
                         (R[0, 1] + R[1, 0]) / s,
                         0.25 * s,
                         (R[1, 2] + R[2, 1]) / s])
    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return np.array([(R[1, 0] - R[0, 1]) / s,
                     (R[0, 2] + R[2, 0]) / s,
                     (R[1, 2] + R[2, 1]) / s,
                     0.25 * s])


def _quat_product_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    if q1.ndim == 1: q1 = q1[None]
    if q2.ndim == 1: q2 = q2[None]
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=1)


def main() -> int:
    # ---- Left scene ---------------------------------------------------
    left_cfg = load_yaml(LEFT_SCENE_YAML)
    left_dir = LEFT_SCENE_YAML.parent
    left_fg = build_frame_graph(left_cfg, base_path=left_dir)
    left_yml = _resolve(left_cfg["gsplat_config_yml"], left_dir)
    left_cwd = _resolve(left_cfg["gsplat_data_cwd"], left_dir)
    left_edits = load_scene_edits(left_cfg)
    print(f"[left ]  {left_yml}\n         scene edits: {[e.name for e in left_edits]}")
    left_config, left_pipeline, left_step = _load_pipeline(left_yml, left_cwd)
    if left_edits:
        n = apply_edits_to_pipeline(left_pipeline, left_edits, left_fg)
        print(f"[left ]  applied {len(left_edits)} edit(s); {n} Gaussians modified")

    # ---- Right scene (Gaussian donor) ---------------------------------
    right_cfg = load_yaml(RIGHT_SCENE_YAML)
    right_dir = RIGHT_SCENE_YAML.parent
    right_fg = build_frame_graph(right_cfg, base_path=right_dir)
    right_yml = _resolve(right_cfg["gsplat_config_yml"], right_dir)
    right_cwd = _resolve(right_cfg["gsplat_data_cwd"], right_dir)
    print(f"[right]  {right_yml}")
    _, right_pipeline, _ = _load_pipeline(right_yml, right_cwd)

    # ---- Transforms ---------------------------------------------------
    # right_ns → joint_mocap, then joint_mocap → left_ns. (Both scene
    # YAMLs declare a `mocap` frame that's the SAME joint mocap by
    # construction — see data/gate_scenes_export/objects_final/README.md.)
    T_right_ns_to_mocap = right_fg.transform("ns", "mocap")
    T_mocap_to_left_ns  = left_fg.transform("mocap", "ns")
    # Composition: a single Sim3 ns_right → ns_left.
    T_ns_to_ns = T_mocap_to_left_ns @ T_right_ns_to_mocap
    s_total = getattr(T_ns_to_ns, "s", 1.0)
    R_total = T_ns_to_ns.R
    t_total = T_ns_to_ns.t
    print(f"[xform]  ns(right) → ns(left): scale = {s_total:.4f}")

    # AABB in mocap → bracket in right_ns (the donor scene's native frame).
    src_ns_min, src_ns_max = _aabb_min_max_in(
        SOURCE_AABB_MOCAP_MIN, SOURCE_AABB_MOCAP_MAX,
        right_fg.transform("mocap", "ns"),
    )
    means_right_ns = right_pipeline.model.means.detach().cpu().numpy().astype(np.float64)
    mask = ((means_right_ns >= src_ns_min) & (means_right_ns <= src_ns_max)).all(axis=1)
    n_selected = int(mask.sum())
    print(f"[donor]  {n_selected} Gaussians selected from right_scene "
          f"(AABB joint mocap {SOURCE_AABB_MOCAP_MIN.tolist()} → {SOURCE_AABB_MOCAP_MAX.tolist()})")
    if n_selected == 0:
        raise SystemExit("no donor Gaussians selected — widen SOURCE_AABB or check transforms")

    sel_means_right_ns = means_right_ns[mask]
    sel_means_left_ns = (s_total * (R_total @ sel_means_right_ns.T)).T + t_total

    quats_right = right_pipeline.model.quats.detach().cpu().numpy().astype(np.float64)
    sel_quats_right = quats_right[mask]  # wxyz
    q_R_wxyz = _rotation_to_quat_wxyz(R_total)
    sel_quats_left = _quat_product_wxyz(
        np.broadcast_to(q_R_wxyz, sel_quats_right.shape), sel_quats_right,
    )

    # Scales: log-space. World-size preserved when log_scale shifts by
    # log(s_left / s_right) = log(s_total).
    scales_right = right_pipeline.model.scales.detach().cpu().numpy().astype(np.float64)
    sel_scales = scales_right[mask] + float(np.log(s_total))

    # Color / opacity / sagesplat extras: copy verbatim. (clip_embeds and
    # affordance are scene-local features but visually the means / scales
    # dominate at first glance — the test is whether the *geometry* lines
    # up. We can iterate on these later.)
    def _copy_field(name: str):
        src = getattr(right_pipeline.model, name, None)
        if src is None:
            return None
        return src.detach().cpu().numpy()[mask]

    sel_features_dc  = _copy_field("features_dc")
    sel_features_rest = _copy_field("features_rest")
    sel_opacities    = _copy_field("opacities")
    sel_clip_embeds  = _copy_field("clip_embeds")
    sel_affordance   = _copy_field("affordance")

    # ---- Append to left pipeline --------------------------------------
    device = left_pipeline.model.means.device
    dtype  = left_pipeline.model.means.dtype

    def _cat_field(name: str, new_np):
        if new_np is None:
            return
        # The model's `means` etc. are properties that delegate to the
        # `gauss_params` ParameterDict. Update only the dict entry —
        # `setattr(model, name, ...)` raises because the property is
        # already registered.
        gp = getattr(left_pipeline.model, "gauss_params", None)
        if gp is not None and name in gp:
            old = gp[name]
            new_t = torch.as_tensor(new_np, device=old.device, dtype=old.dtype)
            if new_t.shape[1:] != old.shape[1:]:
                print(f"[merge]  WARNING: {name!r} shape mismatch — "
                      f"left {tuple(old.shape[1:])} vs right {tuple(new_t.shape[1:])}; skipping")
                return
            combined = torch.cat([old.data, new_t], dim=0)
            gp[name] = nn.Parameter(combined, requires_grad=old.requires_grad)
            return
        # Non-sagesplat fallback: legacy attribute on the model.
        old = getattr(left_pipeline.model, name, None)
        if old is None:
            print(f"[merge]  WARNING: left model has no field {name!r}; skipping")
            return
        new_t = torch.as_tensor(new_np, device=old.device, dtype=old.dtype)
        combined = torch.cat([old.data, new_t], dim=0)
        try:
            old.data = combined  # in-place data swap; preserves the Parameter identity
        except Exception:
            print(f"[merge]  WARNING: could not update {name!r} in place")

    _cat_field("means", sel_means_left_ns)
    _cat_field("quats", sel_quats_left)
    _cat_field("scales", sel_scales)
    _cat_field("features_dc",   sel_features_dc)
    _cat_field("features_rest", sel_features_rest)
    _cat_field("opacities",     sel_opacities)
    _cat_field("clip_embeds",   sel_clip_embeds)
    _cat_field("affordance",    sel_affordance)
    print(f"[merge]  left model now has "
          f"{left_pipeline.model.means.shape[0]} Gaussians")

    # ---- Launch viser -------------------------------------------------
    left_config.vis = "viewer"
    left_config.viewer.websocket_host = "0.0.0.0"
    left_config.viewer.websocket_port = 7007
    print(f"[viser]  http://0.0.0.0:7007")
    os.chdir(left_cwd)
    _start_viewer(left_config, left_pipeline, left_step)
    return 0


if __name__ == "__main__":
    sys.exit(main())
