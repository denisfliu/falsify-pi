"""Render one colored point-cloud image per scene YAML.

For each ``configs/scenes/*.yaml`` (or the explicit list passed on the
CLI), loads the gsplat's ``pipeline.model.means`` + ``features_dc``,
applies any declared ``scene_edits`` (pure-numpy applier — no CUDA
rasterizer needed, sidesteps the gsplat-JIT pybind11 breakage), converts
the means NS → MOCAP, decodes the DC spherical harmonic into RGB, and
saves a matplotlib 3-D scatter PNG.

Usage::

    PYTHONPATH=src:external/FiGS/src:external/splatnav \\
        .venv/bin/python scripts/figures/render_scene_pointclouds.py \\
        --out runs/figures/scene_pointclouds

    # one specific scene
    PYTHONPATH=...  python scripts/figures/render_scene_pointclouds.py \\
        --scenes configs/scenes/center_gate.yaml --out /tmp/pc
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SH_C0 = 0.28209479177387814  # SH degree-0 basis value


def _resolve(rel: str, base: Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (base / p).resolve()


def _gate_aabbs_mocap(scene_cfg: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pull MOCAP-frame gate AABBs from either `gate_region` (single) or
    `gate_regions` (list). Other frames are ignored — every shipped scene
    declares gate regions in mocap."""
    out: list[tuple[np.ndarray, np.ndarray]] = []
    blocks: list[dict] = []
    if isinstance(scene_cfg.get("gate_region"), dict):
        blocks.append(scene_cfg["gate_region"])
    if isinstance(scene_cfg.get("gate_regions"), list):
        blocks.extend(scene_cfg["gate_regions"])
    for b in blocks:
        if b.get("aabb_frame", "mocap") != "mocap":
            continue
        out.append((np.asarray(b["aabb_min"], dtype=np.float64),
                    np.asarray(b["aabb_max"], dtype=np.float64)))
    return out


def _cache_path(scene_yaml: Path) -> Path:
    return REPO_ROOT / "runs" / "cache" / f"scene_{scene_yaml.stem}.npz"


def _cache_is_fresh(scene_yaml: Path, cache: Path) -> bool:
    if not cache.exists():
        return False
    return cache.stat().st_mtime >= scene_yaml.stat().st_mtime


def _load_scene_pointcloud(scene_yaml: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (means_mocap (N,3), rgb (N,3) in [0,1], scene_cfg) for one scene YAML.

    Disk-cached: post-scene_edits means + rgb get pickled to
    ``runs/cache/scene_<stem>.npz``. A cached read is ~100 ms vs ~25 s for
    the nerfstudio eval_setup + checkpoint load path.
    """
    from falsify.io import load_yaml
    cache = _cache_path(scene_yaml)
    scene_cfg = load_yaml(scene_yaml)
    if _cache_is_fresh(scene_yaml, cache):
        data = np.load(cache)
        return data["means_mocap"], data["rgb"], scene_cfg
    return _load_scene_pointcloud_fresh(scene_yaml, scene_cfg, cache)


def _load_scene_pointcloud_fresh(scene_yaml: Path, scene_cfg: dict,
                                  cache: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    from falsify.io import build_frame_graph
    from falsify.sim.scene_edits import apply_edits_to_pipeline, load_scene_edits
    from nerfstudio.utils.eval_utils import eval_setup
    import torch

    scene_dir = scene_yaml.parent
    fg = build_frame_graph(scene_cfg, base_path=scene_dir)

    gsplat_yml = _resolve(scene_cfg["gsplat_config_yml"], scene_dir)
    data_cwd = _resolve(scene_cfg["gsplat_data_cwd"], scene_dir) if "gsplat_data_cwd" in scene_cfg else None

    prev_cwd = os.getcwd()
    if data_cwd is not None:
        os.chdir(data_cwd)
    try:
        _, pipeline, _, _ = eval_setup(gsplat_yml, eval_num_rays_per_chunk=None, test_mode="test")
    finally:
        os.chdir(prev_cwd)

    # Apply scene_edits at the pipeline level — handles
    # DuplicateAABB (which grows N) by also duplicating features_dc etc.,
    # so the post-edit means / features arrays stay aligned. Pure-CPU
    # tensor ops; no CUDA rasterizer involvement.
    edits = load_scene_edits(scene_cfg) or []
    if edits:
        n_moved = apply_edits_to_pipeline(pipeline, edits, fg)
        print(f"  [edits] applied {len(edits)} scene_edit(s) — {n_moved} Gaussians touched")

    model = pipeline.model
    means_ns = model.means.detach().cpu().numpy().astype(np.float64)

    # features_dc: (N, 3) — DC SH coefficient per channel.
    feats_dc = model.features_dc.detach().cpu().numpy().astype(np.float32)
    # Splatfacto stores raw DC (pre-activation). Decode the same way the
    # rasterizer does: rgb = 0.5 + SH_C0 * features_dc, clipped to [0, 1].
    rgb = 0.5 + SH_C0 * feats_dc
    rgb = np.clip(rgb, 0.0, 1.0)

    # NS → MOCAP
    T = fg.transform("ns", "mocap")
    s = getattr(T, "s", 1.0)
    means_mocap = (s * (T.R @ means_ns.T)).T + T.t

    # Free the pipeline to keep VRAM low across scenes.
    del pipeline, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, means_mocap=means_mocap, rgb=rgb.astype(np.float32))
    print(f"  [cache] wrote {cache.relative_to(REPO_ROOT)} "
          f"({means_mocap.shape[0]:,} Gaussians)")
    return means_mocap, rgb, scene_cfg


def _gate_blocks_with_pose(scene_cfg: dict) -> list[dict]:
    """Return MOCAP gate blocks that carry anchor + normal (used for the
    gate-to-gate OBB transform)."""
    blocks: list[dict] = []
    if isinstance(scene_cfg.get("gate_region"), dict):
        b = dict(scene_cfg["gate_region"])
        b.setdefault("name", "gate")
        blocks.append(b)
    if isinstance(scene_cfg.get("gate_regions"), list):
        blocks.extend(scene_cfg["gate_regions"])
    return [b for b in blocks
            if b.get("aabb_frame", "mocap") == "mocap"
            and "anchor" in b and "normal" in b]


def _apply_painted_boxes(
    means: np.ndarray, mask: np.ndarray, *,
    boxes: list[dict], src_anchor: np.ndarray, src_normal: np.ndarray,
    scene_cfg: dict,
) -> np.ndarray:
    """For each target gate in `scene_cfg`, transform the painter's
    source-frame Gaussians INTO source-gate-local coords by applying the
    inverse gate-to-gate transform (rotate the candidate means about the
    *target* anchor by -θ, translate to the source-frame), then AABB-test
    against the painted boxes there. Lossless under arbitrary z-rotation
    (no re-bracketing inflation)."""
    src_angle = float(np.arctan2(src_normal[1], src_normal[0]))
    cand_idx = np.where(mask)[0]
    if cand_idx.size == 0:
        return mask
    for gate in _gate_blocks_with_pose(scene_cfg):
        tgt_anchor = np.asarray(gate["anchor"], dtype=np.float64)
        tgt_normal = np.asarray(gate["normal"], dtype=np.float64)
        theta = float(np.arctan2(tgt_normal[1], tgt_normal[0])) - src_angle
        # Inverse: target-MOCAP → source-MOCAP. Translate by source-target,
        # rotate by -θ about source anchor.
        c, s = np.cos(-theta), np.sin(-theta)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        local = (Rz @ (means[cand_idx] - tgt_anchor).T).T + src_anchor
        per_gate_drop = np.zeros(cand_idx.size, dtype=bool)
        for b in boxes:
            bmn = np.asarray(b["min"], dtype=np.float64)
            bmx = np.asarray(b["max"], dtype=np.float64)
            per_gate_drop |= ((local >= bmn) & (local <= bmx)).all(axis=1)
        mask[cand_idx[per_gate_drop]] = False
        cand_idx = np.where(mask)[0]
        if cand_idx.size == 0:
            break
    return mask


def _filter_and_subsample(
    means: np.ndarray,
    rgb: np.ndarray,
    *,
    aabb_min: Optional[np.ndarray],
    aabb_max: Optional[np.ndarray],
    max_points: int,
    rng: np.random.Generator,
    z_cull: Optional[float] = None,
    gate_aabbs: Optional[list[tuple[np.ndarray, np.ndarray]]] = None,
    exclude_aabbs: Optional[list[tuple[np.ndarray, np.ndarray]]] = None,
    exclude_points: Optional[np.ndarray] = None,
    exclude_eps: float = 0.03,
    painted_boxes: Optional[list[dict]] = None,
    painted_src_anchor: Optional[np.ndarray] = None,
    painted_src_normal: Optional[np.ndarray] = None,
    scene_cfg: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.ones(means.shape[0], dtype=bool)
    if aabb_min is not None and aabb_max is not None:
        mask &= ((means >= aabb_min) & (means <= aabb_max)).all(axis=1)
    if z_cull is not None:
        # Cull points above z_cull UNLESS they fall inside a gate AABB.
        above = means[:, 2] > z_cull
        protected = np.zeros(means.shape[0], dtype=bool)
        for mn, mx in (gate_aabbs or []):
            protected |= ((means >= mn) & (means <= mx)).all(axis=1)
        mask &= ~(above & ~protected)
    for mn, mx in (exclude_aabbs or []):
        mask &= ~((means >= mn) & (means <= mx)).all(axis=1)
    if exclude_points is not None and exclude_points.shape[0] > 0:
        from scipy.spatial import cKDTree
        tree = cKDTree(exclude_points)
        # Only test candidates that survived prior filters — keeps the
        # query light when the crop / z-cull have already shrunk N.
        cand_idx = np.where(mask)[0]
        d, _ = tree.query(means[cand_idx], k=1)
        within = d < exclude_eps
        mask[cand_idx[within]] = False
    if painted_boxes:
        assert painted_src_anchor is not None and painted_src_normal is not None
        assert scene_cfg is not None
        mask = _apply_painted_boxes(
            means, mask,
            boxes=painted_boxes,
            src_anchor=painted_src_anchor, src_normal=painted_src_normal,
            scene_cfg=scene_cfg,
        )
    pts = means[mask]
    cols = rgb[mask]
    if pts.shape[0] > max_points:
        idx = rng.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]
        cols = cols[idx]
    return pts, cols


def _render_views(
    means_mocap: np.ndarray,
    rgb: np.ndarray,
    *,
    out_dir: Path,
    name: str,
    point_size: float,
    fov_deg: float,
    dpi: int = 400,
    figsize: tuple[float, float] = (8.0, 8.0),
    trim: bool = True,
) -> list[Path]:
    """Save a 3-up figure (perspective, top-down, side) PNG for this scene."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d)

    out_dir.mkdir(parents=True, exist_ok=True)

    xs, ys, zs = means_mocap[:, 0], means_mocap[:, 1], means_mocap[:, 2]

    # Ego view: eye at MOCAP (-1.5, 0, 2.0) looking toward (1.5, 0, 1.0)
    # — i.e. slightly downward along +x. matplotlib's 3D camera looks at the
    # box center along (elev, azim); we derive those from (focal - eye) and
    # center the axis box on the focal point so the apparent target lines up.
    ego_eye = np.array([-1.5, 0.0, 2.0])
    ego_focal = np.array([1.5, 0.0, 1.0])
    fwd = ego_focal - ego_eye
    azim_deg = np.rad2deg(np.arctan2(fwd[1], fwd[0])) + 180.0   # +180 because matplotlib azim defines incoming-ray direction
    horiz = float(np.hypot(fwd[0], fwd[1]))
    elev_deg = np.rad2deg(np.arctan2(-fwd[2], horiz))           # +ve elev = looking down (camera above)
    # Tight bbox half-width: max distance from focal to any *visible* point.
    # Drives how much of the viewport the cloud fills (smaller bbox → bigger
    # apparent cloud, since matplotlib stretches the bbox to fit the figure).
    rng = float(np.max(np.abs(means_mocap - ego_focal))) * 1.02

    views = [
        ("ego", dict(elev=elev_deg, azim=azim_deg, perspective=True,
                     focal=ego_focal, fov_deg=fov_deg)),
    ]

    def _strip_frame(ax):
        # Hide axes, ticks, labels, grid lines, and the surrounding panes
        # so the scatter is the only visible thing.
        ax.set_axis_off()
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.set_pane_color((1.0, 1.0, 1.0, 0.0))
            pane.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.grid(False)

    # Tight axis bounds = actual cloud extent (per-axis). Proportional
    # box_aspect means the 3-D box matches the data's true shape, not a
    # cube — so the projected silhouette fills the figure instead of
    # leaving big empty wedges in the corners of a cubic hull.
    mn = means_mocap.min(axis=0)
    mx = means_mocap.max(axis=0)
    pad_ax = 0.02 * (mx - mn).max()
    lo = mn - pad_ax
    hi = mx + pad_ax
    extents = hi - lo

    def _apply_view(ax, vkw):
        ax.set_proj_type("persp", focal_length=1.0 / np.tan(
            np.deg2rad(vkw["fov_deg"]) / 2.0))
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(tuple(extents.tolist()))
        ax.view_init(elev=vkw["elev"], azim=vkw["azim"])

    def _trim_white(path: Path) -> None:
        # Post-process PIL trim: find the non-near-white bbox and crop to
        # it. Eliminates the residual matplotlib margin around the 3-D
        # axes without losing any cloud pixels.
        from PIL import Image, ImageChops
        im = Image.open(path).convert("RGB")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        diff = ImageChops.difference(im, bg)
        # threshold to ignore tiny near-white artifacts.
        bbox = diff.point(lambda x: 0 if x < 8 else x).getbbox()
        if bbox is not None and bbox != (0, 0, im.size[0], im.size[1]):
            im.crop(bbox).save(path)

    written: list[Path] = []
    for vname, vkw in views:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(xs, ys, zs, c=rgb, s=point_size, marker=".", linewidths=0,
                   edgecolors="none")
        _apply_view(ax, vkw)
        _strip_frame(ax)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        single_path = out_dir / f"{name}_{vname}.png"
        fig.savefig(single_path, bbox_inches="tight", pad_inches=0,
                    facecolor="white")
        plt.close(fig)
        if trim:
            _trim_white(single_path)
        written.append(single_path)
        print(f"  wrote {single_path}")

    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenes", type=Path, nargs="*", default=None,
                    help="Specific scene YAMLs. Default: every "
                         "configs/scenes/*.yaml in the repo.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output dir for per-scene PNGs.")
    ap.add_argument("--max-points", type=int, default=120_000,
                    help="Cap on point count after AABB crop (default 120k).")
    ap.add_argument("--crop", default=None, metavar="MNX,MNY,MNZ:MXX,MXY,MXZ",
                    help="Optional MOCAP AABB crop, e.g. "
                         "'-1,-1.5,-0.2:3,1.5,2.2'. Default: no crop.")
    ap.add_argument("--point-size", type=float, default=0.6,
                    help="matplotlib scatter point size (default 0.6).")
    ap.add_argument("--fov-deg", type=float, default=110.0,
                    help="Ego-camera vertical FOV in degrees (default 110).")
    ap.add_argument("--dpi", type=int, default=400,
                    help="Output DPI (default 400).")
    ap.add_argument("--figsize", type=float, nargs=2, default=(8.0, 8.0),
                    metavar=("W", "H"),
                    help="Figure size in inches (default 8 x 8).")
    ap.add_argument("--no-trim", action="store_true",
                    help="Skip the post-save PIL trim of white margins.")
    ap.add_argument("--exclude-json", type=Path, default=None,
                    help="Path to JSON list of {min, max} MOCAP AABBs to drop "
                         "from the cloud (paint_exclude_aabbs.py output). "
                         "Applied to every scene.")
    ap.add_argument("--exclude-json-dir", type=Path, default=None,
                    help="Directory containing per-scene "
                         "exclude_aabbs_<scene_stem>.json files "
                         "(transform_exclude_aabbs.py output).")
    ap.add_argument("--exclude-points-dir", type=Path, default=None,
                    help="Directory containing per-scene "
                         "exclude_points_<scene_stem>.json files "
                         "(transform_exclude_points.py output). Each lists "
                         "MOCAP positions; any Gaussian within --exclude-eps "
                         "of one is dropped.")
    ap.add_argument("--exclude-eps", type=float, default=0.03,
                    help="Match radius (meters) for --exclude-points-dir. "
                         "Default 0.03.")
    ap.add_argument("--painted-boxes-json", type=Path, default=None,
                    help="A painter output (boxes + source_gate). The "
                         "renderer transforms each box per-scene gate "
                         "(rotate about source anchor, translate) and "
                         "tests Gaussians in the source frame — lossless, "
                         "no AABB inflation, no point materialization. "
                         "Skip --exclude-points-dir / transform step.")
    ap.add_argument("--z-cull", type=float, default=1.5,
                    help="Drop points above this MOCAP z, except those "
                         "inside a gate AABB. Pass a sentinel like 99 to "
                         "disable. Default 1.5.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.scenes:
        scenes = [(s if s.is_absolute() else (REPO_ROOT / s).resolve()) for s in args.scenes]
    else:
        scenes = sorted((REPO_ROOT / "configs" / "scenes").glob("*.yaml"))
    print(f"[scenes] {len(scenes)} to render")

    aabb_min = aabb_max = None
    if args.crop is not None:
        lo, hi = args.crop.split(":")
        aabb_min = np.array([float(v) for v in lo.split(",")], dtype=np.float64)
        aabb_max = np.array([float(v) for v in hi.split(",")], dtype=np.float64)
        print(f"[crop] mocap AABB min={aabb_min.tolist()} max={aabb_max.tolist()}")

    rng = np.random.default_rng(args.seed)

    import json as _json

    def _load_exclude(path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
        if not path.exists():
            return []
        out = []
        for e in _json.loads(path.read_text()):
            out.append((
                np.asarray(e["min"], dtype=np.float64),
                np.asarray(e["max"], dtype=np.float64),
            ))
        return out

    shared_excludes: list[tuple[np.ndarray, np.ndarray]] = []
    if args.exclude_json is not None:
        shared_excludes = _load_exclude(args.exclude_json)
        print(f"[exclude] shared: {len(shared_excludes)} AABB(s) from {args.exclude_json}")

    painted_boxes: list[dict] = []
    painted_src_anchor = None
    painted_src_normal = None
    if args.painted_boxes_json is not None:
        payload = _json.loads(args.painted_boxes_json.read_text())
        if isinstance(payload, list):
            raise SystemExit(
                f"{args.painted_boxes_json} is the legacy boxes-only "
                "format. Re-save in the current painter, or wrap it "
                "as {boxes: [...], source_gate: {...}}."
            )
        painted_boxes = payload.get("boxes", [])
        src_gate = payload["source_gate"]
        painted_src_anchor = np.asarray(src_gate["anchor"], dtype=np.float64)
        painted_src_normal = np.asarray(src_gate["normal"], dtype=np.float64)
        print(f"[painted-boxes] {len(painted_boxes)} box(es) anchored on "
              f"{src_gate.get('name','gate')}@{painted_src_anchor.tolist()} "
              f"normal {painted_src_normal.tolist()}")

    args.out.mkdir(parents=True, exist_ok=True)

    for scene_yaml in scenes:
        name = scene_yaml.stem
        print(f"\n=== {name} ({scene_yaml.relative_to(REPO_ROOT)}) ===")
        means_mocap, rgb, scene_cfg = _load_scene_pointcloud(scene_yaml)
        print(f"  loaded {means_mocap.shape[0]:,} Gaussians")
        gate_aabbs = _gate_aabbs_mocap(scene_cfg)
        scene_excludes = list(shared_excludes)
        if args.exclude_json_dir is not None:
            per_scene_path = args.exclude_json_dir / f"exclude_aabbs_{name}.json"
            loaded = _load_exclude(per_scene_path)
            scene_excludes.extend(loaded)
            print(f"  [exclude] per-scene: {len(loaded)} from {per_scene_path.name}")
        scene_exclude_points = None
        if args.exclude_points_dir is not None:
            ppath = args.exclude_points_dir / f"exclude_points_{name}.json"
            if ppath.exists():
                payload = _json.loads(ppath.read_text())
                scene_exclude_points = np.asarray(
                    payload["points_mocap"], dtype=np.float64)
                print(f"  [exclude-pts] {scene_exclude_points.shape[0]:,} "
                      f"from {ppath.name} (eps={args.exclude_eps})")
        pts, cols = _filter_and_subsample(
            means_mocap, rgb,
            aabb_min=aabb_min, aabb_max=aabb_max,
            max_points=args.max_points, rng=rng,
            z_cull=args.z_cull, gate_aabbs=gate_aabbs,
            exclude_aabbs=scene_excludes,
            exclude_points=scene_exclude_points,
            exclude_eps=args.exclude_eps,
            painted_boxes=painted_boxes,
            painted_src_anchor=painted_src_anchor,
            painted_src_normal=painted_src_normal,
            scene_cfg=scene_cfg,
        )
        print(f"  rendering {pts.shape[0]:,} points "
              f"(z_cull={args.z_cull}, n_gate_aabbs={len(gate_aabbs)})")
        _render_views(pts, cols, out_dir=args.out, name=name,
                      point_size=args.point_size, fov_deg=args.fov_deg,
                      dpi=args.dpi, figsize=tuple(args.figsize),
                      trim=not args.no_trim)

    print(f"\n[done] PNGs in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
