"""Transform painted exclude *points* from the source gate frame to each
target scene's gate frame(s).

Reads the painter output (``paint_exclude_aabbs.py --gate-only`` writes
``boxes`` + ``exclude_points_mocap`` + ``source_gate``). For each target
gate in each scene, rotates points about the source anchor (z-axis) by
the angle between source and target normals, then translates by
``target_anchor − source_anchor``. Outputs per-scene JSONs that the
renderer consumes via ``--exclude-points-dir``.

Point-level transport sidesteps the AABB re-bracketing inflation that
rotated boxes suffer at non-axis-aligned gates.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/figures/transform_exclude_points.py \\
        --painted-json runs/figures/exclude_aabbs_center_gate.json \\
        --out-dir runs/figures/per_scene_excludes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _angle_xy(v) -> float:
    v = np.asarray(v, dtype=np.float64)
    return float(np.arctan2(v[1], v[0]))


def _gate_blocks(scene_cfg: dict) -> list[dict]:
    out: list[dict] = []
    if isinstance(scene_cfg.get("gate_region"), dict):
        b = dict(scene_cfg["gate_region"])
        b.setdefault("name", "gate")
        out.append(b)
    if isinstance(scene_cfg.get("gate_regions"), list):
        out.extend(scene_cfg["gate_regions"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--painted-json", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--target-scenes", type=Path, nargs="*", default=None,
                    help="Default: every configs/scenes/*.yaml.")
    args = ap.parse_args()

    painted_json = args.painted_json if args.painted_json.is_absolute() \
        else (REPO_ROOT / args.painted_json).resolve()
    out_dir = args.out_dir if args.out_dir.is_absolute() \
        else (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(painted_json.read_text())
    if not isinstance(payload, dict) or "exclude_points_mocap" not in payload:
        raise SystemExit(
            f"{painted_json.name} is not a points payload. Re-save in the "
            "current painter (which writes 'exclude_points_mocap' + "
            "'source_gate')."
        )
    pts_src = np.asarray(payload["exclude_points_mocap"], dtype=np.float64)
    src_gate = payload["source_gate"]
    src_anchor = np.asarray(src_gate["anchor"], dtype=np.float64)
    src_normal = np.asarray(src_gate["normal"], dtype=np.float64)
    src_angle = _angle_xy(src_normal)
    print(f"[load] {pts_src.shape[0]:,} exclude points from {painted_json.name}")
    print(f"[source] anchor={src_anchor.tolist()} normal={src_normal.tolist()} "
          f"angle_deg={np.rad2deg(src_angle):.1f}")

    if args.target_scenes:
        scenes = [s if s.is_absolute() else (REPO_ROOT / s).resolve()
                  for s in args.target_scenes]
    else:
        scenes = sorted((REPO_ROOT / "configs" / "scenes").glob("*.yaml"))

    for scene_yaml in scenes:
        scene_cfg = yaml.safe_load(scene_yaml.read_text())
        gates = _gate_blocks(scene_cfg)
        if not gates:
            print(f"[skip] {scene_yaml.name} — no gate blocks")
            continue

        all_pts: list[list[float]] = []
        for gate in gates:
            if gate.get("aabb_frame", "mocap") != "mocap":
                continue
            if "anchor" not in gate or "normal" not in gate:
                continue
            tgt_anchor = np.asarray(gate["anchor"], dtype=np.float64)
            tgt_normal = np.asarray(gate["normal"], dtype=np.float64)
            theta = _angle_xy(tgt_normal) - src_angle
            c, s = np.cos(theta), np.sin(theta)
            Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            transformed = (Rz @ (pts_src - src_anchor).T).T + tgt_anchor
            all_pts.extend(transformed.tolist())
            print(f"  {scene_yaml.stem}/{gate.get('name','gate')}: "
                  f"θ={np.rad2deg(theta):+.1f}° "
                  f"Δanchor={(tgt_anchor - src_anchor).tolist()}")

        out_path = out_dir / f"exclude_points_{scene_yaml.stem}.json"
        out_path.write_text(json.dumps({"points_mocap": all_pts}, indent=2))
        print(f"  wrote {out_path.relative_to(REPO_ROOT)} "
              f"({len(all_pts):,} point(s))")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
