"""Interactive plotly inspector for picking waypoints in a scene.

Renders the scene's object point clouds + landmarks extracted from
``objects_summary.json`` + (optionally) an authored course's waypoints
and planned spline, all in one HTML you can rotate, zoom, and hover-over
to read MOCAP coordinates.

The intended workflow is iterative:

  1. Run with just ``--scene`` to see what's where (gate AABB, plane-cut
     posts, table) plus the implied start at (0, 0, 1.5).
  2. Hover over the rendered points / landmarks to copy MOCAP
     coordinates into a course YAML.
  3. Re-run with ``--course`` to overlay the result and check the
     spline curves the right way through the gate.

Usage::

    PYTHONPATH=src .venv/bin/python -m falsify.cli.inspect_scene_plotly \\
        --scene configs/scenes/left_gate.yaml \\
        --start "[0, 0, 1.5]" \\
        --out runs/inspect/left_gate.html

    # later, after authoring a course:
    PYTHONPATH=src .venv/bin/python -m falsify.cli.inspect_scene_plotly \\
        --scene configs/scenes/left_gate.yaml \\
        --course configs/courses/through_left_gate.yaml \\
        --out runs/inspect/left_gate.html
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path
from typing import Optional

import numpy as np

from falsify.sim.scene_edits import (
    apply_edits_to_scene_object, load_scene_edits,
)


# ---------------------------------------------------------------------------
# Landmark extraction
# ---------------------------------------------------------------------------


def _objects_summary_path(scene_dir: Path, scene_cfg: dict) -> Optional[Path]:
    """Find objects_summary.json next to scene_objects PLYs (best effort)."""
    for entry in scene_cfg.get("scene_objects", []) or []:
        ply = Path(entry["ply"])
        if not ply.is_absolute():
            ply = (scene_dir / ply).resolve()
        cand = ply.parent / "objects_summary.json"
        if cand.exists():
            return cand
    return None


def _scene_landmarks(scene_cfg: dict, scene_dir: Path) -> dict:
    """Pull AABBs, centers, plane_cuts out of objects_summary.json,
    keyed by the scene_objects names in the scene YAML.

    Match strategy: the summary uses fully-qualified keys like
    ``left_gate`` / ``right_table``; the scene YAML uses short names
    ``gate`` / ``table``. We use the PLY filename stem (e.g.
    ``left_gate.ply`` → ``left_gate``) to look up the summary entry
    unambiguously, regardless of which short name the scene chose.
    """
    summary_path = _objects_summary_path(scene_dir, scene_cfg)
    if summary_path is None:
        return {}
    summary = json.loads(summary_path.read_text())
    objects = summary.get("objects", {})

    landmarks: dict = {}
    for entry in scene_cfg.get("scene_objects", []) or []:
        ply = Path(entry["ply"])
        stem = ply.stem   # e.g. "left_gate"
        if stem in objects:
            landmarks[entry["name"]] = objects[stem]
    return landmarks


# ---------------------------------------------------------------------------
# Plotly traces
# ---------------------------------------------------------------------------


def _pc_trace(points: np.ndarray, name: str, color, size: int = 1):
    import plotly.graph_objects as go
    return go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode="markers",
        marker=dict(size=size, color=color, opacity=0.6),
        name=name,
        hovertemplate=(name + "<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>"),
    )


def _markers_trace(points: list[tuple[str, np.ndarray]], color, size: int = 6, symbol: str = "circle"):
    import plotly.graph_objects as go
    if not points:
        return None
    xs = [p[1][0] for p in points]
    ys = [p[1][1] for p in points]
    zs = [p[1][2] for p in points]
    texts = [p[0] for p in points]
    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="markers+text",
        marker=dict(size=size, color=color, symbol=symbol),
        text=texts,
        textposition="top center",
        hovertemplate="<b>%{text}</b><br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
        name="markers",
    )


def _body_axes_at_yaw(yaw_mocap: float, length: float):
    """Return (body_x, body_y, body_z) unit vectors in MOCAP for a drone
    yawed by ``yaw_mocap`` about mocap +z. FiGS body is FRD (x=forward,
    y=right, z=down) with world=NED; in MOCAP (z-up) this maps to
    body_z = -z_mocap and body_y = body_z × body_x.
    """
    cx, sx = float(np.cos(yaw_mocap)), float(np.sin(yaw_mocap))
    body_x = np.array([cx,  sx, 0.0]) * length
    body_z = np.array([0.0, 0.0, -1.0]) * length
    # body_y = body_z × body_x  (gives FRD's "right" in mocap)
    body_y = np.cross(body_z / length, body_x / length) * length
    return body_x, body_y, body_z


def _body_frames_traces(positions_xyz, yaws_mocap, length: float = 0.25, name_prefix: str = "body"):
    """Three plotly Scatter3d traces (one per axis) drawing the body frame
    at each (position, yaw) pair as a small RGB triad.
    """
    import plotly.graph_objects as go
    traces = []
    for axis_name, axis_idx, color in (("x", 0, "red"), ("y", 1, "green"), ("z", 2, "blue")):
        xs, ys, zs = [], [], []
        for pos, yaw in zip(positions_xyz, yaws_mocap):
            bx, by, bz = _body_axes_at_yaw(float(yaw), length)
            v = (bx, by, bz)[axis_idx]
            p2 = np.asarray(pos) + v
            xs += [pos[0], p2[0], None]
            ys += [pos[1], p2[1], None]
            zs += [pos[2], p2[2], None]
        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(color=color, width=6),
            name=f"{name_prefix}_{axis_name}",
            hoverinfo="skip",
        ))
    return traces


def _polyline_trace(points: np.ndarray, name: str, color, width: int = 4):
    import plotly.graph_objects as go
    return go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode="lines",
        line=dict(color=color, width=width),
        name=name,
        hovertemplate=(name + "<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>"),
    )


def _aabb_edges_trace(aabb_min: np.ndarray, aabb_max: np.ndarray, name: str, color):
    import plotly.graph_objects as go
    x0, y0, z0 = aabb_min
    x1, y1, z1 = aabb_max
    # 12 edges of the box, with `None` breaks so plotly draws each segment.
    pts = []
    edges = [
        ((x0, y0, z0), (x1, y0, z0)), ((x1, y0, z0), (x1, y1, z0)),
        ((x1, y1, z0), (x0, y1, z0)), ((x0, y1, z0), (x0, y0, z0)),
        ((x0, y0, z1), (x1, y0, z1)), ((x1, y0, z1), (x1, y1, z1)),
        ((x1, y1, z1), (x0, y1, z1)), ((x0, y1, z1), (x0, y0, z1)),
        ((x0, y0, z0), (x0, y0, z1)), ((x1, y0, z0), (x1, y0, z1)),
        ((x1, y1, z0), (x1, y1, z1)), ((x0, y1, z0), (x0, y1, z1)),
    ]
    xs, ys, zs = [], [], []
    for a, b in edges:
        xs += [a[0], b[0], None]
        ys += [a[1], b[1], None]
        zs += [a[2], b[2], None]
    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line=dict(color=color, width=3, dash="dash"),
        name=name,
        hoverinfo="skip",
    )


def _plane_cut_trace(p1_xy, p2_xy, z_range, name: str, color):
    """Render a gate's plane_cut as a vertical quadrilateral spanning z_range."""
    import plotly.graph_objects as go
    p1 = np.array([p1_xy[0], p1_xy[1], z_range[0]])
    p2 = np.array([p2_xy[0], p2_xy[1], z_range[0]])
    p3 = np.array([p2_xy[0], p2_xy[1], z_range[1]])
    p4 = np.array([p1_xy[0], p1_xy[1], z_range[1]])
    return go.Mesh3d(
        x=[p1[0], p2[0], p3[0], p4[0]],
        y=[p1[1], p2[1], p3[1], p4[1]],
        z=[p1[2], p2[2], p3[2], p4[2]],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color=color, opacity=0.25,
        name=name,
        hovertemplate=(name + "<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>"),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scene", required=True, type=Path)
    p.add_argument("--course", type=Path, default=None,
                   help="Optional course YAML to overlay (waypoints + planned spline).")
    p.add_argument("--courses-dir", type=Path, default=None,
                   help="Directory of course YAMLs to overlay together (each gets its "
                        "own color). Useful for visualizing a perturbation spread.")
    p.add_argument("--start", type=str, default="[0, 0, 1.5]",
                   help="Implied start waypoint in MOCAP (JSON array). Default [0, 0, 1.5].")
    p.add_argument("--out", required=True, type=Path,
                   help="Output HTML path.")
    p.add_argument("--max-points-per-cloud", type=int, default=4000,
                   help="Subsample each scene_objects PLY to this many points.")
    p.add_argument("--full-scene-ply", type=Path, default=None,
                   help="Path to a whole-scene PLY (e.g. mocap_processed/sparse_pc.ply). "
                        "When omitted, the inspector auto-detects the SfM sparse cloud "
                        "under the scene's gsplat_data_cwd. Use 'none' to disable.")
    p.add_argument("--max-full-scene-points", type=int, default=8000,
                   help="Subsample the full-scene cloud to at most this many points.")
    p.add_argument("--marker", action="append", default=[], metavar="NAME:[x,y,z]",
                   help="Repeatable; adds an ad-hoc labelled point (MOCAP coords) to "
                        "the plot. Example: --marker 'extra:[2.5, -0.25, 0]'.")
    p.add_argument("--body-frame-scale", type=float, default=0.25,
                   help="Length (m, mocap) of the body-frame axis arrows drawn at "
                        "each waypoint when --course is set. Set to 0 to disable.")
    p.add_argument("--open-browser", action="store_true",
                   help="Open the HTML in the default browser after writing.")
    args = p.parse_args(argv)

    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise SystemExit(
            "plotly is required for this CLI: pip install plotly  "
            "(it's in falsify's pyproject.toml; if the venv is fresh, run uv sync)"
        ) from e

    from falsify.io import build_frame_graph, load_yaml
    from falsify.visualization import read_ply, subsample

    scene_cfg = load_yaml(args.scene)
    scene_dir = args.scene.parent
    fg = build_frame_graph(scene_cfg, base_path=scene_dir)
    mocap = fg.frame("mocap")

    landmarks = _scene_landmarks(scene_cfg, scene_dir)
    edits = load_scene_edits(scene_cfg)
    if edits:
        print(f"[scene_edits] {len(edits)} edit(s) declared: "
              f"{[e.name for e in edits]}")

    traces = []

    # 0. Full-scene SfM sparse cloud (the whole room — context for everything else).
    full_scene_path: Optional[Path] = None
    if args.full_scene_ply is not None and str(args.full_scene_ply).lower() != "none":
        full_scene_path = args.full_scene_ply
    elif args.full_scene_ply is None and "gsplat_data_cwd" in scene_cfg:
        candidate = Path(scene_cfg["gsplat_data_cwd"])
        if not candidate.is_absolute():
            candidate = (scene_dir / candidate).resolve()
        candidate = candidate / "mocap_processed" / "sparse_pc.ply"
        if candidate.exists():
            full_scene_path = candidate
    if full_scene_path is not None:
        full_scene_cloud = read_ply(full_scene_path, mocap)
        if args.max_full_scene_points > 0:
            full_scene_cloud = subsample(full_scene_cloud, args.max_full_scene_points)
        # If the PLY embedded colors, surface them — otherwise muted gray.
        if full_scene_cloud.colors is not None:
            n = full_scene_cloud.points.shape[0]
            cols = np.clip(full_scene_cloud.colors * 255.0, 0, 255).astype(int)
            color_arr = [f"rgb({r},{g},{b})" for r, g, b in cols]
        else:
            color_arr = "rgb(180,180,180)"
        traces.append(go.Scatter3d(
            x=full_scene_cloud.points[:, 0],
            y=full_scene_cloud.points[:, 1],
            z=full_scene_cloud.points[:, 2],
            mode="markers",
            marker=dict(size=1.4, color=color_arr, opacity=0.45),
            name="full_scene (sparse SfM)",
            hovertemplate=("scene<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>"),
        ))

    # 1. Scene object point clouds (in mocap).
    for entry in scene_cfg.get("scene_objects", []) or []:
        ply_path = Path(entry["ply"])
        if not ply_path.is_absolute():
            ply_path = (scene_dir / ply_path).resolve()
        cloud = read_ply(ply_path, mocap)
        if args.max_points_per_cloud > 0:
            cloud = subsample(cloud, args.max_points_per_cloud)
        # Apply scene edits whose `applies_to_scene_objects` includes this name
        # — keeps the inspector view in sync with what the renderer will produce.
        if edits:
            from falsify.geometry import PointCloud
            new_pts = apply_edits_to_scene_object(entry["name"], cloud.points, edits, fg)
            # DuplicateAABB grows the cloud (appends a transformed copy);
            # mirror the same growth on colors so PointCloud's invariant
            # (len(colors) == len(points)) holds. The repeat factor is
            # always an integer (1 = move-only, 2 = one duplicate, …).
            new_colors = cloud.colors
            if new_colors is not None and new_pts.shape[0] != cloud.points.shape[0]:
                if new_pts.shape[0] % cloud.points.shape[0] != 0:
                    raise ValueError(
                        f"scene_edit on {entry['name']}: cloud grew from "
                        f"{cloud.points.shape[0]} → {new_pts.shape[0]} points, "
                        f"not an integer multiple"
                    )
                reps = new_pts.shape[0] // cloud.points.shape[0]
                new_colors = np.tile(cloud.colors, (reps, 1))
            cloud = PointCloud(points=new_pts, frame=cloud.frame, colors=new_colors)
        # Cloud's own colors look muddy in plotly; use the embodiment tint.
        color = "rgb({},{},{})".format(*[int(255 * c) for c in entry.get("color", (0.5, 0.5, 0.5))])
        traces.append(_pc_trace(cloud.points, entry["name"], color, size=2))

    # 2a. Scene-edit AABBs themselves — main + extra includes (orange),
    # oriented includes (yellow), exclusion boxes (purple, dashed),
    # oriented excludes (purple dotted).
    def _oriented_box_edges_trace(box, name, color, dash="solid"):
        import plotly.graph_objects as go
        corners = box.corners()
        edge_idx = [
            (0, 1), (1, 3), (3, 2), (2, 0),
            (4, 5), (5, 7), (7, 6), (6, 4),
            (0, 4), (1, 5), (3, 7), (2, 6),
        ]
        xs, ys, zs = [], [], []
        for a, b in edge_idx:
            xs += [corners[a, 0], corners[b, 0], None]
            ys += [corners[a, 1], corners[b, 1], None]
            zs += [corners[a, 2], corners[b, 2], None]
        return go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(color=color, width=4, dash=dash),
            name=name, hoverinfo="skip",
        )

    if edits:
        for edit in edits:
            mn = np.asarray(edit.target_aabb_min)
            mx = np.asarray(edit.target_aabb_max)
            traces.append(_aabb_edges_trace(mn, mx, f"edit:{edit.name}:include", "orange"))
            for k, box in enumerate(edit.include_aabbs):
                traces.append(_aabb_edges_trace(
                    np.asarray(box.min), np.asarray(box.max),
                    f"edit:{edit.name}:include_extra_{k}", "orange",
                ))
            for k, box in enumerate(edit.oriented_include_aabbs):
                traces.append(_oriented_box_edges_trace(
                    box, f"edit:{edit.name}:oriented_include_{k}", "yellow",
                ))
            for k, box in enumerate(edit.exclude_aabbs):
                traces.append(_aabb_edges_trace(
                    np.asarray(box.min), np.asarray(box.max),
                    f"edit:{edit.name}:exclude_{k}", "purple",
                ))
            for k, box in enumerate(edit.oriented_exclude_aabbs):
                traces.append(_oriented_box_edges_trace(
                    box, f"edit:{edit.name}:oriented_exclude_{k}", "purple", dash="dot",
                ))

    # 2. AABB wireframes + centers from objects_summary.json.
    # If a scene_edit applies to this landmark, render BOTH the original
    # (dashed, dim) and the edited AABB so the user sees the displacement.
    aabb_colors = ["red", "blue", "magenta", "cyan"]
    for i, (name, obj) in enumerate(landmarks.items()):
        color = aabb_colors[i % len(aabb_colors)]
        mn = np.asarray(obj["aabb_min"])
        mx = np.asarray(obj["aabb_max"])
        traces.append(_aabb_edges_trace(mn, mx, f"{name}_aabb", color))
        traces.append(_markers_trace(
            [(f"{name}_center", np.asarray(obj["center"]))],
            color=color, size=4, symbol="diamond",
        ))
        if edits:
            # Apply edits to the 8 corners and re-axis-align — gives a loose
            # box around where the moved AABB now lies.
            corners = np.array([
                [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
                [mn[0], mx[1], mn[2]], [mx[0], mx[1], mn[2]],
                [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
                [mn[0], mx[1], mx[2]], [mx[0], mx[1], mx[2]],
            ])
            moved = apply_edits_to_scene_object(name, corners, edits, fg)
            # A DuplicateAABB grows the cloud to 2× (original + copy), so the
            # shape no longer matches `corners`. Compare on the originals
            # (which always sit at moved[:8]) and draw an edited-AABB box for
            # EACH copy block of 8 corners.
            n_orig = corners.shape[0]
            n_copies = moved.shape[0] // n_orig
            for k in range(n_copies):
                block = moved[k * n_orig : (k + 1) * n_orig]
                if k == 0 and np.allclose(block, corners):
                    # First block is the original (unchanged) — skip.
                    continue
                label = f"{name}_aabb (edited)" if n_copies <= 2 else f"{name}_aabb (copy {k})"
                traces.append(_aabb_edges_trace(
                    block.min(axis=0), block.max(axis=0), label, color,
                ))

    # 3. Gate plane_cut overlays (drone flies through these).
    midpoints = []
    for name, obj in landmarks.items():
        plane = obj.get("plane_cut")
        if not plane:
            continue
        p1 = plane["P1"]
        p2 = plane["P2"]
        z_range = (float(obj["aabb_min"][2]), float(obj["aabb_max"][2]))
        traces.append(_plane_cut_trace(p1, p2, z_range, f"{name}_plane", "yellow"))
        # Midpoint at z=1.5 (drone-hover height) as a candidate "gate waypoint".
        midpoint_xy = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
        midpoints.append((f"{name}_plane_mid (z=1.5)",
                          np.array([midpoint_xy[0], midpoint_xy[1], 1.5])))

    # 4. Origin + implied start + any --marker entries.
    start = np.asarray(json.loads(args.start), dtype=np.float64)
    landmarks_pts = [
        ("origin (ArUco)", np.array([0.0, 0.0, 0.0])),
        ("start", start),
    ] + midpoints
    for spec in args.marker:
        if ":" not in spec:
            raise SystemExit(f"--marker must be 'NAME:[x,y,z]'; got {spec!r}")
        name, coords = spec.split(":", 1)
        landmarks_pts.append((name.strip(), np.asarray(json.loads(coords), dtype=np.float64)))
    traces.append(_markers_trace(
        landmarks_pts, color="lime", size=8, symbol="x",
    ))

    # 5. Optional course overlay(s).
    course_paths: list[Path] = []
    if args.course is not None:
        course_paths.append(args.course)
    if args.courses_dir is not None:
        course_paths.extend(sorted(args.courses_dir.glob("*.yaml")))

    if course_paths:
        from falsify.planning import load_course, plan_spline
        from falsify.geometry import PointCloud
        from falsify.geometry import Trajectory as GeoTraj
        import colorsys

        for ci, cpath in enumerate(course_paths):
            course = load_course(cpath)
            # Course waypoints — convert to mocap if needed for the plot.
            wps_xyz = course.positions
            if course.frame != "mocap":
                wps_xyz = fg.convert(
                    PointCloud(points=wps_xyz, frame=fg.frame(course.frame)),
                    to="mocap",
                ).points

            # Distinct color per course (HSV around the wheel).
            if len(course_paths) == 1:
                marker_color, spline_color = "white", "yellow"
            else:
                hue = (ci / max(1, len(course_paths))) % 1.0
                r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.95)
                spline_color = f"rgb({int(255*r)},{int(255*g)},{int(255*b)})"
                marker_color = spline_color

            wp_labels = [
                (f"{course.name}.{wp.name}", wps_xyz[i])
                for i, wp in enumerate(course.waypoints)
            ]
            traces.append(_markers_trace(
                wp_labels, color=marker_color, size=6, symbol="circle",
            ))

            # Draw the FRD body frame at each waypoint, but only for the
            # single-course case — for a courses-dir overlay this would be
            # visual chaos. Skip if --body-frame-scale 0.
            if args.courses_dir is None and args.body_frame_scale > 0:
                yaws_mocap = course.resolved_yaws()
                # `resolved_yaws` returns angles in the course frame; the
                # scene's course is in mocap so we can use them directly.
                traces.extend(_body_frames_traces(
                    wps_xyz, yaws_mocap,
                    length=float(args.body_frame_scale),
                    name_prefix=f"body[{course.name}]",
                ))

            # Spline polyline (mocap).
            traj = plan_spline(course, fg)
            geo = GeoTraj(
                times=traj.times,
                positions=traj.positions_ned,
                frame=fg.frame("ned"),
            )
            in_mocap = fg.convert(geo, to="mocap")
            traces.append(_polyline_trace(
                in_mocap.positions, f"spline:{course.name}",
                spline_color, width=4,
            ))

    layout = go.Layout(
        title=f"falsify scene inspector — {scene_cfg.get('scene_key', args.scene.stem)} (mocap)",
        scene=dict(
            xaxis_title="x_mocap",
            yaxis_title="y_mocap",
            zaxis_title="z_mocap",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(itemsizing="constant"),
    )
    fig = go.Figure(data=[t for t in traces if t is not None], layout=layout)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.out, include_plotlyjs="cdn", auto_open=False)
    print(f"[done] wrote {args.out}")
    print()
    print("Landmarks (MOCAP) extracted for course authoring:")
    for name, obj in landmarks.items():
        print(f"  {name}_center        = {obj['center']}")
        print(f"  {name}_aabb          = min {obj['aabb_min']}  max {obj['aabb_max']}")
        if obj.get("plane_cut"):
            p1, p2 = obj["plane_cut"]["P1"], obj["plane_cut"]["P2"]
            mid_xy = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            print(f"  {name}_plane_posts   = {p1}  ↔  {p2}")
            print(f"  {name}_plane_mid    = {[round(mid_xy[0], 3), round(mid_xy[1], 3), 1.5]}   (drone-height waypoint)")
    print(f"  start (--start)     = {start.tolist()}")

    if args.open_browser:
        webbrowser.open(args.out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
