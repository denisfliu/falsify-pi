"""PointCloud / Trajectory → PLY writers + small helpers.

Frame-aware: the writer encodes the frame name into the PLY header as a
comment so a reader (and a human) can tell which frame a `.ply` is in.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from falsify.geometry import Frame, PointCloud, Trajectory


# Map PLY property type names → (numpy dtype, struct format code, byte size).
_PLY_DTYPES: dict[str, tuple[str, str, int]] = {
    "char":   ("i1", "b", 1), "int8":    ("i1", "b", 1),
    "uchar":  ("u1", "B", 1), "uint8":   ("u1", "B", 1),
    "short":  ("i2", "h", 2), "int16":   ("i2", "h", 2),
    "ushort": ("u2", "H", 2), "uint16":  ("u2", "H", 2),
    "int":    ("i4", "i", 4), "int32":   ("i4", "i", 4),
    "uint":   ("u4", "I", 4), "uint32":  ("u4", "I", 4),
    "float":  ("f4", "f", 4), "float32": ("f4", "f", 4),
    "double": ("f8", "d", 8), "float64": ("f8", "d", 8),
}


def trajectory_to_pointcloud(
    traj: Trajectory,
    color: Optional[Sequence[float]] = None,
) -> PointCloud:
    """View a `Trajectory` as a `PointCloud` in the same frame."""
    colors = None
    if color is not None:
        c = np.asarray(color, dtype=np.float64)
        if c.shape != (3,):
            raise ValueError("color must be a 3-tuple")
        colors = np.tile(c[None, :], (traj.positions.shape[0], 1))
    return PointCloud(points=traj.positions.copy(), frame=traj.frame, colors=colors)


def write_ply(pc: PointCloud, path: str | Path) -> Path:
    """Write a `PointCloud` to an ASCII PLY file, recording its frame in a comment."""
    path = Path(path)
    n = pc.points.shape[0]
    has_color = pc.colors is not None
    header = [
        "ply",
        "format ascii 1.0",
        f"comment falsify frame: {pc.frame.name}",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_color:
        header += [
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
    header.append("end_header")
    lines = ["\n".join(header)]

    if has_color:
        colors = pc.colors
        if colors.dtype.kind == "f" and colors.max() <= 1.0 + 1e-6:
            colors = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
        else:
            colors = np.clip(colors, 0, 255).astype(np.uint8)
        for p, c in zip(pc.points, colors):
            lines.append(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}")
    else:
        for p in pc.points:
            lines.append(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def read_ply(path: str | Path, frame: Frame) -> PointCloud:
    """Read an ASCII or binary-little-endian PLY into a `PointCloud`.

    The PLY's geometry is taken from the ``vertex`` element (``x``, ``y``,
    ``z`` properties). Color properties (``red``/``green``/``blue``, possibly
    8-bit) are loaded into ``colors`` when present and normalized to [0, 1]
    floats. The frame is **caller-supplied** — PLY files don't reliably encode
    frame metadata, so the call site must know what frame the file lives in
    (typically via the scene YAML's ``scene_objects`` block).
    """
    path = Path(path)
    with open(path, "rb") as f:
        # Header is always ASCII; read until 'end_header'.
        header_lines: list[str] = []
        while True:
            line_bytes = f.readline()
            if not line_bytes:
                raise ValueError(f"{path}: unexpected EOF in PLY header")
            line = line_bytes.decode("ascii", errors="replace").rstrip("\r\n")
            header_lines.append(line)
            if line.strip() == "end_header":
                break
        body_offset = f.tell()
        body = f.read()

    if not header_lines or header_lines[0].strip() != "ply":
        raise ValueError(f"{path}: not a PLY file")

    fmt = None
    elements: list[dict] = []
    current: dict | None = None
    for line in header_lines[1:]:
        tokens = line.split()
        if not tokens or tokens[0] == "comment":
            continue
        if tokens[0] == "format":
            fmt = tokens[1]
        elif tokens[0] == "element":
            current = {"name": tokens[1], "count": int(tokens[2]), "props": []}
            elements.append(current)
        elif tokens[0] == "property":
            if current is None:
                raise ValueError(f"{path}: property before element")
            if tokens[1] == "list":
                # list properties (e.g. face vertex_indices) — we don't need them
                # for points, but record so reading can step over them if present.
                current["props"].append({
                    "kind": "list",
                    "count_type": tokens[2],
                    "value_type": tokens[3],
                    "name": tokens[4],
                })
            else:
                current["props"].append({"kind": "scalar", "type": tokens[1], "name": tokens[2]})
        elif tokens[0] == "end_header":
            break

    vertex = next((e for e in elements if e["name"] == "vertex"), None)
    if vertex is None:
        raise ValueError(f"{path}: no 'vertex' element")
    names = [p["name"] for p in vertex["props"] if p["kind"] == "scalar"]
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError(f"{path}: vertex missing x/y/z properties")

    has_color = {"red", "green", "blue"}.issubset(names)

    if fmt == "binary_little_endian":
        # Each scalar property is fixed size; build a struct format string.
        scalar_props = [p for p in vertex["props"] if p["kind"] == "scalar"]
        # Reject any list properties on the vertex element — none of our
        # inputs have them; this keeps the reader simple.
        if any(p["kind"] == "list" for p in vertex["props"]):
            raise NotImplementedError(f"{path}: list properties on vertex not supported")
        dtype_fields = []
        for p in scalar_props:
            np_dtype, _, _ = _PLY_DTYPES[p["type"]]
            dtype_fields.append((p["name"], np_dtype))
        dt = np.dtype(dtype_fields)
        expected = dt.itemsize * vertex["count"]
        if len(body) < expected:
            raise ValueError(f"{path}: truncated body (got {len(body)}, want ≥ {expected})")
        arr = np.frombuffer(body[:expected], dtype=dt)
        pts = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
        colors = None
        if has_color:
            colors = np.stack([arr["red"], arr["green"], arr["blue"]], axis=1).astype(np.float64) / 255.0
    elif fmt == "ascii":
        rows = [ln.split() for ln in body.decode("ascii").splitlines() if ln.strip()]
        rows = rows[: vertex["count"]]
        scalar_props = [p for p in vertex["props"] if p["kind"] == "scalar"]
        col_idx = {p["name"]: i for i, p in enumerate(scalar_props)}
        pts = np.array([[float(r[col_idx["x"]]), float(r[col_idx["y"]]), float(r[col_idx["z"]])] for r in rows])
        colors = None
        if has_color:
            colors = np.array(
                [[float(r[col_idx["red"]]), float(r[col_idx["green"]]), float(r[col_idx["blue"]])]
                 for r in rows]
            )
            if colors.max() > 1.0 + 1e-6:
                colors = colors / 255.0
    else:
        raise ValueError(f"{path}: unsupported PLY format {fmt!r}")

    return PointCloud(points=pts, frame=frame, colors=colors)


def subsample(pc: PointCloud, max_points: int, *, seed: int = 0) -> PointCloud:
    """Return at most ``max_points`` points, sampled deterministically."""
    n = pc.points.shape[0]
    if n <= max_points:
        return pc
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    idx.sort()
    return PointCloud(
        points=pc.points[idx],
        frame=pc.frame,
        colors=None if pc.colors is None else pc.colors[idx],
    )


def stack_pointclouds(clouds: Iterable[PointCloud]) -> PointCloud:
    """Concatenate point clouds **in the same frame** into one."""
    clouds = list(clouds)
    if not clouds:
        raise ValueError("no clouds to stack")
    frame = clouds[0].frame
    for c in clouds[1:]:
        if c.frame.name != frame.name:
            raise ValueError(
                f"stack_pointclouds: frame mismatch ({c.frame.name!r} vs {frame.name!r})"
            )
    pts = np.concatenate([c.points for c in clouds], axis=0)
    if any(c.colors is None for c in clouds):
        return PointCloud(points=pts, frame=frame)
    cols = np.concatenate([c.colors for c in clouds], axis=0)
    return PointCloud(points=pts, frame=frame, colors=cols)
