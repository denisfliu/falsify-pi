"""The `FrameGraph` — runtime registry of frames and transforms.

One `FrameGraph` instance is constructed per scene. It owns every named
frame and every edge (SE3 or Sim3) declared in the scene YAML, plus any
edges added at runtime (e.g. the body→camera SE3s registered by `CameraRig`).

Conversion is BFS over registered edges, with inverse edges derived
lazily. Composition uses the `@` rules in `transforms.py` (SE3 ∘ SE3 = SE3,
anything-with-Sim3 = Sim3).
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, Tuple, Union

from .frames import Frame, frame_by_name
from .transforms import SE3, Sim3
from .types import Point, Pose, Trajectory, PointCloud, assert_frame

Edge = Union[SE3, Sim3]
Transformable = Union[Point, Pose, Trajectory, PointCloud]


class FrameGraph:
    """A registry of frames and transforms with auto-composing conversion.

    Frames are addressed by name (str). Edges are directional but inverse
    edges are derived automatically — registering ``ned → mocap`` makes
    ``mocap → ned`` available.
    """

    def __init__(self) -> None:
        self._frames: Dict[str, Frame] = {}
        self._edges: Dict[Tuple[str, str], Edge] = {}

    # ---- registration --------------------------------------------------

    def register_frame(self, frame: Frame) -> None:
        existing = self._frames.get(frame.name)
        if existing is not None and existing != frame:
            raise ValueError(
                f"frame {frame.name!r} already registered with different metadata: "
                f"{existing} vs {frame}"
            )
        self._frames[frame.name] = frame

    def register_edge(self, T: Edge) -> None:
        src, dst = T.src.name, T.dst.name
        if src not in self._frames or dst not in self._frames:
            raise ValueError(
                f"register_edge: frames {src!r}/{dst!r} must be registered first"
            )
        if (src, dst) in self._edges:
            raise ValueError(f"edge {src}→{dst} already registered")
        if (dst, src) in self._edges:
            raise ValueError(
                f"edge {dst}→{src} already registered (inverse derived automatically); "
                f"add only one direction"
            )
        self._edges[(src, dst)] = T

    # ---- introspection -------------------------------------------------

    @property
    def frames(self) -> tuple[Frame, ...]:
        return tuple(self._frames.values())

    def frame(self, name: str) -> Frame:
        if name not in self._frames:
            raise KeyError(f"frame {name!r} not in graph (have: {sorted(self._frames)})")
        return self._frames[name]

    def has_path(self, src: str, dst: str) -> bool:
        try:
            self._find_path(src, dst)
            return True
        except KeyError:
            return False

    def describe(self) -> str:
        lines = [f"FrameGraph ({len(self._frames)} frames, {len(self._edges)} edges):"]
        for f in self._frames.values():
            note = f" — {f.notes}" if f.notes else ""
            lines.append(f"  frame  {f.name}{note}")
        for (s, d), T in self._edges.items():
            kind = "Sim3" if isinstance(T, Sim3) else "SE3"
            lines.append(f"  edge   {s} → {d}  ({kind})")
        return "\n".join(lines)

    # ---- conversion ----------------------------------------------------

    def convert(self, value: Transformable, to: str) -> Transformable:
        """Convert ``value`` into the frame named ``to``.

        Composes all transforms along the BFS path through the graph. If
        ``value.frame.name == to`` this is a no-op (returns ``value``).
        """
        assert_frame(value, value.frame)  # sanity
        src = value.frame.name
        if src == to:
            return value
        T_total = self._composed_transform(src, to)
        return T_total @ value

    def transform(self, src: str, dst: str) -> Edge:
        """Return the composed transform from ``src`` to ``dst``."""
        return self._composed_transform(src, dst)

    # ---- internals -----------------------------------------------------

    def _composed_transform(self, src: str, dst: str) -> Edge:
        path = self._find_path(src, dst)
        # path is a list of frame names; build the transform step by step.
        T: Edge | None = None
        for a, b in zip(path[:-1], path[1:]):
            step = self._edge_between(a, b)
            T = step if T is None else (step @ T)
        assert T is not None
        return T

    def _edge_between(self, a: str, b: str) -> Edge:
        if (a, b) in self._edges:
            return self._edges[(a, b)]
        if (b, a) in self._edges:
            return self._edges[(b, a)].inv()
        raise KeyError(f"no edge between {a!r} and {b!r}")

    def _neighbors(self, name: str) -> Iterable[str]:
        for s, d in self._edges:
            if s == name:
                yield d
            elif d == name:
                yield s

    def _find_path(self, src: str, dst: str) -> list[str]:
        if src not in self._frames or dst not in self._frames:
            raise KeyError(f"unknown frame in {src!r}→{dst!r}")
        if src == dst:
            return [src]
        # BFS
        prev: Dict[str, str] = {src: src}
        q = deque([src])
        while q:
            cur = q.popleft()
            for nxt in self._neighbors(cur):
                if nxt in prev:
                    continue
                prev[nxt] = cur
                if nxt == dst:
                    # reconstruct
                    path = [dst]
                    while path[-1] != src:
                        path.append(prev[path[-1]])
                    path.reverse()
                    return path
                q.append(nxt)
        raise KeyError(f"no path from {src!r} to {dst!r}")
