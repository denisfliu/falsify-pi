"""YAML config loading and `FrameGraph` construction.

Adding a new frame or transform to a scene is a YAML edit. See
``src/falsify/geometry/CLAUDE.md`` for the schema.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from falsify.geometry import (
    Frame,
    FrameGraph,
    frame_by_name,
    get_loader,
)


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    return yaml.safe_load(path.read_text())


def build_frame_graph(scene_cfg: dict, *, base_path: Path | None = None) -> FrameGraph:
    """Build a `FrameGraph` from a parsed scene YAML.

    Expected schema (see ``configs/scenes/*.yaml`` for examples)::

        frames:
          - { name: ned, notes: "..." }
          - { name: mocap }
          - ...
        transforms:
          - { src: ned, dst: mocap, type: permutation, preset: "perm5" }
          - { src: mocap, dst: colmap, type: sim3_file, path: ... }
          - ...

    Frame *order* of declaration is irrelevant — the graph is rebuilt from
    scratch each call. Transforms reference frames by name; unknown names
    raise immediately.
    """
    base_path = Path(base_path) if base_path is not None else Path.cwd()

    graph = FrameGraph()

    # 1. Register all frames first so transform loaders can look them up.
    declared: dict[str, Frame] = {}
    for entry in scene_cfg.get("frames", []):
        if isinstance(entry, str):
            entry = {"name": entry}
        frame = Frame(
            name=entry["name"],
            convention=entry.get("convention", "right_handed"),
            notes=entry.get("notes", ""),
        )
        declared[frame.name] = frame
        graph.register_frame(frame)

    def lookup(name: str) -> Frame:
        if name in declared:
            return declared[name]
        # Fall back to canonical defaults — convenient but optional.
        return frame_by_name(name)

    # 2. Load each transform via its registered loader and register the edge.
    for spec in scene_cfg.get("transforms", []):
        loader = get_loader(spec["type"])
        edge = loader(spec, lookup, base_path)
        graph.register_edge(edge)

    return graph
