"""Pluggable transform loaders.

A scene YAML declares its frame transforms like:

    transforms:
      - { src: ned, dst: mocap, type: permutation, preset: "perm5" }
      - { src: mocap, dst: colmap, type: sim3_file, path: data/alignment/.../colmap_to_mocap_sim3.json, invert: true }
      - { src: colmap, dst: ns,   type: dataparser, path: .../dataparser_transforms.json }
      - { src: cam_body, dst: ned, type: se3_inline, R: [[...]], t: [...] }

Each ``type`` key maps to a loader function registered here. Loaders take
``(spec_dict, frame_lookup, base_path)`` and return either an `SE3` or `Sim3`.

Third-party loaders register themselves with ``register_loader("custom", fn)``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict

import numpy as np

from .frames import Frame
from .presets import axis_permutation
from .transforms import SE3, Sim3


# A loader takes (spec, frame_lookup, base_path) → SE3 | Sim3.
# ``frame_lookup`` resolves a frame name to a Frame (used for src/dst).
# ``base_path`` is the directory of the scene YAML (for resolving relative paths).
LoaderFn = Callable[[dict, Callable[[str], Frame], Path], "SE3 | Sim3"]


_REGISTRY: Dict[str, LoaderFn] = {}


def register_loader(type_name: str, fn: LoaderFn) -> None:
    if type_name in _REGISTRY:
        raise ValueError(f"loader {type_name!r} already registered")
    _REGISTRY[type_name] = fn


def get_loader(type_name: str) -> LoaderFn:
    if type_name not in _REGISTRY:
        raise KeyError(
            f"unknown transform type {type_name!r}; "
            f"available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[type_name]


def available_loaders() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


# ---------------------------------------------------------------------------
# Built-in loaders
# ---------------------------------------------------------------------------


def _resolve_path(p: str, base: Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (base / path).resolve()


def _src_dst(spec: dict, lookup) -> tuple[Frame, Frame]:
    return lookup(spec["src"]), lookup(spec["dst"])


def _load_permutation(spec, lookup, base):
    """SE3 with R = axis_permutation(preset), t = 0."""
    src, dst = _src_dst(spec, lookup)
    R = axis_permutation(spec["preset"])
    T = SE3(R=R, t=np.zeros(3), src=src, dst=dst)
    return T.inv() if spec.get("invert", False) else T


def _load_se3_inline(spec, lookup, base):
    src, dst = _src_dst(spec, lookup)
    R = np.asarray(spec["R"], dtype=np.float64)
    t = np.asarray(spec.get("t", [0.0, 0.0, 0.0]), dtype=np.float64)
    T = SE3(R=R, t=t, src=src, dst=dst)
    return T.inv() if spec.get("invert", False) else T


def _load_sim3_inline(spec, lookup, base):
    src, dst = _src_dst(spec, lookup)
    s = float(spec.get("scale", spec.get("s", 1.0)))
    R = np.asarray(spec["R"], dtype=np.float64)
    t = np.asarray(spec.get("t", [0.0, 0.0, 0.0]), dtype=np.float64)
    T = Sim3(s=s, R=R, t=t, src=src, dst=dst)
    return T.inv() if spec.get("invert", False) else T


def _load_se3_file(spec, lookup, base):
    """Load an SE3 from a JSON file with keys ``R`` (3x3) and ``t`` (3)."""
    src, dst = _src_dst(spec, lookup)
    path = _resolve_path(spec["path"], base)
    data = json.loads(path.read_text())
    T = SE3(R=np.asarray(data["R"]), t=np.asarray(data["t"]), src=src, dst=dst)
    return T.inv() if spec.get("invert", False) else T


def _load_sim3_file(spec, lookup, base):
    """Load a Sim3 from a JSON file with keys ``scale`` (float), ``R`` (3x3), ``t`` (3).

    Matches SousVide's ``colmap_to_mocap_sim3.json`` format.
    """
    src, dst = _src_dst(spec, lookup)
    path = _resolve_path(spec["path"], base)
    data = json.loads(path.read_text())
    T = Sim3(
        s=float(data["scale"]),
        R=np.asarray(data["R"]),
        t=np.asarray(data["t"]),
        src=src,
        dst=dst,
    )
    return T.inv() if spec.get("invert", False) else T


def _load_dataparser(spec, lookup, base):
    """Load a Nerfstudio ``dataparser_transforms.json`` (COLMAP → NS).

    Nerfstudio stores a 3×4 affine ``transform`` (R | t) and a ``scale`` float.
    Applied as ``p_ns = scale * (R @ p_colmap + t)`` — equivalently a Sim3 with
    ``s = scale``, ``R = R``, ``t = scale * t`` so ``Sim3.apply`` (which does
    ``s*(R p) + t``) matches Nerfstudio's convention.
    """
    src, dst = _src_dst(spec, lookup)
    path = _resolve_path(spec["path"], base)
    data = json.loads(path.read_text())
    transform = np.asarray(data["transform"], dtype=np.float64)  # (3, 4)
    if transform.shape != (3, 4):
        raise ValueError(f"dataparser transform expected (3,4), got {transform.shape}")
    scale = float(data["scale"])
    R = transform[:3, :3]
    t_pre_scale = transform[:3, 3]
    T = Sim3(s=scale, R=R, t=scale * t_pre_scale, src=src, dst=dst)
    return T.inv() if spec.get("invert", False) else T


# Register built-ins.
register_loader("permutation", _load_permutation)
register_loader("se3_inline", _load_se3_inline)
register_loader("sim3_inline", _load_sim3_inline)
register_loader("se3_file", _load_se3_file)
register_loader("sim3_file", _load_sim3_file)
register_loader("dataparser", _load_dataparser)
