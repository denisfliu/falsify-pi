"""GSplat renderer wrapper — frame-aware boundary around `figs.render.gsplat.GSPlat`.

FiGS' renderer expects a 4×4 camera-to-world matrix in **FiGS NED** and
internally applies a ``Tw2g`` (NED → gsplat-internal NS). On the falsify
side we receive a ``Pose`` whose ``.frame`` must be ``"ned"`` — the only
boundary conversion happens here, and the rule is enforced.

Tw2g source of truth: the active `FrameGraph`. Different FiGS submodule SHAs
have historically loaded ``dataparser_transforms.json`` from disk
inconsistently (the falsify-pinned FiGS doesn't load it at all, leaving
``Tw2g`` as just the axis-flip and rendering empty space). To insulate the
project from this, we override ``self._impl.Tw2g`` with the composed
NED → NS transform from the `FrameGraph` whenever one is supplied. This
keeps the project's frame contract honored — *one* source of truth for
every coordinate conversion.

FiGS / nerfstudio / CUDA are imported lazily so this module stays loadable
on machines without the full stack.

Nerfstudio's ``eval_setup`` resolves the training config's ``data:`` field
relative to **the current working directory** (the gate-scene exports store
``data: mocap_processed`` as a bare relative string). Pass ``data_cwd`` to
chdir into the directory that contains the training-data folder during
config load. The original cwd is restored before the constructor returns.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import numpy as np

from falsify.geometry import FrameGraph, Pose, Sim3, SE3, assert_frame
from .scene_edits import SceneEdit, apply_edits_to_pipeline, load_scene_edits


def _resolve_rel(value: str | Path, base: Path) -> Path:
    """Resolve a possibly-relative path against ``base``."""
    p = Path(value)
    return p if p.is_absolute() else (base / p).resolve()


@contextmanager
def _chdir(path: Optional[Path]):
    """Temporarily change cwd. No-op when ``path`` is None."""
    if path is None:
        yield
        return
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


class GSplatRenderer:
    """Render a single camera against a Nerfstudio Gaussian-splat workspace.

    Construction lazy-imports FiGS / nerfstudio so the rest of falsify can be
    imported on machines without CUDA.
    """

    def __init__(
        self,
        gsplat_path: str | Path,
        world_frame: str = "ned",
        *,
        data_cwd: str | Path | None = None,
        frame_graph: Optional[FrameGraph] = None,
        gsplat_frame: str = "ns",
        scene_edits: Optional[list[SceneEdit]] = None,
        scene_cfg: Optional[dict] = None,
        near_plane: Optional[float] = None,
    ) -> None:
        # If the caller hands us the scene_cfg, derive scene_edits from it.
        # This is the safe-by-construction path used by ``from_scene_cfg`` —
        # no chance of silently dropping edits because the caller forgot to
        # call ``load_scene_edits`` themselves.
        if scene_cfg is not None:
            declared = load_scene_edits(scene_cfg)
            if scene_edits is None:
                scene_edits = declared or None
            else:
                # Caller passed both — they MUST agree, otherwise one of
                # the two is stale.
                declared_names = [e.name for e in (declared or [])]
                passed_names = [e.name for e in scene_edits]
                if declared_names != passed_names:
                    raise ValueError(
                        f"Inconsistent scene_edits: scene_cfg declares "
                        f"{declared_names} but scene_edits= passed {passed_names}. "
                        "Pass exactly one of (scene_cfg=, scene_edits=); "
                        "prefer GSplatRenderer.from_scene_cfg(scene_cfg, ...)."
                    )

        gsplat_path = Path(gsplat_path)
        data_cwd = Path(data_cwd) if data_cwd is not None else None
        # Lazy import — FiGS pulls in nerfstudio + CUDA bindings.
        from figs.render.gsplat import GSplat as _FiGSGSplat  # type: ignore

        # Lower the splatfacto rasterizer's near-plane clip threshold if
        # requested. gsplat 0.1.13's ``project_gaussians`` has a default
        # ``clip_thresh=0.01`` (gsplat/project_gaussians.py:26) and
        # nerfstudio's splatfacto calls it without overriding — so gaussians
        # within 0.01 view-space-z of the camera get culled, which can eat
        # gate edges and table corners when the camera is close. Monkey-patch
        # the splatfacto module-level binding so every downstream render in
        # this process sees the lowered threshold.
        if near_plane is not None:
            self._patch_splatfacto_near_plane(float(near_plane))
            print(f"[gsplat] near plane (clip_thresh) → {float(near_plane):.5f}")

        with _chdir(data_cwd):
            self._impl = _FiGSGSplat(gsplat_path)
        self._world_frame = world_frame
        self._cameras: dict[tuple, Any] = {}   # cache nerfstudio Camera objects by intrinsics

        # Override FiGS' built-in Tw2g with the composition the FrameGraph
        # computes — see this module's docstring for why. Without this the
        # renderer can silently land in an empty region of the gsplat
        # because FiGS' baked-in Tw2g misses the dataparser step.
        if frame_graph is not None:
            self._set_tw2g_from_graph(frame_graph, world_frame, gsplat_frame)

        # Apply scene edits (Gaussian-level translations / rotations) once
        # at load time. Subsequent renders pick them up automatically since
        # they mutate ``pipeline.model.means`` / ``.quats`` in place.
        if scene_edits and frame_graph is not None:
            n = apply_edits_to_pipeline(self._impl.pipeline, scene_edits, frame_graph)
            print(f"[gsplat] applied {len(scene_edits)} scene edit(s); "
                  f"{n} Gaussians modified")
        elif scene_edits:
            raise ValueError(
                "scene_edits requires frame_graph= to lift the edit into NS"
            )

        # Save a baseline of (means, quats) AFTER any construction-time edits
        # so per-episode environment perturbations can restore to a clean
        # post-static-edit state before applying their own random edits.
        self._baseline_means = None
        self._baseline_quats = None
        self._frame_graph = frame_graph

    @staticmethod
    def _patch_splatfacto_near_plane(clip_thresh: float) -> None:
        """Override ``nerfstudio.models.splatfacto.project_gaussians``'s
        default ``clip_thresh`` so the rasterizer's near plane is set to
        ``clip_thresh`` rather than the upstream default 0.01. Idempotent
        per-process (re-patching reuses the still-bound original).
        """
        import nerfstudio.models.splatfacto as _sf  # type: ignore
        original = getattr(_sf, "_falsify_orig_project_gaussians", _sf.project_gaussians)

        def _patched(*args, **kwargs):
            kwargs.setdefault("clip_thresh", clip_thresh)
            return original(*args, **kwargs)

        _sf._falsify_orig_project_gaussians = original  # type: ignore[attr-defined]
        _sf.project_gaussians = _patched

    @classmethod
    def from_scene_cfg(
        cls,
        scene_cfg: dict,
        *,
        scene_dir: str | Path,
        world_frame: str = "ned",
        gsplat_frame: str = "ns",
        near_plane: Optional[float] = None,
    ) -> "GSplatRenderer":
        """Construct a renderer from a parsed scene YAML.

        Resolves ``gsplat_config_yml`` / ``gsplat_data_cwd`` (both relative
        to ``scene_dir``), builds the FrameGraph, and loads any declared
        ``scene_edits`` — the four boilerplate pieces every caller would
        otherwise have to assemble by hand. Use this in preference to the
        bare constructor: it makes it impossible to forget ``scene_edits``.
        """
        # Lazy import — avoid pulling falsify.io at module import time.
        from falsify.io import build_frame_graph

        scene_dir = Path(scene_dir)
        gsplat_config = _resolve_rel(scene_cfg["gsplat_config_yml"], scene_dir)
        data_cwd = (_resolve_rel(scene_cfg["gsplat_data_cwd"], scene_dir)
                    if "gsplat_data_cwd" in scene_cfg else None)
        fg = build_frame_graph(scene_cfg, base_path=scene_dir)
        return cls(
            gsplat_config,
            world_frame=world_frame,
            data_cwd=data_cwd,
            frame_graph=fg,
            gsplat_frame=gsplat_frame,
            scene_cfg=scene_cfg,
            near_plane=near_plane,
        )

    def _set_tw2g_from_graph(
        self,
        fg: FrameGraph,
        world_frame: str,
        gsplat_frame: str,
    ) -> None:
        T = fg.transform(world_frame, gsplat_frame)
        M = np.eye(4)
        M[:3, :3] = T.R if not isinstance(T, Sim3) else T.s * T.R
        M[:3, 3] = T.t
        self._impl.Tw2g = M

    @property
    def world_frame(self) -> str:
        return self._world_frame

    @property
    def pipeline(self):
        """Nerfstudio pipeline owned by the underlying FiGS GSplat."""
        return self._impl.pipeline

    @property
    def frame_graph(self) -> Optional[FrameGraph]:
        return self._frame_graph

    def snapshot_baseline(self) -> None:
        """Clone the current `pipeline.model.means` / `.quats` into a CPU
        baseline. Idempotent — calling twice keeps the first snapshot, so
        callers can invoke this on every episode without worrying about
        accumulated drift."""
        if self._baseline_means is not None:
            return
        import torch
        with torch.no_grad():
            self._baseline_means = self._impl.pipeline.model.means.detach().clone()
            self._baseline_quats = self._impl.pipeline.model.quats.detach().clone()

    def restore_baseline(self) -> None:
        """Copy the baseline back into the live pipeline. No-op if no baseline
        has been taken yet — in that case the pipeline IS the baseline."""
        if self._baseline_means is None:
            return
        import torch
        with torch.no_grad():
            self._impl.pipeline.model.means.data.copy_(self._baseline_means)
            self._impl.pipeline.model.quats.data.copy_(self._baseline_quats)

    def apply_dynamic_edits(self, edits) -> int:
        """Apply a list of `SceneEdit`s to the live pipeline and return the
        number of Gaussians modified. Restores to baseline first so each call
        starts clean (and snapshots a baseline if none exists yet)."""
        if self._frame_graph is None:
            raise ValueError("apply_dynamic_edits requires a FrameGraph")
        self.snapshot_baseline()
        self.restore_baseline()
        return apply_edits_to_pipeline(self._impl.pipeline, edits, self._frame_graph)

    def render(
        self,
        camera_pose: Pose,
        intrinsics: dict,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Render a single camera at the given pose.

        Parameters
        ----------
        camera_pose
            Camera-to-world `Pose` in this renderer's ``world_frame``
            (default ``"ned"``). Frame mismatch raises immediately.
        intrinsics
            Pinhole intrinsics dict: ``{width, height, fx, fy, cx, cy}``.

        Returns
        -------
        (rgb_uint8, depth_float32_or_None) — image dims match the intrinsics.
        """
        assert_frame(camera_pose, self._world_frame)
        camera = self._get_or_build_camera(intrinsics)
        T_c2w = camera_pose.as_matrix()
        result = self._impl.render_rgb(camera, T_c2w)
        # Older FiGS returns just rgb; newer returns (rgb, depth).
        if isinstance(result, tuple) and len(result) == 2:
            rgb, depth = result
        else:
            rgb, depth = result, None
        return rgb, depth

    # ---- internals -----------------------------------------------------

    def _get_or_build_camera(self, intrinsics: dict) -> Any:
        key = tuple(sorted(intrinsics.items()))
        cam = self._cameras.get(key)
        if cam is None:
            cam = self._impl.generate_output_camera(intrinsics)
            self._cameras[key] = cam
        return cam
