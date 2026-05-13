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
    ) -> None:
        gsplat_path = Path(gsplat_path)
        data_cwd = Path(data_cwd) if data_cwd is not None else None
        # Lazy import — FiGS pulls in nerfstudio + CUDA bindings.
        from figs.render.gsplat import GSplat as _FiGSGSplat  # type: ignore
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
