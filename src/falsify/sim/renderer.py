"""GSplat renderer wrapper — frame-aware boundary around `figs.render.gsplat.GSPlat`.

FiGS' renderer expects a 4×4 camera-to-world matrix in **FiGS NED** and
internally applies its own ``Tw2g`` (NED → gsplat-internal). On the falsify
side we receive a ``Pose`` whose ``.frame`` must be ``"ned"`` — the only
boundary conversion happens here, and the rule is enforced.

FiGS / nerfstudio / CUDA are imported lazily so this module stays loadable
on machines without the full stack.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from falsify.geometry import Pose, assert_frame


class GSplatRenderer:
    """Render a single camera against a Nerfstudio Gaussian-splat workspace.

    Construction lazy-imports FiGS / nerfstudio so the rest of falsify can be
    imported on machines without CUDA.
    """

    def __init__(self, gsplat_path: str | Path, world_frame: str = "ned") -> None:
        gsplat_path = Path(gsplat_path)
        # Lazy import — FiGS pulls in nerfstudio + CUDA bindings.
        from figs.render.gsplat import GSplat as _FiGSGSplat  # type: ignore
        self._impl = _FiGSGSplat(gsplat_path)
        self._world_frame = world_frame
        self._cameras: dict[tuple, Any] = {}   # cache nerfstudio Camera objects by intrinsics

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
