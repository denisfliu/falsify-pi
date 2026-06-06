"""Per-camera postprocess pipeline shared by every consumer of rendered images.

A raw render from ``GSplatRenderer.render(pose, intrinsics)`` is a square-ish
RGB uint8 array at the camera's native resolution. Before it is handed to a
policy or written to a training-data parquet, every consumer applies the
same three transforms in the same order:

1. Resize to ``image_size`` (PIL bilinear). Skipped if already that size.
2. Channel swap RGB → BGR if the embodiment was authored in BGR (cv2 was
   the data-collection tool, so the on-disk PNG bytes are BGR-semantics
   even though PIL labels the PNG mode "RGB").
3. Composite a per-camera RGBA overlay onto the result. Used by the
   downward camera so the sim render shows the same drone-strut /
   gripper occlusion the real wrist-cam shipped to training did.

Before this module existed, those three lines were duplicated in
``falsify.training.exporter``, ``falsify.policy.pi_gateway``, and
``falsify.policy.vla``; parity was enforced by a `CLAUDE.md` comment. The
``CameraPostprocess`` dataclass centralizes them so the parity is by
construction.

The overlay PNG is stored in whatever channel order it was authored
against — typically the channel order of the dataset it was derived from.
For the carl wrist overlay (built from `gate_scenes_real_combined`, which
holds BGR pixels labeled RGB), the overlay file's channels are also
BGR-semantics. Composite happens **after** the channel swap, so target
and overlay end up in the same channel space and the colors match.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Overlay primitives
# ---------------------------------------------------------------------------


def load_overlay_rgba(path: str | Path) -> np.ndarray:
    """Load an RGBA PNG into a (H, W, 4) uint8 array.

    Raises a `ValueError` if the file is not 4-channel RGBA.
    """
    from PIL import Image as _Image
    img = _Image.open(path)
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError(
            f"load_overlay_rgba({str(path)!r}): expected (H, W, 4) RGBA, "
            f"got shape {arr.shape}"
        )
    return arr


def apply_overlay(img_uint8: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    """Standard straight-alpha compositing of ``overlay_rgba`` onto ``img_uint8``.

    Both must be uint8 with matching ``(H, W)``. The base image's channel
    semantics (RGB vs BGR) must match the overlay's by convention — see the
    module docstring. Returns a new uint8 array; does not mutate the inputs.
    """
    if img_uint8.dtype != np.uint8 or overlay_rgba.dtype != np.uint8:
        raise TypeError(
            f"apply_overlay: both inputs must be uint8; got "
            f"img={img_uint8.dtype}, overlay={overlay_rgba.dtype}"
        )
    if img_uint8.shape[:2] != overlay_rgba.shape[:2]:
        raise ValueError(
            f"apply_overlay: shape mismatch — img {img_uint8.shape[:2]} vs "
            f"overlay {overlay_rgba.shape[:2]}"
        )
    if img_uint8.shape[-1] != 3 or overlay_rgba.shape[-1] != 4:
        raise ValueError(
            f"apply_overlay: expected img (H, W, 3) and overlay (H, W, 4); "
            f"got img {img_uint8.shape} and overlay {overlay_rgba.shape}"
        )
    rgb = overlay_rgba[..., :3].astype(np.float32)
    a = overlay_rgba[..., 3:4].astype(np.float32) / 255.0
    base = img_uint8.astype(np.float32)
    out = (1.0 - a) * base + a * rgb
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# The chokepoint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraPostprocess:
    """One camera's render → consumer-ready pipeline.

    Constructed once per camera at consumer init time. ``apply(rgb_native)``
    is the only runtime hot-path call and is what both the training
    exporter and the live policy invoke.
    """
    image_size: Optional[int]
    channel_order: Literal["RGB", "BGR"]
    overlay_rgba: Optional[np.ndarray] = None     # (H, W, 4) uint8, sized to image_size

    def __post_init__(self) -> None:
        if self.channel_order not in ("RGB", "BGR"):
            raise ValueError(
                f"CameraPostprocess.channel_order must be 'RGB' or 'BGR'; "
                f"got {self.channel_order!r}"
            )
        if self.overlay_rgba is not None and self.image_size is not None:
            h, w = self.overlay_rgba.shape[:2]
            if h != self.image_size or w != self.image_size:
                raise ValueError(
                    f"CameraPostprocess: overlay shape {(h, w)} does not "
                    f"match image_size {self.image_size}"
                )

    @classmethod
    def from_paths(
        cls,
        *,
        image_size: Optional[int],
        channel_order: str = "RGB",
        overlay_path: Optional[str | Path] = None,
    ) -> "CameraPostprocess":
        """Convenience constructor that loads the overlay PNG once at init."""
        overlay = load_overlay_rgba(overlay_path) if overlay_path else None
        return cls(image_size=image_size, channel_order=channel_order, overlay_rgba=overlay)

    def apply(self, rgb_native: np.ndarray) -> np.ndarray:
        """Apply resize → channel swap → overlay composite. Returns uint8 H×W×3."""
        img = np.asarray(rgb_native, dtype=np.uint8)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(
                f"CameraPostprocess.apply: expected (H, W, 3) RGB uint8; got {img.shape}"
            )
        # 1. Resize.
        if self.image_size is not None and (
            img.shape[0] != self.image_size or img.shape[1] != self.image_size
        ):
            from PIL import Image as _Image
            img = np.asarray(
                _Image.fromarray(img).resize(
                    (self.image_size, self.image_size), _Image.BILINEAR,
                ),
                dtype=np.uint8,
            )
        # 2. Channel swap.
        if self.channel_order == "BGR":
            img = img[..., ::-1]
            # Ensure C-contiguous for downstream consumers that expect it.
            if not img.flags["C_CONTIGUOUS"]:
                img = np.ascontiguousarray(img)
        # 3. Overlay.
        if self.overlay_rgba is not None:
            img = apply_overlay(img, self.overlay_rgba)
        return img
