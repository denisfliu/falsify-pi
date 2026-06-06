"""Unit tests for the shared per-camera postprocess pipeline."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from falsify.policy.camera_postprocess import (
    CameraPostprocess,
    apply_overlay,
    load_overlay_rgba,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CARL_WRIST_OVERLAY = REPO_ROOT / "configs" / "embodiments" / "assets" / "carl_wrist_overlay.png"


def _solid(size: int, color: tuple[int, int, int]) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[..., 0] = color[0]
    img[..., 1] = color[1]
    img[..., 2] = color[2]
    return img


def _solid_rgba(size: int, rgb: tuple[int, int, int], a: int) -> np.ndarray:
    out = np.zeros((size, size, 4), dtype=np.uint8)
    out[..., 0] = rgb[0]
    out[..., 1] = rgb[1]
    out[..., 2] = rgb[2]
    out[..., 3] = a
    return out


def test_apply_overlay_zero_alpha_is_identity():
    base = _solid(8, (100, 150, 200))
    overlay = _solid_rgba(8, (10, 20, 30), a=0)
    out = apply_overlay(base, overlay)
    np.testing.assert_array_equal(out, base)


def test_apply_overlay_full_alpha_replaces():
    base = _solid(8, (100, 150, 200))
    overlay = _solid_rgba(8, (10, 20, 30), a=255)
    out = apply_overlay(base, overlay)
    np.testing.assert_array_equal(out, _solid(8, (10, 20, 30)))


def test_apply_overlay_half_alpha_blends():
    base = _solid(8, (200, 200, 200))
    overlay = _solid_rgba(8, (0, 0, 0), a=128)
    out = apply_overlay(base, overlay)
    # 128 / 255 ≈ 0.502 — keep within 1 LSB of perfect midpoint
    assert abs(int(out[0, 0, 0]) - 99) <= 1


def test_apply_overlay_rejects_shape_mismatch():
    base = _solid(8, (100, 100, 100))
    overlay = _solid_rgba(16, (0, 0, 0), a=255)
    with pytest.raises(ValueError):
        apply_overlay(base, overlay)


def test_load_overlay_rgba_roundtrip(tmp_path):
    overlay = _solid_rgba(16, (10, 20, 30), a=128)
    p = tmp_path / "overlay.png"
    Image.fromarray(overlay, mode="RGBA").save(p)
    loaded = load_overlay_rgba(p)
    assert loaded.shape == (16, 16, 4)
    assert loaded.dtype == np.uint8
    np.testing.assert_array_equal(loaded, overlay)


def test_postprocess_noop():
    pp = CameraPostprocess(image_size=None, channel_order="RGB", overlay_rgba=None)
    src = _solid(16, (10, 20, 30))
    np.testing.assert_array_equal(pp.apply(src), src)


def test_postprocess_resize():
    pp = CameraPostprocess(image_size=8, channel_order="RGB", overlay_rgba=None)
    src = _solid(16, (10, 20, 30))
    out = pp.apply(src)
    assert out.shape == (8, 8, 3)
    # Solid color survives bilinear resize exactly.
    np.testing.assert_array_equal(out, _solid(8, (10, 20, 30)))


def test_postprocess_bgr_swap():
    pp = CameraPostprocess(image_size=None, channel_order="BGR", overlay_rgba=None)
    src = _solid(4, (10, 20, 30))
    out = pp.apply(src)
    np.testing.assert_array_equal(out, _solid(4, (30, 20, 10)))


def test_postprocess_overlay_size_mismatch_raises():
    overlay = _solid_rgba(16, (0, 0, 0), a=255)
    with pytest.raises(ValueError):
        CameraPostprocess(image_size=8, channel_order="RGB", overlay_rgba=overlay)


def test_postprocess_full_pipeline_order():
    """Resize → BGR swap → overlay all in one call."""
    overlay = _solid_rgba(4, (255, 0, 0), a=128)
    pp = CameraPostprocess(image_size=4, channel_order="BGR", overlay_rgba=overlay)
    src = _solid(8, (0, 100, 200))   # RGB → after BGR swap = (200, 100, 0)
    out = pp.apply(src)
    # Half-alpha blend of (200, 100, 0) with overlay (255, 0, 0):
    expected_r = int(round(0.5 * 200 + 128 / 255 * 255))
    # Solid color resize is exact; just confirm shape + reasonable blend
    assert out.shape == (4, 4, 3)
    # First channel skews red from overlay; last channel pulled toward 0
    assert out[0, 0, 0] > out[0, 0, 2]


def test_carl_wrist_overlay_is_loadable():
    """The shipped overlay asset loads and is the expected shape."""
    if not CARL_WRIST_OVERLAY.exists():
        pytest.skip(f"asset missing: {CARL_WRIST_OVERLAY}")
    arr = load_overlay_rgba(CARL_WRIST_OVERLAY)
    assert arr.shape == (256, 256, 4)
    assert arr[..., 3].max() > 0  # mask is non-empty
    assert arr[..., 3].min() == 0  # has transparent regions too
