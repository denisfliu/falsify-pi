"""Train/eval preprocess parity — exporter and PiGatewayPolicy must produce
byte-identical wrist-camera bytes from the same native render.

Two embodiment eras (see root CLAUDE.md § Image convention):

- ``carl_dual_mocap_legacy_bgr.yaml`` — frozen BGR + raw-fisheye era. The
  twelve shipped v7/v9 pi_gateway YAMLs were trained against it, so the
  per-policy parity tests pair against this file.
- ``carl_dual_mocap.yaml`` — canonical RGB + pinhole convention
  (2026-06-12). New checkpoints/policy YAMLs pair against this one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from falsify.policy.camera_postprocess import CameraPostprocess
from falsify.training.embodiment import load_embodiment


REPO_ROOT = Path(__file__).resolve().parents[1]
EMBODIMENT = REPO_ROOT / "configs" / "embodiments" / "carl_dual_mocap.yaml"
LEGACY_EMBODIMENT = (
    REPO_ROOT / "configs" / "embodiments" / "carl_dual_mocap_legacy_bgr.yaml"
)
POLICY_YAMLS = sorted((REPO_ROOT / "configs" / "policies" / "pi_gateway").glob("*.yaml"))
LEGACY_OVERLAY = REPO_ROOT / "configs" / "embodiments" / "assets" / "carl_wrist_overlay.png"
RGB_OVERLAY = (
    REPO_ROOT / "configs" / "embodiments" / "assets" / "carl_wrist_overlay_pinhole_rgb.png"
)


def _wrist_postprocess_from_embodiment(embodiment_path: Path) -> CameraPostprocess:
    emb = load_embodiment(embodiment_path)
    for cam in emb.cameras:
        if cam.column == "wrist_image":
            assert cam.gripper_overlay_path, "embodiment YAML must declare wrist overlay path"
            return CameraPostprocess.from_paths(
                image_size=cam.image_size,
                channel_order=cam.channel_order,
                overlay_path=cam.gripper_overlay_path,
            )
    pytest.fail(f"no wrist_image entry in {embodiment_path.name}")


def _wrist_postprocess_from_policy_yaml(path: Path) -> CameraPostprocess:
    pcfg = yaml.safe_load(path.read_text())
    overlay = (pcfg.get("gripper_overlay_paths") or {}).get("downward")
    return CameraPostprocess.from_paths(
        image_size=pcfg.get("image_size"),
        channel_order=str(pcfg.get("channel_order", "RGB")),
        overlay_path=overlay,
    )


def _policy_embodiment(policy_path: Path) -> Path:
    """Which embodiment era a policy YAML belongs to (by channel_order)."""
    pcfg = yaml.safe_load(policy_path.read_text())
    order = str(pcfg.get("channel_order", "RGB"))
    return LEGACY_EMBODIMENT if order == "BGR" else EMBODIMENT


def test_legacy_embodiment_has_wrist_overlay():
    assert LEGACY_OVERLAY.exists(), f"overlay asset missing at {LEGACY_OVERLAY}"
    pp = _wrist_postprocess_from_embodiment(LEGACY_EMBODIMENT)
    assert pp.channel_order == "BGR"
    assert pp.overlay_rgba is not None
    assert pp.overlay_rgba.shape == (256, 256, 4)


def test_canonical_embodiment_is_rgb_pinhole_era():
    emb = load_embodiment(EMBODIMENT)
    for cam in emb.cameras:
        assert cam.channel_order == "RGB", (
            f"{cam.column}: canonical embodiment must be RGB (convention "
            f"2026-06-12); BGR lives only in carl_dual_mocap_legacy_bgr.yaml"
        )
    wrist = [c for c in emb.cameras if c.column == "wrist_image"][0]
    assert wrist.gripper_overlay_path == str(
        RGB_OVERLAY.relative_to(REPO_ROOT)
    ), "canonical embodiment must use the RGB/pinhole overlay asset"


@pytest.mark.skipif(
    not RGB_OVERLAY.exists(),
    reason="RGB overlay not generated yet (scripts/dataset/build_wrist_overlay.py)",
)
def test_canonical_embodiment_overlay_loadable():
    pp = _wrist_postprocess_from_embodiment(EMBODIMENT)
    assert pp.channel_order == "RGB"
    assert pp.overlay_rgba is not None
    assert pp.overlay_rgba.shape == (256, 256, 4)


@pytest.mark.parametrize("policy_path", POLICY_YAMLS, ids=lambda p: p.name)
def test_pi_gateway_yaml_declares_overlay(policy_path: Path):
    """Every pi_gateway YAML must declare a wrist overlay path."""
    pcfg = yaml.safe_load(policy_path.read_text())
    overlay_map = pcfg.get("gripper_overlay_paths") or {}
    assert "downward" in overlay_map, (
        f"{policy_path.name}: missing gripper_overlay_paths.downward"
    )


@pytest.mark.parametrize("policy_path", POLICY_YAMLS, ids=lambda p: p.name)
def test_train_eval_parity_on_random_render(policy_path: Path):
    """Train (exporter) and eval (PiGatewayPolicy) postprocess must match
    byte-for-byte against the embodiment era the policy belongs to."""
    embodiment = _policy_embodiment(policy_path)
    if embodiment == EMBODIMENT and not RGB_OVERLAY.exists():
        pytest.skip("RGB overlay not generated yet")
    rng = np.random.default_rng(0)
    native = rng.integers(0, 256, size=(320, 320, 3), dtype=np.uint8)

    train_pp = _wrist_postprocess_from_embodiment(embodiment)
    eval_pp = _wrist_postprocess_from_policy_yaml(policy_path)

    train_out = train_pp.apply(native)
    eval_out = eval_pp.apply(native)
    np.testing.assert_array_equal(
        train_out, eval_out,
        err_msg=(
            f"preprocess parity broken: {embodiment.name} vs {policy_path.name}"
        ),
    )


def test_overlay_actually_modifies_pixels():
    """Sanity: applying the wrist postprocess to a uniform render must change pixels
    where the overlay's alpha is non-zero. Catches a silently-no-op overlay path."""
    pp = _wrist_postprocess_from_embodiment(LEGACY_EMBODIMENT)
    src = np.full((320, 320, 3), 200, dtype=np.uint8)   # uniform gray
    out = pp.apply(src)
    # Resize 320→256 of a uniform image is still uniform; BGR swap on uniform
    # is identity; the only thing that can change pixels is the overlay.
    assert (out != 200).any(), "overlay had no effect — pipeline is silently a no-op"
