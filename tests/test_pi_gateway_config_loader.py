"""PiGatewayConfig.from_yaml / from_dict — single source of truth for
loading the shipped policy YAMLs."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from falsify.policy import PiGatewayConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_YAMLS = sorted((REPO_ROOT / "configs" / "policies" / "pi_gateway").glob("*.yaml"))


@pytest.mark.parametrize("path", POLICY_YAMLS, ids=lambda p: p.name)
def test_every_shipped_yaml_loads(path: Path):
    cfg = PiGatewayConfig.from_yaml(path)
    assert cfg.gateway_url, f"{path.name}: missing gateway_url"
    assert cfg.hz > 0
    assert cfg.state_dim > 0
    assert cfg.action_dim > 0
    assert "downward" in cfg.gripper_overlay_paths


def test_prompt_override_wins():
    p = POLICY_YAMLS[0]
    cfg = PiGatewayConfig.from_yaml(p, prompt_override="my new prompt")
    assert cfg.prompt == "my new prompt"


def test_use_rtc_override():
    p = POLICY_YAMLS[0]
    cfg_off = PiGatewayConfig.from_yaml(p, use_rtc_override=False)
    cfg_on = PiGatewayConfig.from_yaml(p, use_rtc_override=True)
    assert cfg_off.use_rtc is False
    assert cfg_on.use_rtc is True


def test_execute_chunk_size_override():
    p = POLICY_YAMLS[0]
    cfg = PiGatewayConfig.from_yaml(p, execute_chunk_size_override=99)
    assert cfg.execute_chunk_size == 99


def test_from_dict_rejects_wrong_type():
    cfg = {"type": "openpi", "gateway_url": "x"}
    with pytest.raises(ValueError, match="expected type=pi_gateway"):
        PiGatewayConfig.from_dict(cfg)


def test_from_dict_accepts_missing_type():
    cfg = {"gateway_url": "x"}
    out = PiGatewayConfig.from_dict(cfg)
    assert out.gateway_url == "x"


def test_record_dir_from_yaml_field():
    cfg = {"gateway_url": "x", "record_dir": "/tmp/foo"}
    out = PiGatewayConfig.from_dict(cfg)
    assert out.record_dir == Path("/tmp/foo")
