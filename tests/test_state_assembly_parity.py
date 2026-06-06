"""Train/eval state-vector parity: every shipped policy YAML must build the
same state vector the embodiment-driven exporter would build for the same pose."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from falsify.policy import PiGatewayConfig
from falsify.policy.state_assembly import build_state_vector
from falsify.training import load_embodiment


REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_YAMLS = sorted((REPO_ROOT / "configs" / "policies" / "pi_gateway").glob("*.yaml"))
EMBODIMENT_YAML = REPO_ROOT / "configs" / "embodiments" / "carl_dual_mocap.yaml"


@pytest.mark.parametrize("policy_path", POLICY_YAMLS, ids=lambda p: p.name)
def test_pi_gateway_yaml_wires_embodiment(policy_path: Path):
    cfg = PiGatewayConfig.from_yaml(policy_path)
    assert cfg.embodiment_path, f"{policy_path.name}: missing embodiment_path"
    assert (REPO_ROOT / cfg.embodiment_path).exists(), (
        f"{policy_path.name}: embodiment_path resolves to a missing file: "
        f"{cfg.embodiment_path}"
    )


def test_build_state_v7_layout():
    """The carl_dual_mocap embodiment must produce the v7 layout the policy
    used to hardcode: `[px_mocap, py_mocap, pz_mocap, -yaw_ned, 0, 0, 0]`."""
    emb = load_embodiment(EMBODIMENT_YAML)
    pos_mocap = np.array([1.0, 2.0, 3.0])
    yaw_ned = 0.4
    yaw_mocap = -yaw_ned  # perm5 sign flip
    pos_ned = np.array([1.0, -2.0, -3.0])
    state = build_state_vector(
        emb, pos_mocap=pos_mocap, yaw_mocap=yaw_mocap,
        pos_ned=pos_ned, yaw_ned=yaw_ned,
    )
    assert state.shape == (7,)
    np.testing.assert_array_almost_equal(state[0:3], pos_mocap)
    assert state[3] == pytest.approx(yaw_mocap)
    np.testing.assert_array_equal(state[4:], np.zeros(3, dtype=np.float32))


def test_build_state_matches_hardcoded_v7():
    """Cross-check the embodiment-driven assembly against the v7 hardcoded path."""
    emb = load_embodiment(EMBODIMENT_YAML)
    rng = np.random.default_rng(7)
    for _ in range(10):
        pos_mocap = rng.normal(size=3)
        yaw_ned = rng.uniform(-np.pi, np.pi)
        pos_ned = rng.normal(size=3)
        yaw_mocap = -yaw_ned
        # Hardcoded v7 layout the policy used to produce inline
        expected = np.zeros(7, dtype=np.float32)
        expected[0:3] = pos_mocap
        expected[3] = yaw_mocap
        # Schema-driven version
        actual = build_state_vector(
            emb, pos_mocap=pos_mocap, yaw_mocap=yaw_mocap,
            pos_ned=pos_ned, yaw_ned=yaw_ned,
        )
        np.testing.assert_array_almost_equal(actual, expected)


def test_build_state_unknown_field_raises():
    class _FakeEmb:
        name = "test"
        state = [type("F", (), {"name": "unknown_field"})()]
        def state_dim(self): return 1
    with pytest.raises(ValueError, match="no getter registered"):
        build_state_vector(_FakeEmb(), pos_mocap=np.zeros(3), yaw_mocap=0.0)


def test_build_state_missing_ned_raises():
    class _FakeEmb:
        name = "ned-test"
        state = [type("F", (), {"name": "x_ned"})()]
        def state_dim(self): return 1
    with pytest.raises(ValueError, match="pos_ned"):
        build_state_vector(_FakeEmb(), pos_mocap=np.zeros(3), yaw_mocap=0.0)
