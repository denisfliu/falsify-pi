"""Pin behavioural equivalence between ``falsify.eval.sampling`` and the
private helpers it replaced in ``scripts/generate_eval_bundles.py``.

The extracted module is reused by ``scripts/collect_recovery_trajectories.py``,
so the contract must stay byte-identical with what the bundle generator
emits — otherwise bundles and on-the-fly collected cards would draw from
different RNG streams.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def script_helpers():
    """Load the bundle-generator script as a module so we can call its
    re-exported aliases (`_seed_for`, `_sample_*`) and confirm they're
    the same functions the extracted module exposes."""
    name = "_gen_bundles_mod"
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts" / "eval" / "generate_eval_bundles.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_seed_for_is_re_exported(script_helpers):
    """The script's ``_seed_for`` must be the same object the module
    exports — not a copy — so behaviour can't drift."""
    from falsify.eval.sampling import seed_for
    assert script_helpers._seed_for is seed_for


def test_sample_helpers_re_exported(script_helpers):
    from falsify.eval.sampling import (
        sample_gate_perturbation, sample_start_mocap,
    )
    assert script_helpers._sample_gate_perturbation is sample_gate_perturbation
    assert script_helpers._sample_start_mocap is sample_start_mocap


def test_seed_for_deterministic_within_process():
    """Same inputs → same seed, repeated calls. (Process-to-process
    determinism is broken by PYTHONHASHSEED — that's a pre-existing
    bundle-repro issue, not in scope here.)"""
    from falsify.eval.sampling import seed_for
    a = seed_for(0, "pure", "left_gate", 0)
    b = seed_for(0, "pure", "left_gate", 0)
    assert a == b
    assert 0 <= a < 2**32


def test_sample_gate_perturbation_disabled_returns_none():
    from falsify.eval.sampling import sample_gate_perturbation
    rng = np.random.default_rng(0)
    assert sample_gate_perturbation({"gate_perturbation": {"enabled": False}}, rng) is None
    assert sample_gate_perturbation({}, rng) is None


def test_sample_gate_perturbation_respects_bounds_and_pins_z():
    """Z-component must always be 0 (gates don't levitate) and Δxyz/Δyaw
    must stay within the declared half-widths."""
    from falsify.eval.sampling import sample_gate_perturbation
    recipe = {"gate_perturbation": {
        "enabled": True,
        "offset_half_widths": [0.03, 0.03, 0.0],
        "yaw_half_width_rad": 0.05236,
    }}
    rng = np.random.default_rng(42)
    for _ in range(50):
        p = sample_gate_perturbation(recipe, rng)
        dx, dy, dz = p["delta_xyz"]
        assert abs(dx) <= 0.03 + 1e-12
        assert abs(dy) <= 0.03 + 1e-12
        assert dz == 0.0
        assert abs(p["delta_yaw_rad"]) <= 0.05236 + 1e-12


def test_sample_start_mocap_disabled_returns_nominal():
    from falsify.eval.sampling import sample_start_mocap
    scene_cfg = {
        "start_position_mocap": [1.0, 2.0, 3.0],
        "start_randomization": {"half_widths_mocap": [0.05, 0.05, 0.05]},
    }
    rng = np.random.default_rng(0)
    assert sample_start_mocap(scene_cfg, rng, enabled=False) == [1.0, 2.0, 3.0]


def test_sample_start_mocap_uses_scene_half_widths():
    from falsify.eval.sampling import sample_start_mocap
    scene_cfg = {
        "start_position_mocap": [1.0, 2.0, 3.0],
        "start_randomization": {"half_widths_mocap": [0.05, 0.05, 0.05]},
    }
    rng = np.random.default_rng(42)
    for _ in range(20):
        out = sample_start_mocap(scene_cfg, rng, enabled=True)
        assert abs(out[0] - 1.0) <= 0.05 + 1e-12
        assert abs(out[1] - 2.0) <= 0.05 + 1e-12
        assert abs(out[2] - 3.0) <= 0.05 + 1e-12
