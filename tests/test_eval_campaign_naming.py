"""Unit tests for the per-policy run-number derivation in
``scripts/run_eval_campaign.py``.

The naming convention (``runs/eval_campaigns/<policy_id>/run-NNN-...``)
is load-bearing: bumping NNN per policy is what keeps the on-disk
campaign tree scannable. These tests pin the bookkeeping logic so any
refactor of the directory layout has to keep the contract.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_runner_module():
    """Side-step ``import scripts.run_eval_campaign`` machinery — the
    script lives under ``scripts/`` which isn't a package. ``spec_from
    _file_location`` lets the test pull the helpers without standing up
    a full package layout."""
    name = "_run_eval_campaign_mod"
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts" / "eval" / "run_eval_campaign.py",
    )
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so the @dataclass machinery can resolve
    # cls.__module__ via sys.modules.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def runner():
    return _load_runner_module()


def test_next_run_number_when_root_missing(runner, tmp_path):
    assert runner._next_run_number(tmp_path / "does-not-exist") == 1


def test_next_run_number_empty_root(runner, tmp_path):
    (tmp_path / "policy_x").mkdir()
    assert runner._next_run_number(tmp_path / "policy_x") == 1


def test_next_run_number_single_run(runner, tmp_path):
    root = tmp_path / "policy_x"
    (root / "run-001-pure-20260101_000000").mkdir(parents=True)
    assert runner._next_run_number(root) == 2


def test_next_run_number_non_monotonic_uses_max(runner, tmp_path):
    """If run-002 is missing we still start from max+1 = 4, not 3.
    Otherwise we'd reuse a number that the user has already filed away."""
    root = tmp_path / "policy_x"
    for name in ("run-001-pure-x", "run-003-gate_perturbed_small-x"):
        (root / name).mkdir(parents=True)
    assert runner._next_run_number(root) == 4


def test_next_run_number_ignores_non_run_entries(runner, tmp_path):
    root = tmp_path / "policy_x"
    root.mkdir()
    (root / "README.md").write_text("just a doc")
    (root / "_legacy").mkdir()                 # not run-NNN-*
    (root / "run-002-pure-x").mkdir()
    assert runner._next_run_number(root) == 3


def test_next_run_number_ignores_run_prefix_without_number(runner, tmp_path):
    root = tmp_path / "policy_x"
    root.mkdir()
    (root / "run-foo").mkdir()                  # no integer after `run-`
    (root / "running-001-bad").mkdir()          # doesn't start with `run-NNN-`
    assert runner._next_run_number(root) == 1


def test_derive_out_dir_puts_adhoc_under_policy(runner, tmp_path, monkeypatch):
    """Default ``--out`` lands under ``<policy>/adhoc/`` so the policy
    root stays clean for ``sweep-NNN-<ts>/`` folders written by
    ``tools/run_eval_sweep.sh``."""
    fake_root = tmp_path / "fake_repo"
    monkeypatch.setattr(runner, "REPO_ROOT", fake_root)
    policy = fake_root / "configs" / "policies" / "pi_gateway" / "my_policy.yaml"
    policy.parent.mkdir(parents=True)
    policy.write_text("type: pi_gateway\n")

    out = runner._derive_out_dir(policy, scenario_name="gate_perturbed_small")
    assert out.parent == fake_root / "runs" / "eval_campaigns" / "my_policy" / "adhoc"
    assert out.name.startswith("run-001-gate_perturbed_small-")


def test_derive_out_dir_increments_within_adhoc(runner, tmp_path, monkeypatch):
    """Calling ``_derive_out_dir`` twice should produce run-001 then
    run-002 — the auto-increment is scoped to the adhoc/ subfolder, NOT
    to sibling sweep-*/run-* dirs that may also live under the policy."""
    fake_root = tmp_path / "fake_repo"
    monkeypatch.setattr(runner, "REPO_ROOT", fake_root)
    policy = fake_root / "configs" / "policies" / "pi_gateway" / "my_policy.yaml"
    policy.parent.mkdir(parents=True)
    policy.write_text("type: pi_gateway\n")

    # Pre-existing sweep-001 with its own run-001..004 — must NOT bump
    # the adhoc counter.
    sweep = (fake_root / "runs" / "eval_campaigns" / "my_policy"
             / "sweep-001-20260520_214932")
    for sc in ("pure", "gate_perturbed_small"):
        (sweep / f"run-001-{sc}-20260520_214932").mkdir(parents=True)

    out1 = runner._derive_out_dir(policy, scenario_name="pure")
    out1.mkdir(parents=True)
    out2 = runner._derive_out_dir(policy, scenario_name="pure")
    assert out1.name.startswith("run-001-pure-")
    assert out2.name.startswith("run-002-pure-")
    assert out1.parent.name == "adhoc"
