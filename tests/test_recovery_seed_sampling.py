"""Tests for `sample_recovery_seed` — failure-type bias + post-transit scope."""

from __future__ import annotations

import numpy as np
import pytest

from falsify.geometry import Frame, Point
from falsify.recovery import sample_recovery_seed
from falsify.safety.records import FailureType
from falsify.sim.dynamics_state import DroneState


NED = Frame("ned")


def _state(t: float) -> DroneState:
    return DroneState(
        pos=Point(np.array([0.0, 0.0, 1.5]), frame=NED),
        vel=np.zeros(3),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        t=float(t),
    )


def _history(n: int, dt: float = 0.1):
    return [(i, _state(i * dt)) for i in range(n)]


def test_empty_history_raises():
    with pytest.raises(ValueError, match="empty"):
        sample_recovery_seed([], FailureType.MISS_GATE, np.random.default_rng(0))


def test_singleton_history_returns_only_state():
    h = _history(1)
    step, st = sample_recovery_seed(h, FailureType.COLLISION_GATE, np.random.default_rng(0))
    assert step == 0




def test_bias_early_concentrates_low_indices():
    h = _history(100)
    rng = np.random.default_rng(0)
    picks = [sample_recovery_seed(h, FailureType.MISS_GATE, rng)[0] for _ in range(400)]
    # Beta(1,3) mean = 0.25 → expect mean idx around 25.
    assert 15 <= np.mean(picks) <= 35


def test_bias_late_concentrates_high_indices():
    h = _history(100)
    rng = np.random.default_rng(0)
    picks = [sample_recovery_seed(h, FailureType.COLLISION_GATE, rng)[0] for _ in range(400)]
    # Beta(3,1) mean = 0.75 → expect mean idx around 75.
    assert 65 <= np.mean(picks) <= 85


def test_goal_not_reached_scopes_to_post_transit():
    """With transit_time provided, GOAL_NOT_REACHED draws only from
    safe states recorded after transit, with bias-early *within* that subset
    so the seed sits close to the gate crossing."""
    h = _history(100, dt=0.1)  # t ranges 0.0 → 9.9
    transit_time = 5.0  # transit at step 50; post-transit steps are 50..99
    rng = np.random.default_rng(0)
    picks = [
        sample_recovery_seed(h, FailureType.GOAL_NOT_REACHED, rng,
                             transit_time=transit_time)[0]
        for _ in range(400)
    ]
    # Every pick must be post-transit.
    assert min(picks) >= 50
    # Within [50, 100), bias-early ⇒ mean ≈ 50 + 0.25 * 50 = 62.5.
    assert 55 <= np.mean(picks) <= 70


def test_goal_not_reached_without_transit_time_falls_back_to_full_history():
    h = _history(100)
    rng = np.random.default_rng(0)
    picks = [
        sample_recovery_seed(h, FailureType.GOAL_NOT_REACHED, rng)[0]
        for _ in range(400)
    ]
    # No transit time → full history, bias-early ⇒ mean ≈ 25.
    assert 15 <= np.mean(picks) <= 35


def test_goal_not_reached_no_post_transit_safe_falls_back():
    """If detector recorded zero safe states with t >= transit_time, sampler
    must not hand back an empty draw — falls back to the full history."""
    h = _history(10, dt=0.1)  # t in [0.0, 0.9]
    rng = np.random.default_rng(0)
    step, _ = sample_recovery_seed(
        h, FailureType.GOAL_NOT_REACHED, rng, transit_time=5.0,
    )
    # Whole history is pre-transit → fallback to the full list.
    assert 0 <= step < 10
