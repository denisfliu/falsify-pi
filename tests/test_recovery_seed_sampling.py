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


def test_compositional_post_gate_1_scoping():
    """With `gate_1_transit_time` set (compositional: drone cleared
    gate 1 but failed afterward), the sampler must scope its draw to
    post-gate-1 safe states regardless of failure_type — the bias
    inside the window is the existing failure-type bias."""
    # transit at t=5.0; post-gate-1 states are at steps 50..99
    h = _history(100, dt=0.1)
    rng = np.random.default_rng(0)
    # COLLISION_GATE on gate_2: bias-late inside the post-gate-1 window.
    picks = [
        sample_recovery_seed(h, FailureType.COLLISION_GATE, rng,
                             gate_1_transit_time=5.0)[0]
        for _ in range(400)
    ]
    assert all(p >= 50 for p in picks), \
        f"all picks must be post-gate-1; saw min={min(picks)}"
    # Beta(3,1) within 50..99 → mean ≈ 50 + 0.75*50 = 87.5
    assert 80 <= np.mean(picks) <= 95
    # MISS_GATE on gate_2: bias-early inside the same window.
    picks_early = [
        sample_recovery_seed(h, FailureType.MISS_GATE, rng,
                             gate_1_transit_time=5.0)[0]
        for _ in range(400)
    ]
    assert all(p >= 50 for p in picks_early)
    # Beta(1,3) within 50..99 → mean ≈ 50 + 0.25*50 = 62.5
    assert 56 <= np.mean(picks_early) <= 69


def test_goal_not_reached_scope_wins_over_gate_1_scope():
    """When both `transit_time` (GOAL_NOT_REACHED) and `gate_1_transit_time`
    are set — e.g. compositional GOAL_NOT_REACHED after both transits —
    the tighter post-transit (gate-2) scope should be used, not the
    looser post-gate-1 one."""
    # gate-1 transit at t=3.0 (step 30), gate-2 transit (= goal transit
    # for compositional) at t=6.0 (step 60). Post-gate-2 = steps 60..99.
    h = _history(100, dt=0.1)
    rng = np.random.default_rng(0)
    picks = [
        sample_recovery_seed(
            h, FailureType.GOAL_NOT_REACHED, rng,
            transit_time=6.0, gate_1_transit_time=3.0,
        )[0]
        for _ in range(400)
    ]
    assert all(p >= 60 for p in picks), \
        f"goal-transit scope must win; saw min={min(picks)}"


def test_pre_gate_1_failure_uses_full_history():
    """Compositional failure before gate-1 transit (e.g. stuck on
    approach) ⇒ no `gate_1_transit_time` is set ⇒ full history with
    failure-type bias."""
    h = _history(100, dt=0.1)
    rng = np.random.default_rng(0)
    picks = [
        sample_recovery_seed(h, FailureType.MISS_GATE, rng,
                             gate_1_transit_time=None)[0]
        for _ in range(400)
    ]
    # Beta(1,3) over full 0..99 → mean ≈ 25
    assert 15 <= np.mean(picks) <= 35


def test_between_gates_first_plane_cross_fallback_scopes_post_bypass():
    """When phase==between_gates and the drone clipped past gate_1
    (AABB latched, aperture not threaded), `gate_1_transit_time` is
    null but `first_plane_cross_t_1` is set. The orchestrator falls
    that value through as `gate_1_transit_time`; the sampler then
    scopes safe-history to t >= bypass — same behavior as a
    legitimate transit. Without this, the seed could land before the
    drone passed gate_1 → recovery would have to re-fly the approach."""
    h = _history(300, dt=0.1)
    rng = np.random.default_rng(0)
    # Simulate the orchestrator's fallback: when gate_1_transit_time
    # is None but the failure is between_gates, the orchestrator
    # passes first_plane_cross_t_1 as gate_1_transit_time.
    bypass_t = 4.7
    picks = [
        sample_recovery_seed(
            h, FailureType.MISS_GATE, rng,
            gate_1_transit_time=bypass_t,
        )[0]
        for _ in range(400)
    ]
    # All picks must be from steps >= 47 (t >= 4.7s @ 0.1s/step).
    assert all(p >= 47 for p in picks), \
        f"all picks must be post-bypass; saw min={min(picks)}"


def test_pre_gate_bypass_time_scopes_sampling_to_pre_bypass_states():
    """When the drone failed pre-gate-1 but already crossed gate-1's
    plane (clipped outside the aperture), the sampler must scope its
    draw to safe states from BEFORE the bypass. Otherwise the seed
    lands past the gate and the recovery has to reverse and re-cross
    (the 'doubles back' pathology)."""
    h = _history(100, dt=0.1)        # t in [0.0, 9.9]
    rng = np.random.default_rng(0)
    picks = [
        sample_recovery_seed(
            h, FailureType.MISS_GATE, rng,
            pre_gate_bypass_time=5.0,
        )[0]
        for _ in range(400)
    ]
    # Every pick must be from before bypass time (steps 0..49).
    assert all(p < 50 for p in picks), \
        f"all picks must be pre-bypass; saw max={max(picks)}"
    # Beta(1,3) over 0..49 → mean ≈ 12.5
    assert 5 <= np.mean(picks) <= 20


def test_pre_gate_bypass_scope_ignored_when_gate_1_transit_time_present():
    """If `gate_1_transit_time` is set (drone DID transit gate-1
    legitimately, then failed post-transit), the pre-gate bypass scope
    must not be used — the failure is post-gate-1, not pre-gate-1."""
    h = _history(100, dt=0.1)
    rng = np.random.default_rng(0)
    picks = [
        sample_recovery_seed(
            h, FailureType.COLLISION_GATE, rng,
            gate_1_transit_time=5.0,
            # Should be ignored because gate_1_transit_time wins.
            pre_gate_bypass_time=2.0,
        )[0]
        for _ in range(400)
    ]
    # Post-gate-1 scope (steps 50..99) — NOT the pre-bypass scope.
    assert all(p >= 50 for p in picks), \
        f"gate_1_transit_time scope must win; saw min={min(picks)}"


def test_pre_gate_bypass_empty_window_falls_back_to_full_history():
    """If no safe state is recorded BEFORE the bypass time (e.g. the
    bypass fired at step 0 with no prior safe state), the sampler must
    fall back to the full history rather than crashing on an empty
    draw."""
    h = _history(10, dt=0.1)
    rng = np.random.default_rng(0)
    step, _ = sample_recovery_seed(
        h, FailureType.MISS_GATE, rng,
        pre_gate_bypass_time=0.0,    # nothing earlier than t=0.0
    )
    assert 0 <= step < 10


def test_collision_gate_bias_late_regardless_of_phase():
    """COLLISION_GATE always uses bias-late (Beta(3,1)). ``failure_phase``
    is accepted by the sampler for future phase-aware overrides but is
    currently not consulted for collisions; the default ``_BIAS_LATE``
    rule wins. This is a regression guard — we briefly experimented
    with a pre_gate_1 early-bias override and it didn't help."""
    from falsify.recovery import bias_for
    h = _history(100, dt=0.1)
    rng = np.random.default_rng(0)
    for phase in ("pre_gate_1", "between_gates", "post_gate_2", None):
        picks = [
            sample_recovery_seed(
                h, FailureType.COLLISION_GATE, rng,
                failure_phase=phase,
            )[0]
            for _ in range(400)
        ]
        # Beta(3,1) over 0..99 → mean ≈ 75
        assert 65 <= np.mean(picks) <= 85, \
            f"phase={phase!r}: expected late-bias mean, saw {np.mean(picks):.1f}"
        assert bias_for(FailureType.COLLISION_GATE,
                        failure_phase=phase) == "late"
