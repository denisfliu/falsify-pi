"""Frame-awareness audit.

This test enforces the invariant that *every public position-like value* in
falsify carries its frame. New code that leaks a bare ``np.ndarray`` for a
position will trip one of these checks.

Scope of the audit: code under ``src/falsify/``. The external submodules
(``external/FiGS``, ``external/splatnav``, ``external/Splat-MOVER``) are
third-party and intentionally outside this rule. Where falsify wraps a
submodule (e.g. the gsplat renderer, the splatnav planner), the wrapper
itself must convert the boundary to/from frame-tagged types — that contract
is enforced here as the wrappers land.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import numpy as np
import pytest

from falsify.geometry import (
    NED, MOCAP, COLMAP, NS, CAM_BODY, CAM_FORWARD, CAM_DOWNWARD,
    Point, Pose, Trajectory, PointCloud,
    SE3, Sim3, FrameGraph,
)
from falsify.policy import (
    MockStraightLine, MockStraightLineConfig,
    MockNoisy, MockNoisyConfig,
    Observation,
)
from falsify.sensors import (
    SensorRig, StateSensor, PromptSensor, CameraSensor, CameraSpec,
)
from falsify.sim.dynamics_state import DroneState


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src" / "falsify"


# ---------------------------------------------------------------------------
# 1. Every frame-tagged type carries .frame.
# ---------------------------------------------------------------------------


def test_all_geometry_types_carry_a_frame_attribute():
    for cls in (Point, Pose, Trajectory, PointCloud):
        sig = inspect.signature(cls.__init__)
        assert "frame" in sig.parameters, f"{cls.__name__} must accept frame= at construction"


def test_drone_state_exposes_frame_via_position():
    state = DroneState(
        pos=Point.of(0, 0, 0, NED),
        vel=np.zeros(3),
        quat_xyzw=np.array([0, 0, 0, 1.0]),
        t=0.0,
    )
    assert state.frame is NED
    assert state.pos.frame is NED


# ---------------------------------------------------------------------------
# 2. Mock policies emit frame-tagged trajectories.
# ---------------------------------------------------------------------------


def _build_obs(pos_frame):
    state = DroneState(
        pos=Point.of(0, 0, 0, pos_frame),
        vel=np.zeros(3),
        quat_xyzw=np.array([0, 0, 0, 1.0]),
        t=0.0,
    )
    return SensorRig([StateSensor()]).build(state)


@pytest.mark.parametrize("frame", [NED, MOCAP, COLMAP, NS])
def test_mock_policies_return_frame_tagged_trajectories(frame):
    obs = _build_obs(frame)
    goal = Point.of(1, 0, 0, frame)
    for pol in [
        MockStraightLine(MockStraightLineConfig(goal=goal)),
        MockNoisy(MockNoisyConfig(goal=goal, seed=0)),
    ]:
        traj = pol.observe(obs)
        assert isinstance(traj, Trajectory)
        assert traj.frame is frame


# ---------------------------------------------------------------------------
# 3. StateSensor exposes only frame-tagged position keys (no bare ndarrays).
# ---------------------------------------------------------------------------


def test_state_sensor_does_not_emit_bare_arrays():
    obs = _build_obs(NED)
    for key in obs.keys():
        value = obs.require(key)
        if isinstance(value, np.ndarray) and value.shape == (3,):
            pytest.fail(
                f"observation key {key!r} is a bare 3-vector ndarray — "
                f"wrap in a frame-tagged Point or remove the dotted key"
            )


# ---------------------------------------------------------------------------
# 4. SE3 / Sim3 always carry src and dst frames.
# ---------------------------------------------------------------------------


def test_se3_sim3_require_src_and_dst_frames():
    sig_se3 = inspect.signature(SE3.__init__)
    sig_sim3 = inspect.signature(Sim3.__init__)
    for sig in (sig_se3, sig_sim3):
        assert "src" in sig.parameters and "dst" in sig.parameters


def test_apply_rejects_unframed_arrays():
    """Applying a transform to a bare ndarray must fail — preserves the
    invariant that transforms cross module boundaries only with frame tags."""
    T = SE3.identity(NED, MOCAP)
    with pytest.raises(TypeError, match="no .frame attribute"):
        T @ np.array([1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# 5. Static analysis: in our source tree, any function whose name contains
#    "pos" or "position" and returns must not statically return raw arrays
#    from a public module surface. We do a string scan as a tripwire — the
#    real enforcement is the typed Point/Pose etc.
# ---------------------------------------------------------------------------


PUBLIC_MODULES = [
    "geometry/__init__.py",
    "policy/__init__.py",
    "sensors/__init__.py",
    "sim/dynamics_state.py",
]

POSITION_RETURN_PATTERN = re.compile(
    r"def\s+\w*(?:position|pos|point|pose|trajectory)\w*\([^)]*\)\s*->\s*np\.ndarray",
    re.IGNORECASE,
)


def test_no_public_function_returns_bare_ndarray_for_a_position():
    offenders: list[str] = []
    for rel in PUBLIC_MODULES:
        text = (SRC / rel).read_text()
        for match in POSITION_RETURN_PATTERN.finditer(text):
            offenders.append(f"{rel}: {match.group(0)}")
    assert not offenders, (
        "Public modules return bare np.ndarray for position-like values:\n  "
        + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# 6. FrameGraph conversion preserves the frame tag.
# ---------------------------------------------------------------------------


def test_bounds_criterion_requires_frame_tagged_corners():
    """`BoundsCriterion` must read frames from its corner Points, not a string."""
    from falsify.safety import BoundsCriterion
    c = BoundsCriterion(lower=Point.of(-1, -1, 0, NED), upper=Point.of(1, 1, 2, NED))
    assert c.operates_in_frame == "ned"
    # Cross-frame corners must be rejected at construction time.
    with pytest.raises(ValueError, match="frames disagree"):
        BoundsCriterion(lower=Point.of(-1, -1, 0, NED), upper=Point.of(1, 1, 2, MOCAP))


def test_failure_record_carries_frame_tagged_states():
    from falsify.safety import BoundsCriterion, FailureDetector
    import numpy as np
    g = FrameGraph()
    g.register_frame(NED)
    det = FailureDetector(
        [BoundsCriterion(lower=Point.of(-1, -1, 0, NED), upper=Point.of(1, 1, 2, NED))],
        g,
    )
    from falsify.sim.dynamics_state import DroneState
    s_safe = DroneState(
        pos=Point.of(0, 0, 1, NED), vel=np.zeros(3),
        quat_xyzw=np.array([0, 0, 0, 1.0]), t=0.0,
    )
    s_bad = DroneState(
        pos=Point.of(2, 0, 1, NED), vel=np.zeros(3),
        quat_xyzw=np.array([0, 0, 0, 1.0]), t=0.1,
    )
    det.update(s_safe, 0)
    rec = det.update(s_bad, 1)
    assert rec is not None
    # Both failure_state and last_safe_state must be frame-tagged.
    assert rec.failure_state.pos.frame.name == "ned"
    assert rec.last_safe_state.pos.frame.name == "ned"


def test_episode_goal_is_frame_tagged_point():
    """`FalsificationEpisode.goal` must be a `Point`, not a raw list/dict entry."""
    from falsify.orchestrator import FalsificationEpisode
    from falsify.sim import EpisodeTrace
    ep = FalsificationEpisode(
        scene_cfg={}, frame_cfg={}, episode_cfg={},
        trace=EpisodeTrace(),
        goal=Point.of(1, 2, 3, NED),
    )
    assert isinstance(ep.goal, Point)
    assert ep.goal.frame.name == "ned"


def test_frame_graph_converts_with_explicit_dst_frame():
    g = FrameGraph()
    g.register_frame(NED)
    g.register_frame(MOCAP)
    g.register_edge(SE3(R=np.eye(3), t=np.zeros(3), src=NED, dst=MOCAP))
    p = Point.of(0, 0, 0, NED)
    out = g.convert(p, to="mocap")
    assert out.frame is g.frame("mocap")
    assert out.frame.name == "mocap"
