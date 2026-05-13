"""VLA policy tests — uses a stub OpenPI client so no server is required."""

from __future__ import annotations

import numpy as np
import pytest

from falsify.geometry import (
    COLMAP, MOCAP, NED, NS, FrameGraph, Point, SE3, Sim3,
)
from falsify.policy import VLAPolicy, VLAPolicyConfig, Observation
from falsify.sim.dynamics_state import DroneState


def _graph():
    g = FrameGraph()
    for f in (NED, MOCAP, COLMAP, NS):
        g.register_frame(f)
    g.register_edge(SE3(R=np.diag([1.0, -1.0, -1.0]), t=np.zeros(3), src=NED, dst=MOCAP))
    g.register_edge(Sim3(s=1.0, R=np.eye(3), t=np.zeros(3), src=MOCAP, dst=COLMAP))
    g.register_edge(Sim3(s=1.0, R=np.eye(3), t=np.zeros(3), src=COLMAP, dst=NS))
    return g


class _StubOpenPIClient:
    """Returns a fixed action chunk so we can assert frame conversions."""
    def __init__(self, actions: np.ndarray):
        self._actions = actions
        self.payloads = []

    def infer(self, payload: dict) -> dict:
        self.payloads.append(payload)
        return {"actions": self._actions}


def _obs_with_dual_cameras(pos_ned):
    state = DroneState(
        pos=pos_ned, vel=np.zeros(3),
        quat_xyzw=np.array([0, 0, 0, 1.0]), t=0.0,
    )
    fwd = np.zeros((8, 8, 3), dtype=np.uint8)
    dwn = np.zeros((8, 8, 3), dtype=np.uint8)
    return Observation(
        state=state,
        data={"state.pos": pos_ned, "images.forward": fwd, "images.downward": dwn},
        prompt="fly through the gate",
    )


def test_vla_declares_dual_camera_modalities():
    pol = VLAPolicy(VLAPolicyConfig(), _graph())
    assert pol.required_modalities == frozenset({"images.forward", "images.downward"})


def test_vla_query_round_trips_through_mocap():
    g = _graph()
    deltas = np.tile([0.0, 0.1, 0.0], (5, 1))   # 5 steps of +0.1 y in MOCAP
    client = _StubOpenPIClient(actions=deltas)
    pol = VLAPolicy(VLAPolicyConfig(actions_per_chunk=5, image_size=8), g)
    pol._client = client
    pol._connected = True

    start_ned = Point.of(0.0, 0.0, 1.0, NED)
    obs = _obs_with_dual_cameras(start_ned)
    traj_ned = pol.observe(obs)

    # Output is in NED.
    assert traj_ned.frame.name == "ned"
    # Trajectory should have actions_per_chunk + 1 (prepended start) waypoints.
    assert traj_ned.positions.shape == (6, 3)
    # First waypoint equals the input start.
    np.testing.assert_allclose(traj_ned.positions[0], start_ned.xyz, atol=1e-12)
    # Server payload received the MOCAP-frame state.
    sent_state = client.payloads[0]["observation/state"]
    expected_mocap = g.convert(start_ned, to="mocap").xyz
    np.testing.assert_allclose(sent_state[:3], expected_mocap, atol=1e-6)


def test_vla_passes_prompt_through():
    g = _graph()
    client = _StubOpenPIClient(actions=np.zeros((3, 3)))
    pol = VLAPolicy(VLAPolicyConfig(prompt="default prompt", actions_per_chunk=3, image_size=8), g)
    pol._client = client
    pol._connected = True
    obs = _obs_with_dual_cameras(Point.of(0, 0, 1.0, NED))
    pol.observe(obs)
    assert client.payloads[0]["prompt"] == "fly through the gate"   # obs.prompt wins


def test_vla_payload_matches_sousvide_format():
    g = _graph()
    deltas = np.tile([0.05, 0.0, 0.0], (4, 1))
    client = _StubOpenPIClient(actions=deltas)
    pol = VLAPolicy(VLAPolicyConfig(actions_per_chunk=4, image_size=16), g)
    pol._client = client
    pol._connected = True
    obs = _obs_with_dual_cameras(Point.of(0.0, 0.0, 1.0, NED))
    pol.observe(obs)

    sent = client.payloads[0]
    # SousVide-style keys.
    assert set(sent) == {
        "observation/image", "observation/wrist_image", "observation/3pov_1",
        "observation/state", "prompt",
    }
    # All three image channels are 256? No — image_size override is 16.
    assert sent["observation/image"].shape       == (16, 16, 3)
    assert sent["observation/wrist_image"].shape == (16, 16, 3)
    assert sent["observation/3pov_1"].shape      == (16, 16, 3)
    # 3pov default is all-zero.
    assert sent["observation/3pov_1"].sum() == 0
    # State vector is float32 shape (7,) with yaw at index 3 (zero here).
    state_vec = sent["observation/state"]
    assert state_vec.dtype == np.float32
    assert state_vec.shape == (7,)
    assert state_vec[3] == 0.0
    np.testing.assert_array_equal(state_vec[4:], np.zeros(3))


def test_vla_negates_yaw_for_server():
    """NED yaw and MOCAP yaw spin in opposite senses; sign must flip in payload."""
    from scipy.spatial.transform import Rotation as _R
    g = _graph()
    client = _StubOpenPIClient(actions=np.zeros((1, 3)))
    pol = VLAPolicy(VLAPolicyConfig(actions_per_chunk=1, image_size=8), g)
    pol._client = client
    pol._connected = True
    yaw_ned = 0.4
    q = _R.from_euler("z", yaw_ned).as_quat()  # xyzw
    state = DroneState(
        pos=Point.of(0, 0, 1.0, NED), vel=np.zeros(3),
        quat_xyzw=q, t=0.0,
    )
    obs = Observation(
        state=state,
        data={"state.pos": state.pos,
              "images.forward":  np.zeros((8, 8, 3), dtype=np.uint8),
              "images.downward": np.zeros((8, 8, 3), dtype=np.uint8)},
    )
    pol.observe(obs)
    np.testing.assert_allclose(client.payloads[0]["observation/state"][3], -yaw_ned, atol=1e-6)


def test_vla_rejects_non_ned_state():
    g = _graph()
    client = _StubOpenPIClient(actions=np.zeros((3, 3)))
    pol = VLAPolicy(VLAPolicyConfig(actions_per_chunk=3, image_size=8), g)
    pol._client = client
    pol._connected = True
    state = DroneState(
        pos=Point.of(0, 0, 0, MOCAP),
        vel=np.zeros(3), quat_xyzw=np.array([0, 0, 0, 1.0]), t=0.0,
    )
    obs = Observation(
        state=state,
        data={"state.pos": state.pos,
              "images.forward":  np.zeros((8, 8, 3), dtype=np.uint8),
              "images.downward": np.zeros((8, 8, 3), dtype=np.uint8)},
    )
    with pytest.raises(ValueError, match="frame mismatch"):
        pol.observe(obs)
