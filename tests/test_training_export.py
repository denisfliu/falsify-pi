"""Tests for the training-data export pipeline.

No GPU / renderer required — uses a stub renderer that produces solid-color
images so the exporter can be exercised end-to-end on CI.
"""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stub_renderer(pose, intrinsics):
    """Solid-color render whose colour depends on the camera resolution so
    tests can tell forward from downward."""
    h = int(intrinsics["height"])
    w = int(intrinsics["width"])
    # Distinguish cameras by a color cue (R for forward 640x360, G for downward 320x320).
    if w == 640:
        color = (255, 0, 0)   # red in RGB
    else:
        color = (0, 255, 0)   # green in RGB
    img = np.tile(np.array(color, dtype=np.uint8)[None, None, :], (h, w, 1))
    return img, None


def _build_minimal_scene(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write a tiny scene + frame + embodiment so the exporter can run."""
    scene_yaml = tmp_path / "scene.yaml"
    scene_yaml.write_text("""
scene_key: test
frames:
  - { name: mocap }
  - { name: ned }
  - { name: ns }
  - { name: cam_body }
  - { name: cam_forward }
  - { name: cam_downward }
transforms:
  - { src: ned, dst: mocap, type: se3_inline, R: [[1, 0, 0], [0, -1, 0], [0, 0, -1]], t: [0, 0, 0] }
  - { src: mocap, dst: ns, type: sim3_inline, scale: 0.5, R: [[1,0,0],[0,1,0],[0,0,1]], t: [0,0,0] }
  - { src: cam_body, dst: cam_forward, type: se3_inline,
      R: [[0, 1, 0], [0, 0, -1], [-1, 0, 0]], t: [0.03, -0.01, 0.10] }
  - { src: cam_body, dst: cam_downward, type: se3_inline,
      R: [[0, 1, 0], [1, 0, 0], [0, 0, -1]], t: [0, 0, 0.05] }
""")
    frame_yaml = tmp_path / "frame.yaml"
    frame_yaml.write_text("""
mass: 1.0
massless_inertia: [0.01, 0.01, 0.02]
arm_front: [0.075, 0.1]
arm_back: [0.075, 0.1]
motor_thrust_coeff: 6.9
motor_torque_coeff: 0.69
number_of_rotors: 4
cameras:
  forward:
    frame: cam_forward
    model: pinhole
    intrinsics: { width: 640, height: 360, fx: 462.95, fy: 463.0, cx: 323.08, cy: 181.18 }
  downward:
    frame: cam_downward
    model: pinhole
    intrinsics: { width: 320, height: 320, fx: 200.0, fy: 200.0, cx: 160.0, cy: 160.0 }
""")
    emb_yaml = tmp_path / "embodiment.yaml"
    emb_yaml.write_text("""
name: test_emb
fps: 10
image_size: 64
robot_type: panda
cameras:
  - { column: image,        source: render, camera_name: forward,  channel_order: BGR, image_size: 64 }
  - { column: wrist_image,  source: render, camera_name: downward, channel_order: BGR, image_size: 64 }
  - { column: "3pov_1",     source: zeros, image_size: 64 }
state:
  - { name: x_mocap }
  - { name: y_mocap }
  - { name: z_mocap }
  - { name: yaw_mocap }
  - { name: gripper }
  - { name: zero }
  - { name: zero }
actions:
  - { name: d_x_mocap }
  - { name: d_y_mocap }
  - { name: d_z_mocap }
  - { name: d_yaw_mocap }
  - { name: d_gripper }
  - { name: zero }
  - { name: zero }
yaw_wrap: pi
first_action: zeros
""")
    return scene_yaml, frame_yaml, emb_yaml


# ---------------------------------------------------------------------------
# Trajectory roundtrip
# ---------------------------------------------------------------------------


def test_trajectory_roundtrip(tmp_path):
    from falsify.training import Trajectory, save_trajectory, load_trajectory

    n = 20
    times = np.linspace(0, 2.0, n)
    positions = np.cumsum(np.ones((n, 3)) * 0.05, axis=0)
    quats = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
    traj = Trajectory(
        times=times,
        positions_ned=positions,
        quaternions_xyzw=quats,
        prompt="test prompt",
        source="unit_test",
    )
    out = save_trajectory(tmp_path / "traj.npz", traj)
    loaded = load_trajectory(out)
    np.testing.assert_allclose(loaded.times, traj.times)
    np.testing.assert_allclose(loaded.positions_ned, traj.positions_ned)
    np.testing.assert_allclose(loaded.quaternions_xyzw, traj.quaternions_xyzw)
    assert loaded.prompt == "test prompt"
    assert loaded.source == "unit_test"


def test_trajectory_resample_changes_rate(tmp_path):
    from falsify.training import Trajectory, resample

    n = 11
    times = np.linspace(0, 1.0, n)        # 10 Hz
    positions = np.stack([times, np.zeros(n), np.zeros(n)], axis=1)
    quats = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
    traj = Trajectory(times=times, positions_ned=positions, quaternions_xyzw=quats)
    resampled = resample(traj, hz=20.0)
    # Approximately 21 samples for 1 second at 20 Hz.
    assert 19 <= len(resampled) <= 22
    # Linear interp preserves the position-time relation.
    np.testing.assert_allclose(resampled.positions_ned[:, 0], resampled.times, atol=1e-6)


# ---------------------------------------------------------------------------
# Exporter end-to-end
# ---------------------------------------------------------------------------


def test_exporter_produces_reference_schema(tmp_path):
    from falsify.training import (
        TrainingDataExporter, Trajectory, load_embodiment,
    )
    from falsify.io import build_frame_graph, load_yaml

    scene_yaml, frame_yaml, emb_yaml = _build_minimal_scene(tmp_path)
    scene_cfg = load_yaml(scene_yaml)
    frame_cfg = load_yaml(frame_yaml)
    embodiment = load_embodiment(emb_yaml)
    fg = build_frame_graph(scene_cfg, base_path=tmp_path)

    n = 5
    times = np.arange(n) / embodiment.fps
    pos_ned = np.stack([0.1 * np.arange(n), 0.2 * np.arange(n), -1.5 * np.ones(n)], axis=1)
    quats = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
    traj = Trajectory(
        times=times, positions_ned=pos_ned, quaternions_xyzw=quats,
        prompt="unit test", source="unit_test",
    )

    exporter = TrainingDataExporter(
        scene_cfg=scene_cfg, frame_cfg=frame_cfg, frame_graph=fg,
        renderer=_stub_renderer, embodiment=embodiment,
    )

    result = exporter.export_episode(
        traj, tmp_path / "ep", episode_index=3, index_offset=100, task_index=0,
    )
    assert result.n_frames == n

    table = pq.read_table(result.parquet_path)
    cols = table.column_names
    # Reference schema column set + order.
    expected_cols = [
        "image", "wrist_image", "3pov_1",
        "state", "actions",
        "timestamp", "frame_index", "episode_index", "index", "task_index",
    ]
    assert cols == expected_cols, f"got {cols}"

    # Schema types.
    schema = table.schema
    for img_col in ("image", "wrist_image", "3pov_1"):
        f = schema.field(img_col).type
        assert f.num_fields == 2
        assert {f.field(i).name for i in range(2)} == {"bytes", "path"}
    assert schema.field("state").type.list_size == 7
    assert schema.field("actions").type.list_size == 7
    assert str(schema.field("timestamp").type) == "float"
    for col in ("frame_index", "episode_index", "index", "task_index"):
        assert str(schema.field(col).type) == "int64"

    # HF metadata blob is present.
    md = schema.metadata or {}
    assert b"huggingface" in md, f"got keys: {list(md)}"
    hf = json.loads(md[b"huggingface"].decode())
    features = hf["info"]["features"]
    for col in ("image", "wrist_image", "3pov_1"):
        assert features[col]["_type"] == "Image"
    assert features["state"]["length"] == 7
    assert features["actions"]["length"] == 7

    # First row: action = zeros, frame_index = 0, index = offset.
    row = table.slice(0, 1).to_pydict()
    assert row["frame_index"][0] == 0
    assert row["episode_index"][0] == 3
    assert row["index"][0] == 100
    assert row["task_index"][0] == 0
    assert all(v == 0.0 for v in row["actions"][0])

    # Last row: index = offset + n - 1.
    last = table.slice(n - 1, 1).to_pydict()
    assert last["frame_index"][0] == n - 1
    assert last["index"][0] == 100 + n - 1

    # Image bytes decode as PNG and round-trip to BGR-swapped color.
    from PIL import Image as _Image
    img_bytes = row["image"][0]["bytes"]
    decoded = np.asarray(_Image.open(io.BytesIO(img_bytes)))
    # forward stub renders red in RGB → BGR-swapped → (0, 0, 255) when PIL re-reads
    # i.e. PIL.fromarray on a BGR buffer stores BGR-as-RGB in the PNG; the round-trip
    # gives back BGR. Pixel index 2 (last channel) should be max for a "red-in-RGB"
    # solid-color render after BGR swap.
    assert decoded[0, 0, 2] == 255
    assert decoded[0, 0, 0] == 0
    # Image path is just the filename.
    assert row["image"][0]["path"] == "frame_000000.png"

    # Manifest sanity.
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["embodiment"] == "test_emb"
    assert manifest["n_frames"] == n
    assert manifest["episode_index"] == 3


def test_exporter_action_yaw_wraps_to_pi_interval(tmp_path):
    """An ~360° yaw step around the discontinuity must wrap to ~0, not 2π."""
    from falsify.training import (
        TrainingDataExporter, Trajectory, load_embodiment,
    )
    from falsify.io import build_frame_graph, load_yaml
    from scipy.spatial.transform import Rotation as _R

    scene_yaml, frame_yaml, emb_yaml = _build_minimal_scene(tmp_path)
    scene_cfg = load_yaml(scene_yaml)
    frame_cfg = load_yaml(frame_yaml)
    embodiment = load_embodiment(emb_yaml)
    fg = build_frame_graph(scene_cfg, base_path=tmp_path)

    # Two states: yaw_ned ≈ +π and yaw_ned ≈ -π (same physical orientation).
    n = 2
    times = np.array([0.0, 0.1])
    pos = np.zeros((n, 3))
    quats = np.stack([
        _R.from_euler("z", +3.10).as_quat(),
        _R.from_euler("z", -3.10).as_quat(),
    ])
    traj = Trajectory(times=times, positions_ned=pos, quaternions_xyzw=quats)
    exporter = TrainingDataExporter(
        scene_cfg=scene_cfg, frame_cfg=frame_cfg, frame_graph=fg,
        renderer=_stub_renderer, embodiment=embodiment,
    )
    result = exporter.export_episode(traj, tmp_path / "yawtest")
    table = pq.read_table(result.parquet_path)
    action_yaw_delta = table.column("actions")[1].as_py()[3]
    # Unwrapped: -6.20 rad; wrapped to (-π, π] ≈ +0.083 rad
    assert abs(action_yaw_delta) < 0.2, f"yaw not wrapped: got {action_yaw_delta}"
