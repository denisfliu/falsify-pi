"""Round-trip tests for the `FrameGraph` and the YAML scene loader.

These tests do not require GPU or any of the heavy gsplat dependencies — they
exercise the pure-Python geometry layer end-to-end.
"""

from __future__ import annotations

import itertools
import textwrap
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from falsify.geometry import (
    Frame, NED, MOCAP, COLMAP, NS, CAM_BODY, CAM_FORWARD,
    Point, Trajectory,
    SE3, Sim3,
    FrameGraph,
)
from falsify.io import build_frame_graph, load_yaml


# ---------------------------------------------------------------------------
# SE3 / Sim3 algebra
# ---------------------------------------------------------------------------


def test_se3_inverse_recovers_point():
    rot = R.from_euler("xyz", [0.3, -0.6, 1.2]).as_matrix()
    T = SE3(R=rot, t=np.array([1.0, -2.0, 0.5]), src=NED, dst=MOCAP)
    p = Point.of(1.1, 2.2, 3.3, NED)
    q = T @ p
    assert q.frame is MOCAP
    back = T.inv() @ q
    np.testing.assert_allclose(back.xyz, p.xyz, atol=1e-12)


def test_sim3_inverse_recovers_point():
    rot = R.from_euler("zyx", [0.1, 0.4, -0.8]).as_matrix()
    T = Sim3(s=0.37, R=rot, t=np.array([4.0, -1.0, 2.5]), src=COLMAP, dst=NS)
    p = Point.of(0.2, 0.1, -0.4, COLMAP)
    q = T @ p
    assert q.frame is NS
    back = T.inv() @ q
    np.testing.assert_allclose(back.xyz, p.xyz, atol=1e-12)


def test_compose_se3_se3_yields_se3():
    A = SE3(R=R.from_euler("z", 0.5).as_matrix(), t=np.array([1, 0, 0]), src=NED, dst=MOCAP)
    B = SE3(R=R.from_euler("x", 0.3).as_matrix(), t=np.array([0, 1, 0]), src=MOCAP, dst=COLMAP)
    C = B @ A
    assert isinstance(C, SE3)
    assert (C.src.name, C.dst.name) == ("ned", "colmap")
    p = Point.of(1, 2, 3, NED)
    np.testing.assert_allclose((C @ p).xyz, (B @ (A @ p)).xyz, atol=1e-12)


def test_compose_se3_with_sim3_promotes_to_sim3():
    A = Sim3(s=2.5, R=np.eye(3), t=np.zeros(3), src=NED, dst=MOCAP)
    B = SE3(R=R.from_euler("z", 0.7).as_matrix(), t=np.array([1, 1, 1]), src=MOCAP, dst=COLMAP)
    C = B @ A
    assert isinstance(C, Sim3)
    assert C.s == pytest.approx(2.5)
    p = Point.of(1, 2, 3, NED)
    np.testing.assert_allclose((C @ p).xyz, (B @ (A @ p)).xyz, atol=1e-12)


def test_apply_rejects_frame_mismatch():
    T = SE3.identity(NED, MOCAP)
    p_wrong = Point.of(0, 0, 0, COLMAP)
    with pytest.raises(ValueError, match="frame mismatch"):
        T @ p_wrong


# ---------------------------------------------------------------------------
# FrameGraph BFS conversion
# ---------------------------------------------------------------------------


def _build_test_graph() -> FrameGraph:
    g = FrameGraph()
    for f in (NED, MOCAP, COLMAP, NS, CAM_BODY, CAM_FORWARD):
        g.register_frame(f)
    # ned ↔ mocap: rigid permutation
    g.register_edge(SE3(
        R=np.diag([1.0, -1.0, -1.0]), t=np.zeros(3), src=NED, dst=MOCAP,
    ))
    # mocap ↔ colmap: similarity
    g.register_edge(Sim3(
        s=1.7, R=R.from_euler("zyx", [0.2, -0.1, 0.4]).as_matrix(),
        t=np.array([1.0, 2.0, -0.5]), src=MOCAP, dst=COLMAP,
    ))
    # colmap ↔ ns: similarity (nerfstudio-style)
    g.register_edge(Sim3(
        s=0.31, R=R.from_euler("x", 0.6).as_matrix(),
        t=np.array([-0.1, 0.0, 0.05]), src=COLMAP, dst=NS,
    ))
    # cam_body ↔ ned: rigid (body→world style)
    g.register_edge(SE3(
        R=R.from_euler("y", 0.05).as_matrix(),
        t=np.array([0.0, 0.0, 0.0]), src=CAM_BODY, dst=NED,
    ))
    # cam_forward ↔ cam_body: small offset
    g.register_edge(SE3(
        R=np.eye(3), t=np.array([0.1, 0.0, 0.0]), src=CAM_FORWARD, dst=CAM_BODY,
    ))
    return g


def test_graph_round_trips_for_all_frame_pairs():
    g = _build_test_graph()
    rng = np.random.default_rng(0)
    frame_names = [f.name for f in g.frames]
    for a, b in itertools.product(frame_names, repeat=2):
        if a == b:
            continue
        p_a = Point(rng.standard_normal(3), frame=Frame(a))
        # The graph stores its own Frame object — use that to keep equality.
        p_a = Point(p_a.xyz, frame=g.frame(a))
        p_b = g.convert(p_a, to=b)
        assert p_b.frame.name == b
        round = g.convert(p_b, to=a)
        np.testing.assert_allclose(
            round.xyz, p_a.xyz, atol=1e-10,
            err_msg=f"round-trip failed for {a} → {b} → {a}",
        )


def test_graph_convert_trajectory():
    g = _build_test_graph()
    rng = np.random.default_rng(1)
    times = np.linspace(0, 1, 20)
    positions = rng.standard_normal((20, 3))
    velocities = rng.standard_normal((20, 3))
    traj = Trajectory(times=times, positions=positions, frame=g.frame("ned"),
                      velocities=velocities)
    out = g.convert(traj, to="ns")
    assert out.frame.name == "ns"
    back = g.convert(out, to="ned")
    np.testing.assert_allclose(back.positions, traj.positions, atol=1e-10)
    np.testing.assert_allclose(back.velocities, traj.velocities, atol=1e-10)


def test_graph_describe_lists_everything():
    g = _build_test_graph()
    text = g.describe()
    for name in ("ned", "mocap", "colmap", "ns", "cam_body", "cam_forward"):
        assert name in text


def test_graph_rejects_duplicate_inverse_edge():
    g = FrameGraph()
    g.register_frame(NED)
    g.register_frame(MOCAP)
    g.register_edge(SE3.identity(NED, MOCAP))
    with pytest.raises(ValueError, match="inverse"):
        g.register_edge(SE3.identity(MOCAP, NED))


def test_graph_no_path_raises():
    g = FrameGraph()
    g.register_frame(NED)
    g.register_frame(MOCAP)
    # No edges registered.
    with pytest.raises(KeyError):
        g.convert(Point.of(0, 0, 0, NED), to="mocap")


# ---------------------------------------------------------------------------
# YAML → FrameGraph loader
# ---------------------------------------------------------------------------


def test_yaml_inline_loader(tmp_path: Path):
    yaml_text = textwrap.dedent("""
        frames:
          - { name: a }
          - { name: b }
          - { name: c }
        transforms:
          - { src: a, dst: b, type: permutation, preset: "perm5" }
          - { src: b, dst: c, type: sim3_inline, scale: 0.5, R: [[1,0,0],[0,1,0],[0,0,1]], t: [1, 2, 3] }
    """)
    cfg_path = tmp_path / "scene.yaml"
    cfg_path.write_text(yaml_text)
    cfg = load_yaml(cfg_path)
    graph = build_frame_graph(cfg, base_path=cfg_path.parent)
    assert graph.has_path("a", "c")
    p = Point.of(1, 2, 3, graph.frame("a"))
    converted = graph.convert(p, to="c")
    back = graph.convert(converted, to="a")
    np.testing.assert_allclose(back.xyz, p.xyz, atol=1e-10)


def test_yaml_user_can_declare_arbitrary_frame_names(tmp_path: Path):
    """The whole point: you can introduce a totally new frame in YAML alone."""
    yaml_text = textwrap.dedent("""
        frames:
          - { name: weird_room, notes: "future custom frame" }
          - { name: ned }
        transforms:
          - { src: weird_room, dst: ned, type: se3_inline, R: [[0,1,0],[-1,0,0],[0,0,1]], t: [10, -5, 2] }
    """)
    cfg = load_yaml(tmp_path / "scene.yaml" if False else _write_tmp(yaml_text, tmp_path))
    graph = build_frame_graph(cfg, base_path=tmp_path)
    p = Point.of(0.5, 0.5, 0.5, graph.frame("weird_room"))
    q = graph.convert(p, to="ned")
    assert q.frame.name == "ned"
    np.testing.assert_allclose(graph.convert(q, to="weird_room").xyz, p.xyz, atol=1e-10)


def _write_tmp(text: str, tmp_path: Path) -> Path:
    p = tmp_path / "scene.yaml"
    p.write_text(text)
    return p


def test_yaml_dataparser_loader(tmp_path: Path):
    """dataparser_transforms.json convention: ``p_ns = scale * (R p + t)``."""
    import json
    R_mat = R.from_euler("z", 0.3).as_matrix()
    t_pre = np.array([0.1, -0.2, 0.05])
    scale = 0.42
    transform = np.concatenate([R_mat, t_pre[:, None]], axis=1)  # (3, 4)
    dp_path = tmp_path / "dataparser_transforms.json"
    dp_path.write_text(json.dumps({"transform": transform.tolist(), "scale": scale}))

    yaml_text = textwrap.dedent(f"""
        frames:
          - {{ name: colmap }}
          - {{ name: ns }}
        transforms:
          - {{ src: colmap, dst: ns, type: dataparser, path: dataparser_transforms.json }}
    """)
    cfg = load_yaml(_write_tmp(yaml_text, tmp_path))
    graph = build_frame_graph(cfg, base_path=tmp_path)
    p_colmap = np.array([1.0, 2.0, 3.0])
    expected = scale * (R_mat @ p_colmap + t_pre)
    out = graph.convert(Point(p_colmap, frame=graph.frame("colmap")), to="ns")
    np.testing.assert_allclose(out.xyz, expected, atol=1e-10)
