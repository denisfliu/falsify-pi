"""Closed-loop rollout from a recorded synth episode's initialization.

We previously validated *open-loop, single-step* action-prediction MSE for the
center-gate policies on the gsplat-rendered synth datasets
(``scripts/action_prediction/eval_action_prediction.py``): feed each recorded
frame to the policy, compare ``pred_chunk[0]`` to the recorded action. That
never closes the loop, so it can't reveal compounding drift.

This script does the closed-loop counterpart: it lifts the **initial drone
state** out of each recorded synth episode (first parquet frame), sets it as
``run_episode``'s ``initial_state_override``, and lets the **live** policy fly
through the scene — the sim renders its own gsplat observations each chunk.
Per episode it writes the closed-loop rollout trace, the recorded ground-truth
trace, a forward-cam flythrough mp4, and a 3-D HTML overlaying rollout vs GT
(+ the gate). A top-level ``index.html`` links them all.

Two cohorts are run back to back; switching the policy YAML between them flips
``bridge_policy_id`` and the first ``PiGatewayPolicy._ensure_connected`` of the
new cohort issues the ``/admin/switch_policy`` swap automatically.

Usage::

    export PI_API_KEY=...
    source tools/env.sh
    PYTHONPATH=src python scripts/debug/rollout_from_dataset_init.py \\
        --left-dataset  data/no_3pov_v3/synth_center_from_left \\
        --right-dataset data/no_3pov_v3/synth_center_from_right \\
        --left-policy-config  configs/policies/pi_gateway/nonhistory_all_left.yaml \\
        --right-policy-config configs/policies/pi_gateway/nonhistory_all_right.yaml \\
        --scene configs/scenes/center_gate.yaml \\
        --frame configs/frames/carl_dual.yaml \\
        --episodes 0-9 --horizon-s 11.0 --no-rtc

Frame contract: NED is the simulator/policy boundary; the synth parquet state
is MOCAP (``[px,py,pz,-yaw_ned,0,0,0]``). All MOCAP↔NED conversions go through
the scene ``FrameGraph`` — never hardcode perm5.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve(p: str | Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (REPO_ROOT / pp).resolve()


def _parse_episodes(spec: str) -> list[int]:
    """Parse '0-9' or '0,3,5' (or a mix: '0-4,7,9') into a sorted index list."""
    out: set[int] = set()
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(tok))
    return sorted(out)


def _episode_path(dataset: Path, idx: int) -> Path:
    return dataset / "data" / "chunk-000" / f"episode-{idx:06d}.parquet"


def _load_prompt(dataset: Path, task_index: int) -> str:
    """Read the verbatim task string from the dataset's tasks.parquet.

    Prompts MUST match training data exactly (see repo memory); we never
    invent paraphrases — we read whatever the dataset shipped.
    """
    import pyarrow.parquet as pq
    tp = dataset / "meta" / "tasks.parquet"
    if not tp.is_file():
        raise FileNotFoundError(f"tasks parquet missing: {tp}")
    t = pq.read_table(tp).to_pandas()
    text_col = "task" if "task" in t.columns else "__index_level_0__"
    mapping = {int(r["task_index"]): str(r[text_col]) for _, r in t.iterrows()}
    if task_index not in mapping:
        raise KeyError(f"task_index {task_index} not in {tp} (have {sorted(mapping)})")
    return mapping[task_index]


def _read_episode_states(dataset: Path, idx: int):
    """Return (state_rows (N,7), task_index, timestamps (N,)) for one episode."""
    import pyarrow.parquet as pq
    f = _episode_path(dataset, idx)
    if not f.is_file():
        raise FileNotFoundError(f"episode parquet missing: {f}")
    tbl = pq.read_table(
        f, columns=["observation.state", "timestamp", "task_index"]
    ).to_pydict()
    states = np.asarray([np.asarray(s, dtype=np.float64) for s in tbl["observation.state"]])
    times = np.asarray(tbl["timestamp"], dtype=np.float64)
    task_index = int(tbl["task_index"][0])
    return states, task_index, times


def _emit_overlay_html(
    out_path: Path,
    rollout_mocap: np.ndarray,
    gt_mocap: np.ndarray,
    scene_cfg: dict,
    *,
    title: str,
    failure_type: str,
) -> None:
    """Two MOCAP polylines (closed-loop rollout vs recorded GT) + the gate AABB
    + start/goal markers, as a self-contained HTML."""
    import plotly.graph_objects as go
    from falsify.visualization.eval_report import _gate_aabb_wireframe

    fig = go.Figure()

    # Gate AABB (MOCAP — no perturbation in these runs).
    region = scene_cfg.get("gate_region") or {}
    if region.get("aabb_min") is not None:
        gx, gy, gz = _gate_aabb_wireframe(
            np.asarray(region["aabb_min"], dtype=np.float64),
            np.asarray(region["aabb_max"], dtype=np.float64),
        )
        fig.add_trace(go.Scatter3d(
            x=gx, y=gy, z=gz, mode="lines", name="gate AABB",
            line=dict(color="rgb(120,120,120)", width=3),
        ))

    fig.add_trace(go.Scatter3d(
        x=gt_mocap[:, 0], y=gt_mocap[:, 1], z=gt_mocap[:, 2], mode="lines",
        name="recorded GT", line=dict(color="rgb(46,160,67)", width=5),
    ))
    fig.add_trace(go.Scatter3d(
        x=rollout_mocap[:, 0], y=rollout_mocap[:, 1], z=rollout_mocap[:, 2],
        mode="lines", name=f"closed-loop rollout ({failure_type})",
        line=dict(color="rgb(231,76,60)", width=5),
    ))

    # Start (rollout origin) + goal markers.
    fig.add_trace(go.Scatter3d(
        x=[rollout_mocap[0, 0]], y=[rollout_mocap[0, 1]], z=[rollout_mocap[0, 2]],
        mode="markers", name="start", marker=dict(size=5, color="rgb(46,160,67)"),
    ))
    goal = scene_cfg.get("goal_position_mocap")
    if goal is not None:
        g = np.asarray(goal, dtype=np.float64)
        fig.add_trace(go.Scatter3d(
            x=[g[0]], y=[g[1]], z=[g[2]], mode="markers", name="goal",
            marker=dict(size=6, color="rgb(241,196,15)", symbol="diamond"),
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x (mocap)", yaxis_title="y (mocap)", zaxis_title="z (mocap)",
            aspectmode="data",
        ),
        legend=dict(orientation="h"),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def _write_index(out_dir: Path, rows: list[dict]) -> Path:
    """Tiny index.html linking every episode's overlay + flythrough."""
    lines = [
        "<html><head><meta charset='utf-8'><title>rollout_from_init</title>",
        "<style>body{font-family:sans-serif;margin:2rem} "
        "table{border-collapse:collapse} td,th{border:1px solid #ccc;padding:6px 10px} "
        "tr:nth-child(even){background:#f6f6f6}</style></head><body>",
        "<h2>Closed-loop rollout from recorded synth-episode initializations</h2>",
        "<table><tr><th>cohort</th><th>episode</th><th>outcome</th>"
        "<th>stop</th><th>n_states</th><th>overlay</th><th>flythrough</th></tr>",
    ]
    for r in rows:
        overlay = (f"<a href='{r['overlay']}'>overlay.html</a>"
                   if r.get("overlay") else "—")
        fly = (f"<a href='{r['flythrough']}'>flythrough.mp4</a>"
               if r.get("flythrough") else "—")
        lines.append(
            f"<tr><td>{r['cohort']}</td><td>{r['episode']}</td>"
            f"<td>{r.get('outcome','?')}</td><td>{r.get('stop','?')}</td>"
            f"<td>{r.get('n_states','?')}</td><td>{overlay}</td><td>{fly}</td></tr>"
        )
    lines.append("</table></body></html>")
    p = out_dir / "index.html"
    p.write_text("\n".join(lines))
    return p


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--left-dataset", type=Path,
                    default=Path("data/no_3pov_v3/synth_center_from_left"))
    ap.add_argument("--right-dataset", type=Path,
                    default=Path("data/no_3pov_v3/synth_center_from_right"))
    ap.add_argument("--left-policy-config", type=Path,
                    default=Path("configs/policies/pi_gateway/nonhistory_all_left.yaml"))
    ap.add_argument("--right-policy-config", type=Path,
                    default=Path("configs/policies/pi_gateway/nonhistory_all_right.yaml"))
    ap.add_argument("--scene", type=Path, default=Path("configs/scenes/center_gate.yaml"))
    ap.add_argument("--frame", type=Path, default=Path("configs/frames/carl_dual.yaml"))
    ap.add_argument("--episodes", type=str, default="0-9",
                    help="Episode indices to roll out per cohort (e.g. '0-9' or '0,3,5').")
    ap.add_argument("--cohorts", nargs="+", default=["left", "right"],
                    choices=["left", "right"],
                    help="Which cohorts to run (default both).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output dir. Default runs/rollout_from_init/<ts>/.")
    ap.add_argument("--hz", type=int, default=None,
                    help="Sim rate. Default: the policy YAML's hz (30).")
    ap.add_argument("--execute-chunk-size", type=int, default=None,
                    help="Override the policy YAML's execute_chunk_size "
                         "(only meaningful with --no-rtc).")
    ap.add_argument("--horizon-s", type=float, default=11.0,
                    help="Per-episode time budget. Key tunable — confirm a "
                         "single center-gate transit completes on the first run.")
    ap.add_argument("--no-rtc", action="store_true",
                    help="Force use_rtc=False (repo convention for "
                         "deterministic, ~22x-faster sweeps). One infer/chunk.")
    ap.add_argument("--no-flythrough", action="store_true",
                    help="Skip the forward-cam mp4 (saves render time).")
    ap.add_argument("--flythrough-every", type=int, default=2)
    ap.add_argument("--flythrough-fps", type=int, default=15)
    ap.add_argument("--no-viz", action="store_true",
                    help="Skip the per-episode overlay HTMLs.")
    args = ap.parse_args(argv)

    episodes = _parse_episodes(args.episodes)
    if not episodes:
        raise SystemExit("--episodes parsed to an empty set")

    out_dir = args.out or (REPO_ROOT / "runs" / "rollout_from_init"
                           / time.strftime("%Y%m%d_%H%M%S"))
    out_dir = _resolve(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[rollout-init] episodes={episodes} cohorts={args.cohorts}")
    print(f"[rollout-init] out={out_dir}")

    # ---- Smoke checks, then lazy imports -------------------------------
    from falsify.cli.run_vla_episode import _smoke_imports, _render_flythrough
    _smoke_imports(policy_backend="pi_gateway")

    from falsify.geometry import Point, yaw_to_quat_xyzw
    from falsify.io import build_frame_graph, load_yaml
    from falsify.orchestrator import EpisodeConfig, run_episode
    from falsify.policy import PiGatewayConfig, PiGatewayPolicy
    from falsify.sensors.camera import make_camera_sensor_from_yaml
    from falsify.sim import DroneState
    from falsify.sim.poses import camera_to_world_pose
    from falsify.sim.renderer import GSplatRenderer

    scene_yaml = _resolve(args.scene)
    scene_cfg = load_yaml(scene_yaml)
    scene_dir = scene_yaml.parent
    frame_cfg = load_yaml(_resolve(args.frame))
    fg = build_frame_graph(scene_cfg, base_path=scene_dir)

    # Renderer + forward-cam sensor built ONCE for all 20 rollouts (single scene).
    renderer = GSplatRenderer.from_scene_cfg(scene_cfg, scene_dir=scene_dir)
    fwd_sensor = None
    if not args.no_flythrough:
        fwd_sensor = make_camera_sensor_from_yaml(
            "forward", frame_cfg["cameras"]["forward"], fg,
            renderer=renderer.render, body_to_world=camera_to_world_pose,
        )

    cohort_specs = {
        "left":  dict(dataset=_resolve(args.left_dataset),
                      policy=_resolve(args.left_policy_config),
                      scene_key="center_gate_from_left"),
        "right": dict(dataset=_resolve(args.right_dataset),
                      policy=_resolve(args.right_policy_config),
                      scene_key="center_gate_from_right"),
    }

    index_rows: list[dict] = []
    t_total = time.time()

    for cohort in args.cohorts:
        spec = cohort_specs[cohort]
        dataset, policy_cfg_path, scene_key = (
            spec["dataset"], spec["policy"], spec["scene_key"])
        cohort_dir = out_dir / scene_key
        cohort_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[rollout-init] === cohort={cohort} "
              f"policy={policy_cfg_path.name} dataset={dataset.name} ===")

        for idx in episodes:
            ep_dir = cohort_dir / f"ep_{idx:03d}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            row = {"cohort": cohort, "episode": idx}
            try:
                states, task_index, gt_times = _read_episode_states(dataset, idx)
                prompt = _load_prompt(dataset, task_index)

                # ---- Initial state from the recorded first frame (MOCAP→NED).
                s0 = states[0]
                ned_pt = fg.convert(
                    Point.of(float(s0[0]), float(s0[1]), float(s0[2]),
                             fg.frame("mocap")),
                    to="ned",
                )
                quat = yaw_to_quat_xyzw(-float(s0[3]))   # state[3] == -yaw_ned
                start_state = DroneState(
                    pos=ned_pt, vel=np.zeros(3), quat_xyzw=quat, t=0.0,
                )

                # ---- Ground-truth NED trajectory from all recorded frames.
                gt_mocap = states[:, 0:3].astype(np.float64)
                gt_ned = np.asarray([
                    fg.convert(Point(p, frame=fg.frame("mocap")), to="ned").xyz
                    for p in gt_mocap
                ], dtype=np.float64)
                gt_yaw_ned = -states[:, 3].astype(np.float64)
                np.savez(
                    ep_dir / "gt_states.npz",
                    times=gt_times, positions_ned=gt_ned,
                    positions_mocap=gt_mocap, yaws_ned=gt_yaw_ned,
                )

                # ---- Policy + episode config.
                record_dir = ep_dir / "vla_io"
                pgcfg = PiGatewayConfig.from_yaml(
                    policy_cfg_path,
                    prompt_override=prompt,
                    execute_chunk_size_override=args.execute_chunk_size,
                    use_rtc_override=(False if args.no_rtc else None),
                    record_dir=record_dir,
                )
                effective_hz = args.hz or pgcfg.hz
                effective_chunk = 1 if pgcfg.use_rtc else pgcfg.execute_chunk_size

                def policy_factory(_goal_ned, _ec):
                    return PiGatewayPolicy(
                        pgcfg, build_frame_graph(scene_cfg, base_path=scene_dir))

                ec = EpisodeConfig(
                    scene_cfg=scene_cfg, frame_cfg=frame_cfg,
                    episode_cfg={
                        "hz": effective_hz,
                        "horizon_s": args.horizon_s,
                        "chunk_steps": effective_chunk,
                    },
                    scene_cfg_dir=scene_dir,
                )

                print(f"[run] {scene_key}/ep_{idx:03d}: "
                      f"start_ned={np.round(ned_pt.xyz, 3).tolist()} "
                      f"prompt={prompt!r}")
                t_ep = time.time()
                episode = run_episode(
                    ec,
                    policy_factory=policy_factory,
                    renderer=renderer,
                    initial_state_override=start_state,
                )
                dt = time.time() - t_ep

                # ---- Persist rollout trace (same schema as run_eval_campaign).
                traj = episode.trace.trajectory()
                quats = (traj.quaternions.astype(np.float64)
                         if traj.quaternions is not None
                         else np.tile([0., 0., 0., 1.], (len(traj.positions), 1)))
                failure_type = ("NONE" if episode.failure is None
                                else episode.failure.failure_type.name)
                np.savez(
                    ep_dir / "rollout_states.npz",
                    times=traj.times.astype(np.float64),
                    positions_ned=traj.positions.astype(np.float64),
                    quaternions_xyzw=quats,
                    failure_step=np.array(
                        -1 if episode.failure is None
                        else episode.failure.failure_step, dtype=np.int64),
                    failure_type=np.array(failure_type, dtype=object),
                )

                meta = {
                    "cohort": cohort,
                    "scene_key": scene_key,
                    "episode_index": idx,
                    "dataset": str(dataset),
                    "policy_config": str(policy_cfg_path),
                    "prompt": prompt,
                    "start_ned": ned_pt.xyz.tolist(),
                    "start_mocap": s0[0:3].tolist(),
                    "hz": effective_hz,
                    "actions_per_chunk": effective_chunk,
                    "horizon_s": args.horizon_s,
                    "use_rtc": pgcfg.use_rtc,
                    "n_states": len(episode.trace.states),
                    "n_chunks": len(episode.trace.policy_outputs),
                    "gt_n_frames": int(len(gt_ned)),
                    "failure": (None if episode.failure is None else {
                        "step": episode.failure.failure_step,
                        "type": failure_type,
                        "criterion": episode.failure.criterion_name,
                        "description": episode.failure.description,
                    }),
                    "goal_ned": (episode.goal.xyz.tolist()
                                 if episode.goal is not None else None),
                    "policy_traceability": pgcfg.traceability,
                    "elapsed_s": float(dt),
                }
                (ep_dir / "episode_meta.json").write_text(json.dumps(meta, indent=2))
                row.update(n_states=meta["n_states"],
                           stop=("HORIZON" if episode.failure is None
                                 else failure_type))

                # ---- Flythrough mp4.
                if fwd_sensor is not None and episode.trace.states:
                    fly = ep_dir / "flythrough.mp4"
                    _render_flythrough(
                        episode.trace.states, renderer, fwd_sensor.spec,
                        camera_to_world_pose, fly,
                        fps=args.flythrough_fps, every=args.flythrough_every,
                    )
                    row["flythrough"] = str(fly.relative_to(out_dir))

                # ---- Overlay HTML (rollout vs GT, in MOCAP).
                if not args.no_viz:
                    rollout_mocap = np.asarray(
                        fg.convert(traj, to="mocap").positions, dtype=np.float64)
                    overlay = ep_dir / "overlay.html"
                    _emit_overlay_html(
                        overlay, rollout_mocap, gt_mocap, scene_cfg,
                        title=f"{scene_key} / ep_{idx:03d} — closed-loop vs GT",
                        failure_type=failure_type,
                    )
                    row["overlay"] = str(overlay.relative_to(out_dir))

                print(f"  -> n_states={meta['n_states']}  stop={row['stop']}  "
                      f"elapsed={dt:.1f}s")
            except Exception as e:  # noqa: BLE001
                tb = traceback.format_exc()
                print(f"[error] {scene_key}/ep_{idx:03d}: {e}")
                (ep_dir / "error.txt").write_text(tb)
                row.update(outcome="ERROR", stop="ERROR")
            index_rows.append(row)

    index_path = _write_index(out_dir, index_rows)
    print(f"\n[rollout-init] done in {time.time() - t_total:.0f}s")
    print(f"[rollout-init] index → {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
