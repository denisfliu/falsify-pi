"""Build a single HTML page comparing the I/O of a training-dataset
episode against one eval-pipeline trial.

Panels:

  1. Forward / wrist images at frame 0 — training (synth) vs eval (live)
  2. Starting-state vector (mocap) — training vs eval, side by side
  3. Action chunk at query_0000 (training first 25 actions vs eval first
     25 actions emitted by the policy)
  4. Full-episode trajectory in MOCAP — training "ground truth" path
     vs the eval rollout. Plotted on the same scene context (gate +
     table + goal box) used by the rest of the campaign-compare plots.

Usage:

    PYTHONPATH=src python scripts/compare_training_vs_eval.py \\
        --training data/atomic_datasets/synth_center_from_right/data/chunk-000/episode_000003.parquet \\
        --eval-trial runs/eval_campaigns/<campaign>/<scene_key>/trial_NNN \\
        --scene configs/scenes/center_gate.yaml \\
        --out runs/eval_campaigns/training_vs_eval.html
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _ned_to_mocap_arr(scene_yaml: Path, positions_ned: np.ndarray) -> np.ndarray:
    from falsify.geometry import Point
    from falsify.io import build_frame_graph, load_yaml
    scene_cfg = load_yaml(scene_yaml)
    fg = build_frame_graph(scene_cfg, base_path=scene_yaml.parent)
    ned_frame = fg.frame("ned")
    return np.asarray([
        fg.convert(Point(np.asarray(p, dtype=np.float64), frame=ned_frame), to="mocap").xyz
        for p in positions_ned
    ], dtype=np.float64)


def _read_training_episode(parquet_path: Path) -> dict:
    import pyarrow.parquet as pq

    df = pq.read_table(parquet_path).to_pandas()
    states = np.stack([np.asarray(s) for s in df["state"].values])
    actions = np.stack([np.asarray(a) for a in df["actions"].values])
    timestamps = np.asarray(df["timestamp"].values, dtype=float)

    # Decode the frame-0 images.
    f0 = df.iloc[0]
    images = {}
    for col in ("image", "wrist_image", "3pov_1"):
        if col in df.columns and f0.get(col) is not None:
            v = f0[col]
            # `v` is a dict-like with 'bytes'/'path'.
            if hasattr(v, "as_py"):
                v = v.as_py()
            if isinstance(v, dict) and "bytes" in v:
                images[col] = base64.b64encode(v["bytes"]).decode("ascii")
    return {
        "states_mocap": states,
        "actions": actions,
        "timestamps": timestamps,
        "images_b64": images,
        "n_frames": len(df),
    }


def _read_eval_trial(trial_dir: Path) -> dict:
    summary = json.loads((trial_dir / "episode_summary.json").read_text())
    card_path = trial_dir / "trial_card.json"
    card = json.loads(card_path.read_text()) if card_path.is_file() else {}

    rollout = np.load(trial_dir / "rollout_states.npz", allow_pickle=True)
    positions_ned = np.asarray(rollout["positions_ned"])
    times = np.asarray(rollout["times"])

    # First query's snapshot.
    q0_dir = trial_dir / "vla_io" / "query_0000_step_00000"
    q0_data = json.loads((q0_dir / "data.json").read_text())
    q0_actions = np.load(q0_dir / "actions.npy")
    q0_waypoints_ned = np.load(q0_dir / "waypoints_ned.npy")
    images = {}
    for name in (
        "rgb_forward.png", "rgb_downward.png",
        # When the policy YAML sets `image_size` / `channel_order`, the
        # debug recorder also dumps the post-preprocess image actually
        # handed to the gateway. PIL doesn't know about BGR — these
        # PNGs hold BGR bytes labeled RGB, exactly matching the
        # training PNG bytes on disk.
        "sent_forward.png", "sent_downward.png",
    ):
        p = q0_dir / name
        if p.is_file():
            images[name] = _b64(p)
    return {
        "summary": summary,
        "card": card,
        "positions_ned": positions_ned,
        "times": times,
        "q0_data": q0_data,
        "q0_actions": q0_actions,
        "q0_waypoints_ned": q0_waypoints_ned,
        "images_b64": images,
    }


def _gate_aabb_wireframe(aabb_min: np.ndarray, aabb_max: np.ndarray):
    p = np.array([
        [aabb_min[0], aabb_min[1], aabb_min[2]],
        [aabb_max[0], aabb_min[1], aabb_min[2]],
        [aabb_max[0], aabb_max[1], aabb_min[2]],
        [aabb_min[0], aabb_max[1], aabb_min[2]],
        [aabb_min[0], aabb_min[1], aabb_max[2]],
        [aabb_max[0], aabb_min[1], aabb_max[2]],
        [aabb_max[0], aabb_max[1], aabb_max[2]],
        [aabb_min[0], aabb_max[1], aabb_max[2]],
    ])
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    xs, ys, zs = [], [], []
    for a, b in edges:
        xs.extend([p[a, 0], p[b, 0], None])
        ys.extend([p[a, 1], p[b, 1], None])
        zs.extend([p[a, 2], p[b, 2], None])
    return xs, ys, zs


def _box_wireframe(center: np.ndarray, half_extents: np.ndarray):
    return _gate_aabb_wireframe(center - half_extents, center + half_extents)


def _build_scene_traces(scene_yaml: Path, max_cloud_points: int = 4000):
    """Gate + table clouds (scene_edits applied), gate AABB, goal box."""
    from falsify.geometry import PointCloud
    from falsify.io import load_yaml, build_frame_graph
    from falsify.sim.scene_edits import apply_edits_to_scene_object, load_scene_edits
    from falsify.visualization import read_ply, subsample
    import plotly.graph_objects as go

    scene_cfg = load_yaml(scene_yaml)
    scene_dir = scene_yaml.parent
    fg = build_frame_graph(scene_cfg, base_path=scene_dir)
    edits = load_scene_edits(scene_cfg)
    traces = []

    region = scene_cfg.get("gate_region") or {}
    if region:
        aabb_min = np.asarray(region["aabb_min"], dtype=np.float64)
        aabb_max = np.asarray(region["aabb_max"], dtype=np.float64)
        xs, ys, zs = _gate_aabb_wireframe(aabb_min, aabb_max)
        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(color="rgba(20,80,120,0.7)", width=3),
            name="gate AABB", legendgroup="scene", hoverinfo="skip",
        ))

    for entry in scene_cfg.get("scene_objects") or []:
        ply_path = scene_dir / entry["ply"]
        if not ply_path.is_file():
            continue
        cloud = read_ply(ply_path, fg.frame(entry["frame"]))
        cloud = subsample(cloud, max_cloud_points)
        if edits:
            new_pts = apply_edits_to_scene_object(entry["name"], cloud.points, edits, fg)
            new_colors = cloud.colors
            if new_colors is not None and new_pts.shape[0] != cloud.points.shape[0]:
                reps = new_pts.shape[0] // cloud.points.shape[0]
                new_colors = np.tile(cloud.colors, (reps, 1))
            cloud = PointCloud(points=new_pts, frame=cloud.frame, colors=new_colors)
        pts = np.asarray(cloud.points, dtype=np.float64)
        colour = entry.get("color", (0.5, 0.5, 0.5))
        rgb = f"rgb({int(255*colour[0])},{int(255*colour[1])},{int(255*colour[2])})"
        traces.append(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=dict(size=1.2, color=rgb, opacity=0.5),
            name=entry["name"], legendgroup="scene", hoverinfo="skip",
        ))

    goal = scene_cfg.get("goal_position_mocap")
    if goal:
        g = np.asarray(goal, dtype=np.float64)
        traces.append(go.Scatter3d(
            x=[g[0]], y=[g[1]], z=[g[2]],
            mode="markers+text",
            marker=dict(size=6, color="rgba(50,200,50,0.95)", symbol="diamond"),
            text=["goal"], textposition="top center",
            name="goal", legendgroup="scene",
        ))
        # Goal-tolerance box from the matching safety yaml.
        scene_key = scene_cfg.get("scene_key") or scene_yaml.stem
        safety_path = REPO_ROOT / "configs" / "safety" / f"{scene_key}.yaml"
        if safety_path.is_file():
            safety = load_yaml(safety_path) or {}
            miss = safety.get("miss_gate") or {}
            half_extents = miss.get("goal_tolerance_half_extents")
            if half_extents is not None:
                xs, ys, zs = _box_wireframe(g, np.asarray(half_extents, dtype=np.float64))
                traces.append(go.Scatter3d(
                    x=xs, y=ys, z=zs, mode="lines",
                    line=dict(color="rgb(10,90,30)", width=4, dash="dot"),
                    name=f"goal box (half={half_extents})",
                    legendgroup="scene", hoverinfo="skip",
                ))
    return traces


# ---- HTML emit ---------------------------------------------------------

def _state_table(label: str, state: np.ndarray) -> str:
    """7-vector table — first 3 are pos, 4th yaw, remainder zero-pad."""
    components = ["x (m)", "y (m)", "z (m)", "yaw (rad)", "[4]", "[5]", "[6]"]
    rows = "".join(
        f"<tr><td>{name}</td><td>{state[i]:+0.4f}</td></tr>"
        for i, name in enumerate(components)
    )
    return f"""
    <h4>{label}</h4>
    <table class="state">{rows}</table>
    """


def _actions_table(label: str, actions: np.ndarray, n: int = 5) -> str:
    """Show the first n action rows."""
    header = "<th>step</th>" + "".join(f"<th>a{j}</th>" for j in range(actions.shape[1]))
    body = ""
    for i in range(min(n, actions.shape[0])):
        cells = "".join(f"<td>{actions[i,j]:+0.4f}</td>" for j in range(actions.shape[1]))
        body += f"<tr><td>{i}</td>{cells}</tr>"
    return f"""
    <h4>{label} (first {n} of {actions.shape[0]})</h4>
    <table class="actions"><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>
    """


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--training", required=True, type=Path,
                    help="Training-dataset parquet (one episode).")
    ap.add_argument("--eval-trial", required=True, type=Path,
                    help="Eval campaign trial dir.")
    ap.add_argument("--scene", required=True, type=Path,
                    help="Scene YAML — same one the eval trial used.")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-cloud-points", type=int, default=4000)
    args = ap.parse_args(argv)

    import plotly.graph_objects as go
    import plotly.io as pio

    training = _read_training_episode(args.training)
    evalrun = _read_eval_trial(args.eval_trial)
    scene_traces = _build_scene_traces(args.scene, args.max_cloud_points)

    # ---- 3-D trajectory plot ----
    train_pos_mocap = training["states_mocap"][:, :3]
    eval_pos_mocap = _ned_to_mocap_arr(args.scene, evalrun["positions_ned"])
    waypoints_mocap = _ned_to_mocap_arr(args.scene, evalrun["q0_waypoints_ned"])

    fig = go.Figure()
    for tr in scene_traces:
        fig.add_trace(tr)
    fig.add_trace(go.Scatter3d(
        x=train_pos_mocap[:, 0], y=train_pos_mocap[:, 1], z=train_pos_mocap[:, 2],
        mode="lines",
        line=dict(color="rgb( 30, 120, 220)", width=4),
        name="training trajectory (mocap state)",
        hovertemplate="train step %{pointNumber}<br>(%{x:.2f},%{y:.2f},%{z:.2f})<extra></extra>",
    ))
    fig.add_trace(go.Scatter3d(
        x=eval_pos_mocap[:, 0], y=eval_pos_mocap[:, 1], z=eval_pos_mocap[:, 2],
        mode="lines",
        line=dict(color="rgb(220, 100,  30)", width=4),
        name="eval rollout (executed)",
        hovertemplate="eval step %{pointNumber}<br>(%{x:.2f},%{y:.2f},%{z:.2f})<extra></extra>",
    ))
    fig.add_trace(go.Scatter3d(
        x=waypoints_mocap[:, 0], y=waypoints_mocap[:, 1], z=waypoints_mocap[:, 2],
        mode="lines+markers",
        line=dict(color="rgba(220, 100, 30, 0.5)", width=2, dash="dash"),
        marker=dict(size=3, color="rgba(220, 100, 30, 0.7)"),
        name="eval query_0 waypoints (policy output before execution)",
        hovertemplate="wpt %{pointNumber}<br>(%{x:.2f},%{y:.2f},%{z:.2f})<extra></extra>",
    ))
    # Start markers.
    fig.add_trace(go.Scatter3d(
        x=[train_pos_mocap[0, 0]], y=[train_pos_mocap[0, 1]], z=[train_pos_mocap[0, 2]],
        mode="markers",
        marker=dict(size=6, color="rgb( 30, 120, 220)", symbol="circle",
                    line=dict(width=1, color="black")),
        name="train start", showlegend=False,
    ))
    fig.add_trace(go.Scatter3d(
        x=[eval_pos_mocap[0, 0]], y=[eval_pos_mocap[0, 1]], z=[eval_pos_mocap[0, 2]],
        mode="markers",
        marker=dict(size=6, color="rgb(220, 100,  30)", symbol="circle",
                    line=dict(width=1, color="black")),
        name="eval start", showlegend=False,
    ))
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="mocap x (m)"),
            yaxis=dict(title="mocap y (m)"),
            zaxis=dict(title="mocap z (m, up)"),
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.0)),
        ),
        margin=dict(l=0, r=0, t=20, b=0),
        height=620,
        legend=dict(itemsizing="constant"),
    )

    # ---- HTML payload ----
    train_state_t0 = training["states_mocap"][0]
    eval_state_t0 = np.asarray(evalrun["q0_data"]["state_vec_to_vla"], dtype=float)
    prompt_train = "(from tasks.jsonl: see prompt)"
    prompt_eval = evalrun["q0_data"]["prompt"]
    summary = evalrun["summary"]

    # Embedded images. Training has `image` (forward) and `wrist_image`;
    # eval has `rgb_forward.png` and `rgb_downward.png`.
    def _img(html_id: str, b64: Optional[str], label: str) -> str:
        if not b64:
            return f"<div class='img-cell'><div class='img-missing'>{label} (missing)</div></div>"
        return f"<div class='img-cell'><img id='{html_id}' src='data:image/png;base64,{b64}'/><div class='img-label'>{label}</div></div>"

    has_sent = (
        evalrun["images_b64"].get("sent_forward.png") is not None
        or evalrun["images_b64"].get("sent_downward.png") is not None
    )
    sent_note = (
        ""
        if has_sent else
        "<p class='note'>(no `sent_*.png` in vla_io — this trial pre-dates "
        "the image_size/channel_order knobs; only native renders were recorded.)</p>"
    )
    img_html = f"""
    <div class="img-row">
      <h3 class="col">Training (synth)<br/><small>parquet frame 0, channel_order=BGR per embodiment</small></h3>
      <h3 class="col">Eval — native render<br/><small>rgb_<i>cam</i>.png, what GSplatRenderer produced</small></h3>
      <h3 class="col">Eval — sent to server<br/><small>sent_<i>cam</i>.png, post image_size/channel_order</small></h3>
    </div>
    <div class="img-row">
      {_img('t_fwd', training['images_b64'].get('image'), 'image (training)')}
      {_img('e_fwd_native', evalrun['images_b64'].get('rgb_forward.png'), 'rgb_forward.png')}
      {_img('e_fwd_sent',   evalrun['images_b64'].get('sent_forward.png'), 'sent_forward.png')}
    </div>
    <div class="img-row">
      {_img('t_wrist', training['images_b64'].get('wrist_image'), 'wrist_image (training)')}
      {_img('e_down_native', evalrun['images_b64'].get('rgb_downward.png'), 'rgb_downward.png')}
      {_img('e_down_sent',   evalrun['images_b64'].get('sent_downward.png'), 'sent_downward.png')}
    </div>
    {sent_note}
    """

    state_html = f"""
    <div class="state-row">
      {_state_table('training state[0] (mocap)', train_state_t0)}
      {_state_table('eval state_vec_to_vla[0]', eval_state_t0)}
    </div>
    <p class="note">
      Δ mocap pos (eval − train) =
      {(eval_state_t0[:3] - train_state_t0[:3]).round(4).tolist()} m;
      Δ yaw = {float(eval_state_t0[3] - train_state_t0[3]):+0.4f} rad
    </p>
    """

    actions_html = f"""
    <div class="state-row">
      {_actions_table('training actions[0:5]', training['actions'])}
      {_actions_table('eval query_0 actions[0:5]', evalrun['q0_actions'])}
    </div>
    """

    info_html = f"""
    <p>
      <b>Training episode:</b> {args.training}<br>
      training prompt: <i>{open(args.training.parent.parent.parent / 'meta' / 'tasks.jsonl').readline().strip()}</i><br>
      n_frames = {training['n_frames']}, fps from info.json = 10<br><br>
      <b>Eval trial:</b> {args.eval_trial}<br>
      eval prompt: <i>{prompt_eval}</i><br>
      posthoc_outcome = <b>{summary.get('posthoc_outcome')}</b>
      • transited = {summary.get('transited')}
      • correct_crossings = {summary.get('correct_crossings')}
      • wrong_crossings = {summary.get('wrong_crossings')}<br>
      effective_chunk_size = {evalrun['q0_data']['actions_shape'][0]},
      use_rtc = {evalrun['q0_data']['use_rtc']},
      hz (eval) = 30, hz (training) = 10
    </p>
    """

    plot_html = pio.to_html(fig, include_plotlyjs="cdn", full_html=False)

    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>training vs eval</title>
<style>
  body {{ font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
          margin: 12px 24px; color: #222; }}
  h1 {{ font-size: 1.4em; margin: 6px 0; }}
  h2 {{ font-size: 1.1em; margin: 14px 0 4px; }}
  h3 {{ font-size: 1.0em; margin: 6px 0; }}
  h4 {{ font-size: 0.95em; margin: 4px 0; }}
  .img-row {{ display: flex; gap: 8px; align-items: flex-start; margin-bottom: 4px; }}
  .img-row .col {{ flex: 1; text-align: center; }}
  .img-row .col small {{ color: #555; font-weight: normal; }}
  .img-cell {{ flex: 1; max-width: 33%; text-align: center; }}
  .img-cell img {{ max-width: 100%; max-height: 320px; border: 1px solid #ccc; }}
  .img-label {{ font-size: 0.8em; color: #555; margin-top: 2px; }}
  .img-missing {{ height: 180px; display: flex; align-items: center;
                  justify-content: center; border: 1px dashed #aaa; color: #888; }}
  table.state, table.actions {{ border-collapse: collapse; font-family: ui-monospace, monospace; font-size: 0.85em; }}
  table.state td, table.actions td, table.actions th {{
    border: 1px solid #ddd; padding: 2px 6px; }}
  table.actions th {{ background: #f4f4f4; }}
  .state-row {{ display: flex; gap: 24px; }}
  .note {{ font-family: ui-monospace, monospace; font-size: 0.85em; color: #555; }}
</style></head>
<body>
  <h1>Training (synth) vs. Eval pipeline — center_gate_from_right</h1>
  {info_html}
  <h2>1. Frame-0 observation images</h2>
  {img_html}
  <h2>2. Starting state (7-vector to VLA)</h2>
  {state_html}
  <h2>3. First action chunk</h2>
  {actions_html}
  <h2>4. Full trajectories (MOCAP)</h2>
  {plot_html}
</body></html>"""

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page)
    print(f"[plot] wrote {args.out}  ({args.out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
