"""One-off: overlay an original real-world trajectory and a Gaussian-perturbed
copy of its state column over the right_gate scene point clouds, and write a
sibling parquet that is byte-identical to the source except for the state column.

Reads:
  data/gate_scenes_real_combined/data/chunk-000/episode_000098.parquet
    state column = [px_mocap, py_mocap, pz_mocap, -yaw_ned, 0, 0, 0]

Writes:
  runs/inspect/episode_000098_perturbed_overlay.html
  data/gate_scenes_real_combined/data/chunk-000/episode_000099.parquet
"""
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
PARQUET = REPO / "data/gate_scenes_real_combined/data/chunk-000/episode_000098.parquet"
PARQUET_OUT = REPO / "data/gate_scenes_real_combined/data/chunk-000/episode_000099.parquet"
SCENE_YAML = REPO / "configs/scenes/right_gate.yaml"
OUT = REPO / "runs/inspect/episode_000098_perturbed_overlay.html"

POS_SIGMA = 0.02   # m, per-axis 1-sigma Gaussian on (x, y, z)
YAW_SIGMA = 0.01   # rad, 1-sigma on yaw
SEED = 0


def main():
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from falsify.io import build_frame_graph, load_yaml
    from falsify.visualization import read_ply, subsample

    scene_cfg = load_yaml(SCENE_YAML)
    fg = build_frame_graph(scene_cfg, base_path=SCENE_YAML.parent)
    mocap = fg.frame("mocap")

    state = np.stack(pq.read_table(PARQUET, columns=["state"])["state"].to_numpy())
    pos = state[:, :3].astype(np.float64)             # (T, 3) mocap
    yaw_negated = state[:, 3].astype(np.float64)      # = -yaw_ned

    rng = np.random.default_rng(SEED)
    pos_perturbed = pos + rng.normal(scale=POS_SIGMA, size=pos.shape)
    yaw_perturbed = yaw_negated + rng.normal(scale=YAW_SIGMA, size=yaw_negated.shape)

    print(f"[trajectory] T={pos.shape[0]} frames, pos_sigma={POS_SIGMA} m, yaw_sigma={YAW_SIGMA} rad")
    print(f"[trajectory] mean |dp| = {np.linalg.norm(pos_perturbed - pos, axis=1).mean():.4f} m")

    traces = []

    # Scene PLYs (gate + table) in mocap, with their declared tint.
    for entry in scene_cfg.get("scene_objects", []) or []:
        ply = Path(entry["ply"])
        if not ply.is_absolute():
            ply = (SCENE_YAML.parent / ply).resolve()
        cloud = read_ply(ply, mocap)
        cloud = subsample(cloud, 6000)
        r, g, b = [int(255 * c) for c in entry.get("color", (0.5, 0.5, 0.5))]
        traces.append(go.Scatter3d(
            x=cloud.points[:, 0], y=cloud.points[:, 1], z=cloud.points[:, 2],
            mode="markers",
            marker=dict(size=2, color=f"rgb({r},{g},{b})", opacity=0.55),
            name=f"{entry['name']} (scene)",
        ))

    # Full-scene sparse cloud for context (auto-discovered like the inspector).
    data_cwd = Path(scene_cfg["gsplat_data_cwd"])
    if not data_cwd.is_absolute():
        data_cwd = (SCENE_YAML.parent / data_cwd).resolve()
    sparse = data_cwd / "mocap_processed" / "sparse_pc.ply"
    if sparse.exists():
        cloud = read_ply(sparse, mocap)
        cloud = subsample(cloud, 20000)
        if cloud.colors is not None:
            cols = np.clip(cloud.colors * 255.0, 0, 255).astype(int)
            color_arr = [f"rgb({r},{g},{b})" for r, g, b in cols]
        else:
            color_arr = "rgb(180,180,180)"
        traces.append(go.Scatter3d(
            x=cloud.points[:, 0], y=cloud.points[:, 1], z=cloud.points[:, 2],
            mode="markers",
            marker=dict(size=1.2, color=color_arr, opacity=0.35),
            name="full_scene (sparse SfM)",
        ))

    # Original trajectory (blue line + small markers).
    traces.append(go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
        mode="lines+markers",
        line=dict(color="rgb(40, 90, 220)", width=4),
        marker=dict(size=2.5, color="rgb(40, 90, 220)"),
        name=f"original (ep 98)",
        hovertemplate="orig<br>t=%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
        text=[f"{i}" for i in range(pos.shape[0])],
    ))

    # Perturbed trajectory (red line + small markers).
    traces.append(go.Scatter3d(
        x=pos_perturbed[:, 0], y=pos_perturbed[:, 1], z=pos_perturbed[:, 2],
        mode="lines+markers",
        line=dict(color="rgb(220, 60, 60)", width=3),
        marker=dict(size=2.5, color="rgb(220, 60, 60)"),
        name=f"perturbed σ_p={POS_SIGMA} m σ_yaw={YAW_SIGMA} rad",
        hovertemplate="pert<br>t=%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
        text=[f"{i}" for i in range(pos.shape[0])],
    ))

    # Start / end markers.
    for label, idx, color in [("start", 0, "rgb(0, 180, 0)"), ("end", -1, "rgb(0, 0, 0)")]:
        p = pos[idx]
        traces.append(go.Scatter3d(
            x=[p[0]], y=[p[1]], z=[p[2]],
            mode="markers+text",
            marker=dict(size=6, color=color, symbol="diamond"),
            text=[label], textposition="top center",
            name=label, showlegend=False,
        ))

    fig = go.Figure(traces)
    fig.update_layout(
        title=(f"episode_000098 (right_gate) — original vs Gaussian-perturbed state<br>"
               f"<sub>σ_position = {POS_SIGMA} m per-axis, σ_yaw = {YAW_SIGMA} rad, seed = {SEED}</sub>"),
        scene=dict(
            xaxis_title="x_mocap (m)",
            yaxis_title="y_mocap (m)",
            zaxis_title="z_mocap (m)",
            aspectmode="data",
        ),
        legend=dict(orientation="v", x=0.02, y=0.98),
        margin=dict(l=0, r=0, t=60, b=0),
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(OUT)
    print(f"[done] wrote {OUT}")

    # Rebuild the new state column exactly matching the source schema: a
    # FixedSizeList<float32, 7> per row, only dims 0..3 perturbed.
    full = pq.read_table(PARQUET)
    state_arr = np.stack(full["state"].to_numpy()).astype(np.float32, copy=True)
    state_arr[:, 0:3] = pos_perturbed.astype(np.float32)
    state_arr[:, 3] = yaw_perturbed.astype(np.float32)
    # Columns 4..6 left as the zero pad expected by the VLA payload.

    source_field = full.schema.field("state")
    new_state = pa.array(list(state_arr), type=source_field.type)
    new_idx = full.schema.get_field_index("state")
    new_table = full.set_column(new_idx, source_field, new_state)

    # Promote this row group from a copy of ep 98 to a registered ep 99:
    # bump episode_index, shift the global `index` to slot in after the
    # existing total_frames=25103. `frame_index` stays [0..321].
    new_episode_index = 99
    new_global_start = 25103   # current dataset total_frames
    new_table = new_table.set_column(
        new_table.schema.get_field_index("episode_index"),
        new_table.schema.field("episode_index"),
        pa.array(np.full(new_table.num_rows, new_episode_index, dtype=np.int64)),
    )
    new_global_index = np.arange(new_global_start, new_global_start + new_table.num_rows, dtype=np.int64)
    new_table = new_table.set_column(
        new_table.schema.get_field_index("index"),
        new_table.schema.field("index"),
        pa.array(new_global_index),
    )

    PARQUET_OUT.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(new_table, PARQUET_OUT)
    print(f"[done] wrote {PARQUET_OUT}")
    print(f"        rows={new_table.num_rows} schema matches source: "
          f"{new_table.schema.equals(full.schema)}")

    # ---- Update meta files ----
    import json
    meta_dir = REPO / "data/gate_scenes_real_combined/meta"

    # episodes.jsonl — append.
    new_ep_record = {
        "episode_index": new_episode_index,
        "tasks": ["go through the gate on the right and hover over the stuffed animal"],
        "length": int(new_table.num_rows),
    }
    with open(meta_dir / "episodes.jsonl", "a") as f:
        f.write(json.dumps(new_ep_record) + "\n")
    print(f"[meta] appended episodes.jsonl: {new_ep_record}")

    # episodes_stats.jsonl — clone ep 98's stats line, then override the
    # fields that genuinely changed (episode_index column, index column,
    # state column). Images / actions / timestamp / frame_index / task_index
    # are byte-identical to ep 98 so their stats carry over.
    ep98_stats = None
    with open(meta_dir / "episodes_stats.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["episode_index"] == 98:
                ep98_stats = r
                break
    assert ep98_stats is not None, "ep98 stats row missing"
    s = json.loads(json.dumps(ep98_stats))  # deep copy
    s["episode_index"] = new_episode_index
    n = int(new_table.num_rows)
    s["stats"]["episode_index"] = {
        "min": [float(new_episode_index)], "max": [float(new_episode_index)],
        "mean": [float(new_episode_index)], "std": [0.0], "count": [n],
    }
    s["stats"]["index"] = {
        "min": [float(new_global_index.min())],
        "max": [float(new_global_index.max())],
        "mean": [float(new_global_index.mean())],
        "std": [float(new_global_index.std())],
        "count": [n],
    }
    s["stats"]["state"] = {
        "min": state_arr.min(axis=0).astype(float).tolist(),
        "max": state_arr.max(axis=0).astype(float).tolist(),
        "mean": state_arr.mean(axis=0).astype(float).tolist(),
        "std": state_arr.std(axis=0).astype(float).tolist(),
        "count": [n],
    }
    with open(meta_dir / "episodes_stats.jsonl", "a") as f:
        f.write(json.dumps(s) + "\n")
    print(f"[meta] appended episodes_stats.jsonl for ep {new_episode_index}")

    # info.json — bump totals + split.
    info_path = meta_dir / "info.json"
    info = json.loads(info_path.read_text())
    info["total_episodes"] = int(info["total_episodes"]) + 1
    info["total_frames"] = int(info["total_frames"]) + n
    info["splits"]["train"] = f"0:{info['total_episodes']}"
    info_path.write_text(json.dumps(info, indent=4) + "\n")
    print(f"[meta] info.json now total_episodes={info['total_episodes']}, "
          f"total_frames={info['total_frames']}, splits.train={info['splits']['train']}")


if __name__ == "__main__":
    main()
