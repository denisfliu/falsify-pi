"""Browse LeRobot training datasets under data/ (GUI venv: pyarrow + numpy).

Handles both shipped layouts:
- v2.1 (data/atomic_datasets/*): meta/{info.json, episodes.jsonl, tasks.jsonl},
  data/chunk-NNN/episode_NNNNNN.parquet, columns image/wrist_image/state/actions
- v3.0 (data/no_3pov_v3/*): meta/{info.json, tasks.parquet},
  data/chunk-NNN/episode-NNNNNN.parquet, columns observation.images.*/
  observation.state/action

Everything is driven by meta/info.json (`data_path` format string + `features`
dict), so other conforming datasets work too.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np

from ..paths import PLOTS_CACHE, REPO_ROOT

DATA_ROOT = REPO_ROOT / "data"

# image columns whose first frame is smaller than this are placeholders
# (e.g. the 270-byte 3pov_1 stubs) and are hidden from the frame viewer
_PLACEHOLDER_BYTES = 1000

_TTL_S = 10.0
_cache: dict[str, tuple[float, object]] = {}


class PathOutsideData(Exception):
    pass


def _resolve(rel: str) -> Path:
    p = (REPO_ROOT / rel).resolve()
    if not p.is_relative_to(DATA_ROOT.resolve()):
        raise PathOutsideData(rel)
    return p


def _rel(p: Path) -> str:
    # data/ is a symlink; present paths as repo-relative through the link
    return "data/" + str(p.resolve().relative_to(DATA_ROOT.resolve()))


def _cached(key: str, fn):
    now = time.monotonic()
    hit = _cache.get(key)
    if hit and now - hit[0] < _TTL_S:
        return hit[1]
    val = fn()
    _cache[key] = (now, val)
    return val


def _read_info(ds: Path) -> dict | None:
    try:
        return json.loads((ds / "meta" / "info.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _read_tasks(ds: Path) -> dict[int, str]:
    """task_index -> task string, from tasks.jsonl (v2.1) or tasks.parquet (v3)."""
    jsonl = ds / "meta" / "tasks.jsonl"
    if jsonl.exists():
        out = {}
        for line in jsonl.read_text().splitlines():
            try:
                d = json.loads(line)
                out[int(d["task_index"])] = d["task"]
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
        return out
    pq_path = ds / "meta" / "tasks.parquet"
    if pq_path.exists():
        import pyarrow.parquet as pq
        rows = pq.read_table(pq_path).to_pylist()
        out = {}
        for i, r in enumerate(rows):
            task = r.get("task") or r.get("__index_level_0__")
            idx = r.get("task_index", i)
            if task:
                out[int(idx)] = str(task)
        return out
    return {}


def dataset_index() -> list[dict]:
    """Datasets = any dir at data/* or data/*/* with meta/info.json."""
    def build():
        found = []
        if not DATA_ROOT.exists():
            return found
        candidates = []
        for child in sorted(DATA_ROOT.iterdir()):
            if not child.is_dir():
                continue
            if (child / "meta" / "info.json").exists():
                candidates.append(child)
            else:
                candidates.extend(
                    g for g in sorted(child.iterdir())
                    if g.is_dir() and (g / "meta" / "info.json").exists())
        for ds in candidates:
            info = _read_info(ds)
            if info is None:
                continue
            tasks = _read_tasks(ds)
            image_cols = [k for k, v in (info.get("features") or {}).items()
                          if isinstance(v, dict) and v.get("dtype") == "image"]
            if not image_cols:
                # empty features dict — sniff the first parquet's footer
                first_pq = next(iter(ds.glob("data/chunk-*/episode*.parquet")), None)
                if first_pq is not None:
                    import pyarrow.parquet as pq
                    try:
                        image_cols = _image_cols_from_schema(pq.read_schema(first_pq))
                    except OSError:
                        pass
            found.append({
                "name": ds.name,
                "group": ds.parent.name if ds.parent != DATA_ROOT else "data",
                "path": _rel(ds),
                "version": info.get("codebase_version"),
                "total_episodes": info.get("total_episodes"),
                "total_frames": info.get("total_frames"),
                "fps": info.get("fps"),
                "robot_type": info.get("robot_type"),
                "tasks": list(tasks.values()),
                "image_columns": image_cols,
            })
        return found
    return _cached("datasets", build)


def episode_list(ds_rel: str) -> list[dict]:
    ds = _resolve(ds_rel)
    jsonl = ds / "meta" / "episodes.jsonl"
    if jsonl.exists():
        out = []
        for line in jsonl.read_text().splitlines():
            try:
                d = json.loads(line)
                out.append({"episode_index": d["episode_index"],
                            "length": d.get("length"),
                            "tasks": d.get("tasks")})
            except (json.JSONDecodeError, KeyError):
                continue
        return out
    # v3: derive from the data files; length from the parquet footer (cheap)
    import pyarrow.parquet as pq
    out = []
    for p in sorted(ds.glob("data/chunk-*/episode*.parquet")):
        m = re.search(r"episode[-_](\d+)\.parquet$", p.name)
        if not m:
            continue
        try:
            n = pq.read_metadata(p).num_rows
        except OSError:
            n = None
        out.append({"episode_index": int(m.group(1)), "length": n, "tasks": None})
    out.sort(key=lambda d: d["episode_index"])
    return out


def _episode_parquet(ds: Path, info: dict, episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size") or 1000)
    fmt = info.get("data_path") or "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    rel = fmt.format(episode_chunk=episode_index // chunks_size,
                     chunk_index=episode_index // chunks_size,
                     episode_index=episode_index)
    p = ds / rel
    if not p.exists():
        raise FileNotFoundError(rel)
    return p


_table_cache: dict[str, tuple[float, object]] = {}


def _load_table(p: Path):
    """One-entry table cache — the frame scrubber hits the same episode
    repeatedly and the tables are ~50 MB."""
    import pyarrow.parquet as pq
    key = str(p)
    mtime = p.stat().st_mtime_ns
    hit = _table_cache.get(key)
    if hit and hit[0] == mtime:
        return hit[1]
    table = pq.read_table(p)
    _table_cache.clear()
    _table_cache[key] = (mtime, table)
    return table


def _columns(info: dict, table) -> dict:
    feats = info.get("features") or {}
    image_cols = [k for k, v in feats.items()
                  if isinstance(v, dict) and v.get("dtype") == "image"
                  and k in table.column_names]
    if not image_cols:
        # some datasets (e.g. dagger-1_*) ship an empty features dict —
        # detect HF Image columns from the arrow schema (struct{bytes, path})
        image_cols = _image_cols_from_schema(table.schema)
    state_col = next((c for c in table.column_names
                      if c == "state" or c.endswith(".state")), None)
    action_col = next((c for c in table.column_names
                       if c in ("actions", "action")), None)
    return {"images": image_cols, "state": state_col, "action": action_col}


def _image_cols_from_schema(schema) -> list[str]:
    import pyarrow as pa
    out = []
    for field in schema:
        t = field.type
        if pa.types.is_struct(t) and {f.name for f in t} >= {"bytes", "path"}:
            out.append(field.name)
    return out


def episode_detail(ds_rel: str, episode_index: int) -> dict:
    ds = _resolve(ds_rel)
    info = _read_info(ds) or {}
    table = _load_table(_episode_parquet(ds, info, episode_index))
    cols = _columns(info, table)
    first = table.slice(0, 1).to_pylist()[0]
    cameras = []
    for c in cols["images"]:
        blob = (first.get(c) or {}).get("bytes") or b""
        cameras.append({"column": c, "first_frame_bytes": len(blob),
                        "placeholder": len(blob) < _PLACEHOLDER_BYTES})
    task = None
    tasks = _read_tasks(ds)
    if "task_index" in table.column_names:
        task = tasks.get(int(first.get("task_index", 0)))
    return {
        "dataset": ds_rel,
        "episode_index": episode_index,
        "n_frames": table.num_rows,
        "fps": info.get("fps"),
        "task": task,
        "cameras": cameras,
        "state_dim": len(first.get(cols["state"]) or []) if cols["state"] else 0,
        "action_dim": len(first.get(cols["action"]) or []) if cols["action"] else 0,
    }


def frame_png(ds_rel: str, episode_index: int, frame: int, camera: str) -> bytes:
    ds = _resolve(ds_rel)
    info = _read_info(ds) or {}
    table = _load_table(_episode_parquet(ds, info, episode_index))
    if camera not in table.column_names:
        raise KeyError(camera)
    frame = max(0, min(frame, table.num_rows - 1))
    cell = table.column(camera).slice(frame, 1).to_pylist()[0] or {}
    blob = cell.get("bytes")
    if not blob:
        raise FileNotFoundError(f"no image bytes at frame {frame}")
    return blob


def episode_plot(ds_rel: str, episode_index: int) -> str:
    """3-D path of state[:, :3] + per-dim state/action time series.
    Returns cached file name under /gui-cache/plots/."""
    ds = _resolve(ds_rel)
    info = _read_info(ds) or {}
    parquet = _episode_parquet(ds, info, episode_index)
    key = hashlib.sha1(
        f"ds:{ds_rel}:{episode_index}:{parquet.stat().st_mtime_ns}".encode()
    ).hexdigest()[:16]
    out = PLOTS_CACHE / f"{key}.html"
    if out.exists():
        return out.name

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    table = _load_table(parquet)
    cols = _columns(info, table)
    times = np.asarray(table.column("timestamp").to_pylist(), dtype=float) \
        if "timestamp" in table.column_names else np.arange(table.num_rows, dtype=float)

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.55, 0.45],
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        subplot_titles=("state[:, :3] path (training frame)", "state / action dims"))

    if cols["state"]:
        state = np.asarray(table.column(cols["state"]).to_pylist(), dtype=float)
        if state.shape[1] >= 3:
            x, y, z = state[:, 0], state[:, 1], state[:, 2]
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z, mode="lines",
                line=dict(width=4, color=times, colorscale="Viridis"),
                name="path",
                text=[f"frame {i}<br>t={t:.2f}s" for i, t in enumerate(times)],
                hoverinfo="text"), row=1, col=1)
            fig.add_trace(go.Scatter3d(
                x=[x[0]], y=[y[0]], z=[z[0]], mode="markers",
                marker=dict(size=6, color="#3fb950"), name="start"), row=1, col=1)
            fig.add_trace(go.Scatter3d(
                x=[x[-1]], y=[y[-1]], z=[z[-1]], mode="markers",
                marker=dict(size=6, color="#d29922"), name="end"), row=1, col=1)
        for d in range(state.shape[1]):
            fig.add_trace(go.Scatter(
                x=times, y=state[:, d], mode="lines",
                name=f"state[{d}]", legendgroup="state"), row=1, col=2)
    if cols["action"]:
        action = np.asarray(table.column(cols["action"]).to_pylist(), dtype=float)
        for d in range(action.shape[1]):
            fig.add_trace(go.Scatter(
                x=times, y=action[:, d], mode="lines", visible="legendonly",
                name=f"action[{d}]", legendgroup="action",
                line=dict(dash="dot")), row=1, col=2)

    fig.update_layout(
        title=f"{ds.name} · episode {episode_index}",
        template="plotly_dark", height=620,
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=60, b=0))
    fig.write_html(out, include_plotlyjs="cdn")
    return out.name
