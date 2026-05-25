"""Offline action-prediction eval against a LeRobot v3 parquet dataset.

For each frame in the dataset we feed the (image, wrist_image, state, prompt)
to the policy gateway and compare the first action of the returned chunk
against the dataset's ground-truth action at that frame. "MPC style": one
infer per frame, take ``actions[0]``.

The dataset's ``observation.state`` is already in the policy's training
frame (MOCAP, negated yaw on dim 3) so we pass it through verbatim — this
script intentionally bypasses ``PiGatewayPolicy``'s NED↔MOCAP boundary.

Usage:

    bash -c 'export PI_API_KEY=...; source tools/env.sh; \\
        source tools/pi_inference_env.sh; \\
        PYTHONPATH=src python scripts/action_prediction/eval_action_prediction.py \\
            --dataset data/no_3pov_v3/gate_scenes_real_synth_no_3pov \\
            --policy-config configs/policies/pi_gateway/nonhistory_ccvhs1do_20k.yaml \\
            --max-episodes 5 --out runs/action_prediction/<name>'
"""
from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve(p: str | Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (REPO_ROOT / pp).resolve()


def _load_yaml(path: Path) -> dict:
    import yaml
    with path.open() as f:
        return yaml.safe_load(f)


def _decode_png(struct_val) -> np.ndarray:
    img = Image.open(io.BytesIO(struct_val["bytes"]))
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _list_episodes(dataset_dir: Path) -> list[Path]:
    files = sorted((dataset_dir / "data" / "chunk-000").glob("episode-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no episode parquets under {dataset_dir}/data/chunk-000/")
    return files


def _load_task_strings(dataset_dir: Path) -> dict[int, str]:
    tp = dataset_dir / "meta" / "tasks.parquet"
    if not tp.is_file():
        raise FileNotFoundError(f"tasks parquet missing: {tp}")
    t = pq.read_table(tp).to_pandas()
    # Robust to a couple of column-name conventions.
    text_col = "task" if "task" in t.columns else "__index_level_0__"
    return {int(r["task_index"]): str(r[text_col]) for _, r in t.iterrows()}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, type=Path,
                    help="LeRobot v3 dataset root (containing data/ + meta/).")
    ap.add_argument("--policy-config", required=True, type=Path,
                    help="Pi-gateway policy YAML — only its gateway_url, api_key, "
                         "camera_map, state_key, and prompt are read.")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output directory for the run summary.")
    ap.add_argument("--max-episodes", type=int, default=None,
                    help="Cap on episodes to evaluate (default: all).")
    ap.add_argument("--episode-stride", type=int, default=1,
                    help="Stride over episode files. Combine with --max-episodes "
                         "to subsample evenly across the dataset.")
    ap.add_argument("--episode-indices", type=str, default=None,
                    help="Comma-separated list of episode indices to evaluate "
                         "(e.g. '110,130,160,190'). Overrides --max-episodes "
                         "and --episode-stride.")
    ap.add_argument("--max-frames-per-episode", type=int, default=None,
                    help="Cap on frames per episode (default: all).")
    ap.add_argument("--frame-stride", type=int, default=1,
                    help="Stride within an episode (1 = every frame).")
    ap.add_argument("--prompt-override", type=str, default=None,
                    help="If set, send this prompt for every frame instead of "
                         "looking up tasks.parquet by task_index.")
    args = ap.parse_args(argv)

    dataset = _resolve(args.dataset)
    policy_yaml = _load_yaml(_resolve(args.policy_config))
    out_dir = _resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    camera_map = dict(policy_yaml.get("camera_map") or {
        "forward":  "observation/rgb/image",
        "downward": "observation/wrist_image/rgb/image",
    })
    state_key = policy_yaml.get("state_key", "observation/state")

    # Parquet image columns the LeRobot exporter writes for falsify.
    parquet_image_cols = {
        "forward":  "observation.images.image",
        "downward": "observation.images.wrist_image",
    }
    for name in camera_map:
        if name not in parquet_image_cols:
            raise SystemExit(
                f"camera_map declares {name!r} but this script only knows the "
                f"parquet columns for {list(parquet_image_cols)}; extend the map.")

    print(f"[eval] dataset    = {dataset}")
    print(f"[eval] policy     = {args.policy_config}")
    print(f"[eval] gateway    = {policy_yaml.get('gateway_url')}")
    print(f"[eval] out_dir    = {out_dir}")

    # ---- connect once ------------------------------------------------------
    from pi_inference_client import ClientConfig, InputData, PolicyClient
    import os, re
    _ENV_RE = re.compile(r"\$\{env:([A-Z_][A-Z0-9_]*)\}")
    def _expand(s: str | None) -> str | None:
        if s is None:
            return None
        def _sub(m):
            v = os.environ.get(m.group(1))
            if v is None:
                raise RuntimeError(f"env var {m.group(1)!r} unset")
            return v
        return _ENV_RE.sub(_sub, s)

    cc = ClientConfig(
        gateway_url=_expand(policy_yaml["gateway_url"]),
        api_key=_expand(policy_yaml.get("api_key", "")),
        execute_chunk_size=int(policy_yaml.get("execute_chunk_size", 25)),
        robot_task_string=policy_yaml.get("prompt", ""),
        compress_images=True,
        jpeg_quality=85,
    )
    client = PolicyClient(cc)
    client.connect()
    srv = client.server_config
    declared = set(srv.camera_names or [])
    wanted = set(camera_map.values())
    if declared and (wanted - declared):
        raise SystemExit(
            f"camera_map references keys server does not declare: "
            f"{sorted(wanted - declared)}; server has {sorted(declared)}")

    tasks = _load_task_strings(dataset)
    all_episodes = _list_episodes(dataset)
    if args.episode_indices:
        wanted = [int(x) for x in args.episode_indices.split(",")]
        by_idx = {int(p.stem.split("-")[-1]): p for p in all_episodes}
        missing = [i for i in wanted if i not in by_idx]
        if missing:
            raise SystemExit(f"episode indices not present in dataset: {missing}")
        episode_files = [by_idx[i] for i in wanted]
    else:
        episode_files = all_episodes[::args.episode_stride]
        if args.max_episodes is not None:
            episode_files = episode_files[: args.max_episodes]

    print(f"[eval] episodes   = {len(episode_files)}")

    per_step: list[dict] = []
    per_episode: list[dict] = []
    t_total = time.time()

    try:
        for ep_idx, ep_path in enumerate(episode_files):
            t_ep = time.time()
            t = pq.read_table(ep_path)
            n = t.num_rows
            stride = max(1, args.frame_stride)
            indices = list(range(0, n, stride))
            if args.max_frames_per_episode is not None:
                indices = indices[: args.max_frames_per_episode]

            ep_id = int(t.column("episode_index")[0].as_py())
            print(f"\n[ep {ep_idx+1}/{len(episode_files)}] {ep_path.name} "
                  f"ep_id={ep_id} n_frames={n} eval={len(indices)}")

            try:
                client.reset()
            except Exception:
                pass

            ep_errors: list[np.ndarray] = []
            ep_meta_rows: list[dict] = []
            img_cols = {
                cam: t.column(parquet_image_cols[cam]) for cam in camera_map
            }
            state_col  = t.column("observation.state")
            action_col = t.column("action")
            task_col   = t.column("task_index")

            for k, row_i in enumerate(indices):
                state = np.asarray(state_col[row_i].as_py(), dtype=np.float32)
                gt_action = np.asarray(action_col[row_i].as_py(), dtype=np.float64)
                task_idx = int(task_col[row_i].as_py())
                prompt = args.prompt_override or tasks[task_idx]

                images = {
                    camera_map[cam]: _decode_png(img_cols[cam][row_i].as_py())
                    for cam in camera_map
                }
                obs = InputData(
                    images=images,
                    states={state_key: state},
                    robot_task_string=prompt,
                )
                result = client.infer(obs)
                pred_chunk = np.asarray(result.actions, dtype=np.float64)
                if pred_chunk.ndim != 2 or pred_chunk.shape[1] != gt_action.shape[0]:
                    raise SystemExit(
                        f"shape mismatch: server actions {pred_chunk.shape} "
                        f"vs gt action {gt_action.shape}")
                pred_first = pred_chunk[0]
                err = pred_first - gt_action
                ep_errors.append(err)
                ep_meta_rows.append({
                    "episode_id": ep_id,
                    "frame_index": int(row_i),
                    "task_index": task_idx,
                    "gt_action": gt_action.tolist(),
                    "pred_first": pred_first.tolist(),
                    "err": err.tolist(),
                })
                if (k + 1) % 25 == 0 or k == 0:
                    se = float(np.mean(err ** 2))
                    print(f"  step {k+1}/{len(indices)}  "
                          f"|err|={np.linalg.norm(err):.4f}  per-step MSE={se:.4e}")

            ep_errors_arr = np.stack(ep_errors, axis=0) if ep_errors else np.zeros((0, gt_action.shape[0]))
            per_dim_mse = (ep_errors_arr ** 2).mean(axis=0) if ep_errors_arr.size else np.zeros(gt_action.shape[0])
            ep_summary = {
                "episode_index": ep_id,
                "parquet": str(ep_path),
                "n_frames_evaluated": len(indices),
                "elapsed_s": float(time.time() - t_ep),
                "mse_overall": float((ep_errors_arr ** 2).mean()) if ep_errors_arr.size else None,
                "mse_per_dim": per_dim_mse.tolist(),
                "max_abs_err_per_dim": np.abs(ep_errors_arr).max(axis=0).tolist() if ep_errors_arr.size else [],
            }
            per_episode.append(ep_summary)
            per_step.extend(ep_meta_rows)
            print(f"  -> ep_id={ep_id}  mse={ep_summary['mse_overall']:.4e}  "
                  f"elapsed={ep_summary['elapsed_s']:.1f}s")
    finally:
        try:
            client.close()
        except Exception:
            pass

    # ---- aggregate ---------------------------------------------------------
    all_errs = np.array([r["err"] for r in per_step], dtype=np.float64) if per_step else np.zeros((0, 7))
    all_preds = np.array([r["pred_first"] for r in per_step], dtype=np.float64) if per_step else np.zeros((0, 7))
    all_gt = np.array([r["gt_action"] for r in per_step], dtype=np.float64) if per_step else np.zeros((0, 7))
    aggregate = {
        "dataset": str(dataset),
        "policy_config": str(args.policy_config),
        "gateway_url": policy_yaml.get("gateway_url"),
        "traceability": policy_yaml.get("traceability"),
        "n_episodes": len(per_episode),
        "n_frames_total": len(per_step),
        "elapsed_total_s": float(time.time() - t_total),
        "mse_overall": float((all_errs ** 2).mean()) if all_errs.size else None,
        "mse_per_dim": (all_errs ** 2).mean(axis=0).tolist() if all_errs.size else [],
        "rmse_per_dim": np.sqrt((all_errs ** 2).mean(axis=0)).tolist() if all_errs.size else [],
        "mae_per_dim": np.abs(all_errs).mean(axis=0).tolist() if all_errs.size else [],
        "gt_std_per_dim": all_gt.std(axis=0).tolist() if all_gt.size else [],
        "pred_std_per_dim": all_preds.std(axis=0).tolist() if all_preds.size else [],
        "per_episode": per_episode,
    }
    (out_dir / "summary.json").write_text(json.dumps(aggregate, indent=2))

    # Per-step records can get large; save as npz for downstream analysis.
    if per_step:
        np.savez(
            out_dir / "per_step.npz",
            gt=all_gt, pred=all_preds, err=all_errs,
            episode_ids=np.array([r["episode_id"] for r in per_step], dtype=np.int64),
            frame_indices=np.array([r["frame_index"] for r in per_step], dtype=np.int64),
        )

    print(f"\n[eval] done: {aggregate['n_episodes']} eps, "
          f"{aggregate['n_frames_total']} frames, "
          f"MSE={aggregate['mse_overall']:.4e}, "
          f"elapsed={aggregate['elapsed_total_s']:.0f}s")
    print(f"[eval] mse_per_dim = {[f'{x:.3e}' for x in aggregate['mse_per_dim']]}")
    print(f"[eval] rmse_per_dim = {[f'{x:.3e}' for x in aggregate['rmse_per_dim']]}")
    print(f"[eval] wrote {out_dir/'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
