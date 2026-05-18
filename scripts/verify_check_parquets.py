"""Verify the 12 *_check parquets and dump per-episode camera image folders.

Checks per file:
  1. Schema matches the reference real-world parquet (column names + types).
  2. Row count > 0.
  3. episode_index constant; frame_index == [0, N-1]; index strictly increasing.
  4. state is (7,), dims 0..3 finite, dims 4..6 exactly zero.
  5. actions is (7,), finite, std on at least one of dims 0..3 > 0.
  6. timestamp strictly monotonic with dt ≈ 0.1 s (10 Hz, tol 5e-3).
  7. image, wrist_image decode as 256×256×3 PNGs and have nontrivial variance.
     3pov_1 must decode as 256×256×3 and be all-zero (per the embodiment).

Also writes per-episode animated GIFs under runs/check_image_dumps/<dataset>/<episode>/:
  forward.gif and wrist.gif (10 fps, one frame per parquet row), plus
  3pov.gif (2-frame: first+last, just enough to sanity-check the all-zero
  invariant without flooding the folder).
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
REFERENCE = REPO / "data/gate_scenes_real_combined/data/chunk-000/episode_000098.parquet"
DUMP_ROOT = REPO / "runs/check_image_dumps"
DATASETS = sorted((REPO / "runs/datasets").glob("synth_*"))

OK = "✓"
BAD = "✗"


def decode_png(blob: dict) -> np.ndarray:
    img = Image.open(io.BytesIO(blob["bytes"]))
    return np.asarray(img)


def check_one(path: Path, ref_schema, dump_dir: Path) -> tuple[bool, list[str]]:
    failures: list[str] = []
    t = pq.read_table(path)
    n = t.num_rows

    # 1. Schema.
    if not t.schema.equals(ref_schema):
        ref_cols = {f.name: str(f.type) for f in ref_schema}
        got_cols = {f.name: str(f.type) for f in t.schema}
        diffs = []
        for c in set(ref_cols) | set(got_cols):
            if ref_cols.get(c) != got_cols.get(c):
                diffs.append(f"{c}: ref={ref_cols.get(c)} got={got_cols.get(c)}")
        failures.append("schema mismatch: " + "; ".join(diffs))

    # 2. Row count.
    if n <= 0:
        failures.append(f"row count = {n}")

    # 3. Index/episode bookkeeping.
    epi = t["episode_index"].to_numpy()
    fi = t["frame_index"].to_numpy()
    idx = t["index"].to_numpy()
    if not np.all(epi == epi[0]):
        failures.append("episode_index not constant within file")
    if not np.array_equal(fi, np.arange(n)):
        failures.append(f"frame_index != [0..{n-1}]")
    if not np.all(np.diff(idx) == 1):
        failures.append("global index not strictly +1 monotonic")

    # 4. state.
    state = np.stack(t["state"].to_numpy())
    if state.shape != (n, 7):
        failures.append(f"state shape = {state.shape}, expected ({n}, 7)")
    elif not np.all(np.isfinite(state[:, :4])):
        failures.append("state[:, 0:4] has non-finite values")
    elif not np.all(state[:, 4:] == 0):
        failures.append("state[:, 4:7] not all zero")

    # 5. actions.
    act = np.stack(t["actions"].to_numpy())
    if act.shape != (n, 7):
        failures.append(f"actions shape = {act.shape}, expected ({n}, 7)")
    elif not np.all(np.isfinite(act)):
        failures.append("actions has non-finite values")
    elif not np.any(act[:, :4].std(axis=0) > 0):
        failures.append("actions dims 0..3 are all constant (suspect degenerate)")

    # 6. timestamps.
    ts = t["timestamp"].to_numpy().astype(float).ravel()
    if not np.all(np.diff(ts) > 0):
        failures.append("timestamps not strictly monotonic")
    else:
        dt = np.diff(ts)
        if not np.allclose(dt, 0.1, atol=5e-3):
            failures.append(f"dt not ≈0.1s (min={dt.min():.4f}, max={dt.max():.4f})")

    # 7. images.
    dump_dir.mkdir(parents=True, exist_ok=True)
    bad_imgs = []
    for col, label, dump_all in [("image", "forward", True),
                                  ("wrist_image", "wrist", True),
                                  ("3pov_1", "3pov", False)]:
        arrs = []
        rows = t[col].to_pylist()
        for k, blob in enumerate(rows):
            arr = decode_png(blob)
            arrs.append(arr)
            if arr.shape != (256, 256, 3):
                bad_imgs.append(f"{col} frame {k} shape={arr.shape}")
        stack = np.stack(arrs)
        if col == "3pov_1":
            if stack.any():
                bad_imgs.append("3pov_1 not all-zero (expected zero pad)")
        else:
            # Each frame should have at least some variance.
            per_frame_std = stack.reshape(stack.shape[0], -1).std(axis=1)
            if (per_frame_std < 1.0).any():
                k = int(np.argmin(per_frame_std))
                bad_imgs.append(f"{col} frame {k} near-constant (std={per_frame_std[k]:.3f})")

        # GIF dumping skipped for the 200-episode synth run (would produce
        # 600 GIFs × ~250 frames each — too much disk). Re-enable per-episode
        # by setting a SKIP_GIFS=0 env var if needed.
        import os
        if os.environ.get("SKIP_GIFS", "1") != "1":
            if dump_all:
                frames = [Image.fromarray(a) for a in arrs]
            else:
                frames = [Image.fromarray(arrs[0]), Image.fromarray(arrs[-1])] \
                         if len(arrs) >= 2 else [Image.fromarray(arrs[0])]
            if len(frames) > 1:
                frames[0].save(
                    dump_dir / f"{label}.gif",
                    save_all=True, append_images=frames[1:],
                    duration=100, loop=0, optimize=False, disposal=2,
                )
            else:
                frames[0].save(dump_dir / f"{label}.gif")
    if bad_imgs:
        failures.append("image issues: " + "; ".join(bad_imgs))

    return len(failures) == 0, failures


def main() -> int:
    ref_schema = pq.read_table(REFERENCE).schema
    print(f"reference schema: {REFERENCE.relative_to(REPO)}")
    print(f"  columns: {[f.name for f in ref_schema]}")
    print()

    overall_ok = True
    all_paths = []
    for ds in DATASETS:
        all_paths.extend(sorted(ds.rglob("*.parquet")))
    if not all_paths:
        print("NO PARQUETS FOUND", file=sys.stderr)
        return 1

    fmt = "{:60s}  {:>4s}  {}"
    print(fmt.format("file", "rows", "result"))
    print("-" * 110)
    for p in all_paths:
        dataset_name = p.parts[-3]   # runs/datasets/<dataset>/episode_NNN/...
        ep_name = p.stem
        dump_dir = DUMP_ROOT / dataset_name / ep_name
        ok, failures = check_one(p, ref_schema, dump_dir)
        if not ok:
            overall_ok = False
        rel = str(p.relative_to(REPO))
        n = pq.read_table(p, columns=["frame_index"]).num_rows
        status = OK if ok else BAD + " " + " | ".join(failures)
        print(fmt.format(rel, str(n), status))

    print()
    print(f"image dumps: {DUMP_ROOT.relative_to(REPO)}/<dataset>/<episode>/")
    print(f"overall: {'PASS' if overall_ok else 'FAIL'}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
