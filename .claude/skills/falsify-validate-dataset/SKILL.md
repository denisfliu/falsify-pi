---
name: falsify-validate-dataset
description: Validate a falsify-produced LeRobot v2.1 dataset against Physical Intelligence's external dataset-sharing validator (~/code/dataset_validation/pi-data-sharing). Generates the required `meta/custom_metadata.csv` sidecar from the dataset's own `episodes.jsonl` + parquet mtimes, then runs the validator's `validate` (and optionally `compute-path`) CLI from its own venv.
---

# falsify-validate-dataset

The PI dataset validator is the upload-readiness gate for any LeRobot
dataset we hand off to Physical Intelligence's training infrastructure.
It lives outside this repo at `../dataset_validation/pi-data-sharing/`
and expects a sidecar CSV (`meta/custom_metadata.csv`) that LeRobot
datasets don't carry by default. This skill closes that gap.

## When to use

- After `falsify-combine-datasets` produces a final LeRobot v2.1 bundle
  and before uploading it to GCS.
- After collecting a real-world dataset (e.g. `gate_scenes_real_combined`)
  whose episode timestamps need to be reconstructed from file mtimes.
- When you need to compute the GCS upload path for a dataset (e.g.
  `gs://<bucket>/<dataset>/<version>/teleop/`).

Not for: validating individual `.parquet` files in isolation, or
checking the *content* of a dataset against the falsify schema — for
that, see the schema-parity checks in `falsify-combine-datasets`.

## Prerequisites

- The dataset is laid out as LeRobot v2.1:
  ```
  <dataset>/
    data/chunk-000/episode_*.parquet
    meta/info.json
    meta/episodes.jsonl
    meta/tasks.jsonl
    meta/episodes_stats.jsonl
  ```
- The PI validator venv at `~/code/dataset_validation/pi-data-sharing/.venv/`
  has been created. **Bootstrap** (one-time, do **not** reuse the falsify
  SousVide venv — cloudpathlib + google-cloud-storage would pollute the
  pinned ML stack):

  ```bash
  cd ~/code/dataset_validation/pi-data-sharing
  uv venv .venv --python 3.11
  uv pip install --python .venv/bin/python -r requirements.txt
  ```

## Procedure

### 1. Generate the sidecar CSV

```bash
.venv/bin/python tools/build_validator_metadata.py \
    --dataset-path data/<your_dataset>
```

This writes `data/<your_dataset>/meta/custom_metadata.csv` with one
row per `episodes.jsonl` entry. Re-running overwrites idempotently.

Defaults:

- `operator_id=denis`, `robot_id=carl` — override with
  `--operator-id` / `--robot-id` for differently-collected bundles.
- `station_id` is derived from the task string: `"left"` → `left_gate`,
  `"right"` → `right_gate`, anything else → `unknown_gate`. Edit the
  `task_to_station` function in the script if your dataset has more
  stations.
- `start_timestamp` is the parquet file's mtime (UTC seconds). The
  dataset stores only episode-relative timestamps, so mtime is the
  cleanest stable choice — and falls comfortably in the validator's
  required `[year 2000, 2100]` range.
- `is_eval_episode=False`, `checkpoint_path=""`, `success=True` for
  every row (i.e. teleop defaults). Override `--success-default false`
  if a known-bad batch is being archived. There is currently no
  per-episode success column — fix the script first if you need it.

### 2. Run the validator

```bash
cd ~/code/dataset_validation/pi-data-sharing
.venv/bin/python validate.py validate \
    --dataset-path /absolute/path/to/data/<your_dataset> \
    --data-type teleop
```

Pass `--data-type eval` instead for evaluation rollouts; in that case
the metadata CSV must have `is_eval_episode=True` and a populated
`checkpoint_path` (a valid `gs://bucket/path` URI) for every row, so
the generator script needs adjusting before step 1.

Success looks like:

```
✓ All validations passed!
```

Failure prints a numbered list. The most common failures and their
fixes are in *Gotchas* below.

### 3. (Optional) Compute the GCS upload path

```bash
.venv/bin/python validate.py compute-path \
    --dataset-path /absolute/path/to/data/<your_dataset> \
    --dataset-name <dataset-slug> \
    --bucket-name <gcs-bucket> \
    --data-type teleop \
    --dataset-version v0.1.0
```

Prints the `gsutil -m cp -r ...` command. The path format is
`gs://<bucket>/<dataset-slug>/<version>/<data_type>/`. Add
`--custom-folder-prefix experiments/<batch>` to nest under a prefix.

The `compute-path` command runs `validate` first by default; pass
`--skip-validation` to skip it (not recommended).

## Worked example: `gate_scenes_real_combined`

```bash
# From the falsify repo root.
.venv/bin/python tools/build_validator_metadata.py \
    --dataset-path data/gate_scenes_real_combined
# → wrote .../meta/custom_metadata.csv (100 episodes)
#   stations: ['left_gate', 'right_gate']

cd ~/code/dataset_validation/pi-data-sharing
.venv/bin/python validate.py validate \
    --dataset-path /home/dfliu/code/falsify/data/gate_scenes_real_combined \
    --data-type teleop
# → ✓ All validations passed!

.venv/bin/python validate.py compute-path \
    --dataset-path /home/dfliu/code/falsify/data/gate_scenes_real_combined \
    --dataset-name gate_scenes_real \
    --bucket-name falsify-data \
    --data-type teleop \
    --dataset-version v0.1.0
# → gs://falsify-data/gate-scenes-real/v0.1.0/teleop/
```

## Required CSV columns (validator-side contract)

Exact match — extras or missing columns both fail.

| Column | Type | Notes |
|---|---|---|
| `episode_index` | int | matches `episodes.jsonl` |
| `operator_id` | string | who collected |
| `is_eval_episode` | bool | must match `--data-type` |
| `episode_id` | string | unique; default `<dataset>_ep_NNNN` |
| `start_timestamp` | int | UTC seconds, range 2000–2100 |
| `checkpoint_path` | string | `gs://...` URI; only for eval; empty for teleop |
| `success` | bool | per-episode success |
| `station_id` | string | scene / site identifier |
| `robot_id` | string | drone hardware identifier |

## Gotchas

- **Run the validator from its own venv.** The falsify venv (symlinked
  to SousVide's) doesn't have `cloudpathlib` / `tyro`. Adding them with
  `uv pip install --force-reinstall` would re-resolve the ML stack and
  break `nerfstudio` (see the top-level `CLAUDE.md` env caveat).
- **Use absolute dataset paths** when invoking the validator from
  outside the falsify repo, otherwise `cloudpathlib`'s `AnyPath`
  resolution gets confused about relative vs gs:// URIs.
- **Eval data needs real checkpoint URIs.** The validator rejects
  `checkpoint_path=""` rows when `is_eval_episode=True`, and rejects
  non-empty `checkpoint_path` when `is_eval_episode=False`. Patch
  `tools/build_validator_metadata.py` (it currently hard-codes both as
  teleop) before validating an eval bundle.
- **Timestamp range is enforced.** If your parquet mtimes predate 2000
  or post-date 2100 (touched files, archive extraction, etc.), the
  validator rejects them. Re-touch the files or override
  `start_timestamp` in the script with a synthetic ramp from `time.time()`.
- **Annotation JSON is optional.** Don't generate
  `meta/custom_annotation.json` unless you actually have span-level
  annotations to declare; an empty or malformed file is worse than no
  file.

## Reference

- `tools/build_validator_metadata.py` — CSV generator (falsify side).
- `~/code/dataset_validation/pi-data-sharing/validate.py` — the
  validator CLI (`validate` + `compute-path` subcommands).
- `~/code/dataset_validation/pi-data-sharing/README.md` — column
  definitions, full validation rules, GCP path format.
- `~/code/dataset_validation/pi-data-sharing/demo/sample_dataset/` —
  reference layout. Useful when debugging "what does a valid CSV look
  like" without re-reading the validator source.
