---
name: falsify-upload-dataset
description: Upload a validated LeRobot v2.1 dataset from `data/<name>/` to the Physical Intelligence GCS partner bucket `dronevla-raw-data`. Activates the service-account key shipped with the PI validator (`~/code/dataset_validation/pi-data-sharing/pi-external-partners-...json`), uses `validate.py compute-path` to derive the canonical `gs://...` destination, then runs `gsutil -m cp -r` to push the bundle. Pinned defaults: bucket `dronevla-raw-data`, version `v0.1.0`, data-type `teleop`.
---

# falsify-upload-dataset

This is the **second half** of the upload-readiness flow. Pair it with
[`falsify-validate-dataset`](../falsify-validate-dataset/SKILL.md), which
generates `meta/custom_metadata.csv` and confirms the bundle passes PI's
schema checks. Once a dataset validates, this skill ships it to GCS.

## When to use

- After `falsify-validate-dataset` reports `✓ All validations passed!`
  and you're ready to hand the bundle off to Physical Intelligence.
- When mirroring a `*_no_3pov` variant alongside its parent dataset.
- When re-uploading after fixing a per-episode bug (bump the version).

Not for: validation (use `falsify-validate-dataset`), or uploading
arbitrary local files (`gsutil cp` directly is fine for one-offs).

## Pinned conventions

| Knob | Value | Reason |
|---|---|---|
| Bucket | `dronevla-raw-data` | PI partner bucket; only writable by the SA below |
| Service account | `dronevla-external-sa@pi-external-partners.iam.gserviceaccount.com` | Scoped to this one bucket — cannot list other buckets in the project, by design |
| Key file | `~/code/dataset_validation/pi-data-sharing/pi-external-partners-814af4af5a99.json` | Ships with the PI validator repo |
| Default version | `v0.1.0` | Use the same version for sibling datasets that should be co-located (e.g. all four `*_no_3pov` variants) |
| `data-type` | `teleop` | Use `eval` only for evaluation rollouts that carry per-episode `checkpoint_path` |
| Dataset slug | `<dataset-dirname>` with `_` → `-` | `gate_scenes_real_synth_no_3pov` → `gate-scenes-real-synth-no-3pov` |

The canonical destination is:

```
gs://dronevla-raw-data/<slug>/<version>/<data-type>/
```

## Prerequisites

1. `falsify-validate-dataset` has produced `meta/custom_metadata.csv`
   and the validator returned `✓ All validations passed!`.
2. `gcloud` and `gsutil` are installed (they ship with the Google Cloud
   SDK; check with `which gsutil`).
3. The PI validator venv at
   `~/code/dataset_validation/pi-data-sharing/.venv/` exists. (See the
   `falsify-validate-dataset` skill for bootstrap.)

## Procedure

### 1. Activate the service account

The SA key lives next to the PI validator. Activate once per shell —
re-running is idempotent.

```bash
gcloud auth activate-service-account \
    --key-file=$HOME/code/dataset_validation/pi-data-sharing/pi-external-partners-814af4af5a99.json
gcloud config set project pi-external-partners
gcloud auth list   # confirm dronevla-external-sa@... has the asterisk
```

> `gsutil ls` at the project root will fail with `403
> storage.buckets.list denied` — that's expected. The SA can write to
> `dronevla-raw-data` but cannot enumerate buckets.

### 2. Compute the destination path (and re-validate)

```bash
cd ~/code/dataset_validation/pi-data-sharing
.venv/bin/python validate.py compute-path \
    --dataset-path /absolute/path/to/data/<dataset-dir> \
    --dataset-name <slug> \
    --bucket-name dronevla-raw-data \
    --data-type teleop \
    --dataset-version v0.1.0
```

`compute-path` runs `validate` first by default — pass
`--skip-validation` only if you've already validated in this session
and don't want to re-pay the parquet decode cost.

The command prints:

```
GCP Destination Path:
gs://dronevla-raw-data/<slug>/v0.1.0/teleop/

To upload your dataset, run:
    gsutil -m cp -r /abs/path/data/<dataset-dir>/* gs://dronevla-raw-data/<slug>/v0.1.0/teleop/
```

Copy that `gsutil` command verbatim — don't hand-author the path, since
the slug/version/data-type concatenation rules can drift.

### 3. Upload

```bash
gsutil -m cp -r /absolute/path/to/data/<dataset-dir>/* \
    gs://dronevla-raw-data/<slug>/v0.1.0/teleop/
```

- `-m` enables parallel uploads. A 300-episode dataset (~2.5 GB of
  PNG-encoded parquets) takes a few minutes on a typical connection.
- The trailing `/*` is intentional: it uploads the `data/` and `meta/`
  directories as siblings under `teleop/`, matching the PI consumer's
  expected layout.

### 4. Verify

```bash
gsutil ls gs://dronevla-raw-data/<slug>/v0.1.0/teleop/
# Expect: data/ and meta/ subdirs

gsutil du -sh gs://dronevla-raw-data/<slug>/v0.1.0/teleop/
# Confirm the byte count matches `du -sh data/<dataset-dir>/`
```

## Worked example: the `*_no_3pov` quartet

```bash
# One-time auth.
gcloud auth activate-service-account \
    --key-file=$HOME/code/dataset_validation/pi-data-sharing/pi-external-partners-814af4af5a99.json
gcloud config set project pi-external-partners

# Loop over the four bundles.
cd ~/code/dataset_validation/pi-data-sharing
for ds in gate_scenes_real_synth_no_3pov \
          gate_scenes_real_center_no_3pov \
          gate_scenes_center_no_3pov \
          gate_scenes_all_no_3pov; do
  slug=$(echo "$ds" | tr '_' '-')
  .venv/bin/python validate.py compute-path \
      --dataset-path /home/dfliu/code/falsify/data/$ds \
      --dataset-name $slug \
      --bucket-name dronevla-raw-data \
      --data-type teleop \
      --dataset-version v0.1.0 \
      --skip-validation
  gsutil -m cp -r /home/dfliu/code/falsify/data/$ds/* \
      gs://dronevla-raw-data/$slug/v0.1.0/teleop/
done
```

Resulting `gs://` layout:

```
gs://dronevla-raw-data/
├── gate-scenes-real-synth-no-3pov/v0.1.0/teleop/{data,meta}/
├── gate-scenes-real-center-no-3pov/v0.1.0/teleop/{data,meta}/
├── gate-scenes-center-no-3pov/v0.1.0/teleop/{data,meta}/
└── gate-scenes-all-no-3pov/v0.1.0/teleop/{data,meta}/
```

## Gotchas

- **Don't guess bucket names.** The SA can't `storage.buckets.list`, so
  probing alternative names just returns 403s for everything (including
  the real bucket if you fat-finger it). The bucket is always
  `dronevla-raw-data` for this SA — pin it.
- **Version bumps replace, don't merge.** `gsutil cp` overwrites
  same-key blobs but does not delete extras. If a previous upload had
  more episodes than the new one, the stale tail remains. Use
  `gsutil -m rsync -d -r` if you need delete-on-mirror semantics — but
  prefer bumping the version (`v0.2.0`) instead of mutating `v0.1.0`.
- **Don't upload without a validated `custom_metadata.csv`.** PI's
  downstream tooling reads that sidecar; missing or stale rows will
  silently break training pipelines on their side. Always run
  `falsify-validate-dataset` first.
- **`compute-path` uses a timestamp version by default.** Leaving
  `--dataset-version` unset gives every dataset a unique
  `YYYYMMDD_HHMMSS` slug — fine for one-offs, bad for cohorts that
  should share a version. Always pass `--dataset-version` explicitly
  for grouped uploads.
- **The key file is a real credential.** Don't commit it to a falsify
  branch, don't copy it to other machines, don't paste its contents
  anywhere. It already lives in `~/code/dataset_validation/pi-data-sharing/`
  outside the falsify repo — keep it there.

## Reference

- `~/code/dataset_validation/pi-data-sharing/validate.py` —
  `compute-path` subcommand source of truth.
- `~/code/dataset_validation/pi-data-sharing/README.md` — full GCS path
  format spec and upload notes.
- `.claude/skills/falsify-validate-dataset/SKILL.md` — produces the
  `custom_metadata.csv` this skill assumes exists.
