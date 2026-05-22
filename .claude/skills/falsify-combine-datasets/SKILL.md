---
name: falsify-combine-datasets
description: Merge multiple LeRobot v2.1 dataset directories into one coherent dataset — renumber episode_index / index globally, drop trailing bad trajectories, reassign task_index per range, and regenerate all four meta files. Schema-byte-identical to the DroneVLA2.0 reference parquet.
---

# falsify-combine-datasets

The final assembly step. Once you've got a bunch of per-episode parquets
(from real-world collection runs *or* from `falsify-export-parquet` /
`falsify-orchestrate-batch`), this skill stitches them into a single
LeRobot v2.1 dataset that downstream training consumes directly.

## When to use

- You have N source dataset directories, each a separate LeRobot bundle.
- Some directories have a known-bad final trajectory you want excluded.
- Different episodes belong to different tasks (e.g., left-gate vs.
  right-gate), and you need `task_index` reassigned accordingly.
- You want the regenerated `info.json` / `tasks.jsonl` / `episodes.jsonl`
  / `episodes_stats.jsonl` to match the actual combined contents.

## Inputs

- `--src` — parent directory containing each source LeRobot dataset
  (directories with `data/chunk-000/episode_*.parquet` and
  `meta/info.json`).
- `--out` — destination dataset directory.
- `--task "<count>:<text>"` — repeatable. The first `<count>` episodes
  (in global combined order) get the first task, the next `<count>`
  get the second, etc. Use `rest` as the last entry's count to consume
  whatever's left. **Identical texts across multiple `--task` specs
  collapse to a single `task_index`** in the output — `tasks.jsonl`
  always ends up with one row per unique string, never two rows under
  different indices pointing at the same prompt. This is what prevents
  the historical "doubled task indices" failure where mixing source
  bundles that happened to share a prompt produced
  `total_tasks: N > unique_strings`.
- `--drop-last-pattern` — fnmatch pattern (default `*_bad_last`).
  Dirs matching this pattern have their *last* parquet (by sorted
  episode index) dropped.
- `--overwrite` — wipe `--out` before writing.

## Source ordering

Directories are sorted by their trailing numeric suffix (so
`dataset_2` < `dataset_10`), then alphabetically as a tiebreaker. Within
each directory, parquets are sorted by their `episode_NNNNNN.parquet`
filename. Use this to predict which episodes get which `--task`.

## Procedure

```bash
PYTHONPATH=src .venv/bin/python -m falsify.cli.combine_lerobot \
    --src data/gate_scenes_real/datasets-20260513T185355Z-3-001/datasets \
    --out data/gate_scenes_real_combined \
    --drop-last-pattern "*_bad_last" \
    --task "50:go through the gate on the left and hover over the stuffed animal" \
    --task "rest:go through the gate on the right and hover over the stuffed animal" \
    --overwrite
```

Output layout:

```
<out>/
  data/chunk-000/episode_000000.parquet ... episode_NNNNNN.parquet
  meta/info.json
  meta/tasks.jsonl
  meta/episodes.jsonl
  meta/episodes_stats.jsonl
```

## Verification

Always verify the labels actually match the trajectories — folder
naming is not always the source of truth. Two quick checks:

### 1. Schema parity against the reference

```python
import pyarrow.parquet as pq, json
ours = pq.read_table('<out>/data/chunk-000/episode_000000.parquet')
ref  = pq.read_table('~/Downloads/episode_000008.parquet')
assert ours.column_names == ref.column_names
for c in ours.column_names:
    assert str(ours.schema.field(c).type) == str(ref.schema.field(c).type)
md_ours = json.loads(ours.schema.metadata[b'huggingface'])
md_ref  = json.loads(ref.schema.metadata[b'huggingface'])
assert md_ours == md_ref
```

### 2. Cross-check task labels against trajectory state

The combiner trusts the `--task` order; if the underlying source files
were stored in a different physical order than you assumed, episodes
will end up with the wrong task. Spot-check the boundary episodes:

```python
import pyarrow.parquet as pq, numpy as np, json
out = '<out>'
tasks = [json.loads(l) for l in open(f'{out}/meta/tasks.jsonl')]
for ei in [0, 49, 50, 98]:                       # adjust for your boundaries
    t = pq.read_table(f'{out}/data/chunk-000/episode_{ei:06d}.parquet',
                       columns=['state', 'task_index'])
    arr = np.asarray(t.column('state').to_pylist())
    ti = t.column('task_index')[0].as_py()
    # Pick a feature that distinguishes your tasks (gate y for left/right gate)
    y_range = (arr[:,1].min(), arr[:,1].max())
    print(f'ep {ei:03d}  y range {y_range}  task: {tasks[ti]["task"][:40]}')
```

If a "left-gate" trajectory shows `y_min < -1.0`, it's actually a
right-gate run mislabeled — fix the `--task` counts and re-run.

## Combining synthetic with real data

Same CLI; `--src` can point at any LeRobot-shaped directory tree. To
mix a directory of falsify-generated parquets with a real-world bundle,
put both under a single parent first:

```bash
mkdir -p data/combined_mix/sources
ln -s "$(pwd)/runs/datasets/left_gate_variants"   data/combined_mix/sources/synthetic
ln -s "$(pwd)/data/gate_scenes_real_combined"     data/combined_mix/sources/real

PYTHONPATH=src .venv/bin/python -m falsify.cli.combine_lerobot \
    --src data/combined_mix/sources \
    --out data/combined_mix/dataset \
    --task "rest:go through the gate on the left and hover over the stuffed animal" \
    --overwrite
```

Note: source directories must already use the LeRobot v2.1 layout
(`data/chunk-000/` + `meta/`). `falsify-export-parquet` produces
single-episode parquets but not the full LeRobot directory layout; if
you need to combine those, wrap them into a minimal LeRobot dir first
(one `data/chunk-000/` + a stub `meta/info.json`).

## Hands off to

- Downstream training in DroneVLA2.0 — point the trainer at `<out>` the
  same way it points at any other LeRobot dataset.
- **`falsify-orchestrate-batch`** — if you need to *first* generate
  more parquets before combining.

## Per-episode stats are cached

LeRobot v2.1 source bundles ship `meta/episodes_stats.jsonl` with byte-
identical schema to what `combine_lerobot` would write. When present, the
combiner **reuses** the immutable per-episode fields (image / wrist_image
/ 3pov_1 / state / actions / timestamp / frame_index) verbatim, and only
re-derives the three scalars it actively renumbers (`episode_index`,
`index`, `task_index`) in closed form. The run prints
``[stats] reused precomputed: N  recomputed from pixels: M`` — if the
"recomputed from pixels" count is non-zero you'll pay ~1.5 s/episode on
those entries while the cached ones cost microseconds.

The cache hole that historically dominated wall-clock was decoding 100
PNGs per camera per episode and reducing a `(100, 256, 256, 3) float32`
cube four times for min/max/mean/std. On a 300-episode combine this used
to take ~7.5 minutes; with the cache it's ~25 s, dominated by parquet
copy throughput.

If a source dataset is missing `episodes_stats.jsonl` (older bundle, or
a `_no_3pov_v3` derivative — LeRobot v3.0 replaces it with a single
`stats.json`), the combiner falls back to recomputing for those episodes.

## Gotchas

- **`--task` counts must sum to the number of kept episodes** (after
  bad-last dropping). The CLI errors fast if they don't, but be aware
  when bad-last counts vary.
- **Sorting is numeric-suffix-aware**: `dataset_2` < `dataset_10`. If
  your source dirs don't end in digits, fall back to alphabetical.
- **Image stats are sampled**, not full-pixel. The combiner samples 100
  frames per episode for image min/max/mean/std (matches LeRobot
  convention). Change with `--image-stat-sample`.
- **Codebase version** defaults to `v2.1`. If the trainer expects a
  different version string, override with `--codebase-version`.
- **chunks_size** stays at 1000 by default; all combined episodes go
  into `chunk-000` unless your total exceeds that. For larger datasets,
  the CLI will need a follow-up patch to support multiple chunks.

## Reference

- `src/falsify/cli/combine_lerobot.py` — implementation.
- `~/Downloads/episode_000008.parquet` — reference schema (the
  combiner produces byte-identical column structure + HF metadata).
- `data/gate_scenes_real_combined/` — example output from the real
  gate-scenes bundle.
