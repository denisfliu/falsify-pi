#!/usr/bin/env bash
# Build data/atomic_datasets/real_synth_dagger1 by combining four source
# datasets and stripping the 3pov_1 column.
#
# Sources (in this order — preserves the synth/real/dagger episode-block
# layout the downstream pipeline assumes):
#   1. synth_left_gate                (50 ep, left)
#   2. synth_right_gate               (50 ep, right)
#   3. gate_scenes_real_combined      (100 ep — 50 left then 50 right)
#   4. dagger-1_real_synth            (100 ep — 50 left then 50 right)
#                                        ↑ the new corrective-maneuver dataset
#
# Total: 300 episodes. tasks.jsonl ends up with 6 entries (4 unique
# strings) — same pattern gate_scenes_real_combined already follows for
# its left/right split.
#
# Step 2 strips the `3pov_1` column from every parquet and removes it
# from info.json + episodes_stats.jsonl (mirrors scripts/strip_3pov.py).
#
# Run from repo root:
#   source tools/env.sh
#   bash tools/build_real_synth_dagger1.sh

set -uo pipefail
cd "$(dirname "$0")/.."

OUT_NAME=real_synth_dagger1
FINAL_OUT=data/atomic_datasets/${OUT_NAME}
TMP_COMBINED=/tmp/${OUT_NAME}_with_3pov
STAGING=/tmp/${OUT_NAME}_staging

# ---------- 1. stage source datasets so combine_lerobot finds them ----
rm -rf "$STAGING" "$TMP_COMBINED"
mkdir -p "$STAGING"
ln -s "$(pwd)/data/atomic_datasets/synth_left_gate"           "$STAGING/00_synth_left_gate"
ln -s "$(pwd)/data/atomic_datasets/synth_right_gate"          "$STAGING/01_synth_right_gate"
ln -s "$(pwd)/data/atomic_datasets/gate_scenes_real_combined" "$STAGING/02_gate_scenes_real_combined"
ln -s "$(pwd)/data/atomic_datasets/dagger-1_real_synth"       "$STAGING/03_dagger-1_real_synth"
echo "[stage] staged 4 sources under $STAGING"

# ---------- 2. combine ------------------------------------------------
echo
echo "=========================================================="
echo "[$(date +%H:%M:%S)] combine_lerobot → $TMP_COMBINED"
echo "=========================================================="
PYTHONPATH=src python -m falsify.cli.combine_lerobot \
    --src "$STAGING" \
    --out "$TMP_COMBINED" \
    --task "50:go through the gate on the left and hover over the stuffed animal" \
    --task "50:go through the gate on the right and hover over the stuffed animal" \
    --task "50:go through the gate on the left and hover over the stuffed animal" \
    --task "50:go through the gate on the right and hover over the stuffed animal" \
    --task "50:go through the gate on the left and hover over the stuffed animal" \
    --task "rest:go through the gate on the right and hover over the stuffed animal" \
    --overwrite

# ---------- 3. strip 3pov_1 ------------------------------------------
echo
echo "=========================================================="
echo "[$(date +%H:%M:%S)] strip 3pov_1 → $FINAL_OUT"
echo "=========================================================="
PYTHONPATH=src python - <<EOF
"""In-place port of scripts/strip_3pov.py, parameterised on src/dst."""
import json
import shutil
from pathlib import Path
import pyarrow.parquet as pq

SRC = Path("$TMP_COMBINED")
DST = Path("$FINAL_OUT")
DROP = "3pov_1"

if DST.exists():
    shutil.rmtree(DST)
(DST / "data" / "chunk-000").mkdir(parents=True)
(DST / "meta").mkdir(parents=True)

src_parquets = sorted((SRC / "data" / "chunk-000").glob("*.parquet"))
for p in src_parquets:
    t = pq.read_table(p)
    if DROP in t.column_names:
        t = t.drop([DROP])
    pq.write_table(t, DST / "data" / "chunk-000" / p.name)
print(f"  rewrote {len(src_parquets)} parquets")

info = json.loads((SRC / "meta" / "info.json").read_text())
info.get("features", {}).pop(DROP, None)
(DST / "meta" / "info.json").write_text(json.dumps(info, indent=4))

with (SRC / "meta" / "episodes_stats.jsonl").open() as fin, \\
     (DST / "meta" / "episodes_stats.jsonl").open("w") as fout:
    for line in fin:
        obj = json.loads(line)
        obj.get("stats", {}).pop(DROP, None)
        fout.write(json.dumps(obj) + "\n")

for fname in ("episodes.jsonl", "tasks.jsonl", "custom_metadata.csv"):
    s = SRC / "meta" / fname
    if s.exists():
        shutil.copy2(s, DST / "meta" / fname)
print("  meta: info.json + episodes_stats.jsonl rewritten; episodes/tasks copied")
EOF

# ---------- 4. cleanup -----------------------------------------------
rm -rf "$STAGING" "$TMP_COMBINED"
echo
echo "=========================================================="
echo "[$(date +%H:%M:%S)] done → $FINAL_OUT"
echo "=========================================================="
ls -la "$FINAL_OUT"
echo "---"
echo "parquets: $(ls $FINAL_OUT/data/chunk-000/ | wc -l)"
echo "info.json (excerpt):"
python -c "
import json
d = json.load(open('$FINAL_OUT/meta/info.json'))
print(f'  total_episodes={d[\"total_episodes\"]}  total_frames={d[\"total_frames\"]}  total_tasks={d[\"total_tasks\"]}')
print(f'  features: {list(d.get(\"features\", {}).keys())}')
"
echo "tasks:"
cat "$FINAL_OUT/meta/tasks.jsonl"
