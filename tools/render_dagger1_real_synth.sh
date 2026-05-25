#!/usr/bin/env bash
# Render the 50 left_gate + 50 right_gate recovery NPZs from
# nonhistory_real_synth into one LeRobot v2.1 dataset at
# data/atomic_datasets/dagger-1_real_synth/.
#
# Pipeline:
#   1. render_recoveries_to_dataset.py per scene — applies each trial's
#      GateRigidPerturbation to the gsplat before rendering, so the
#      parquets' camera frames show the actual perturbed gate the
#      recovery was planned for (NOT the nominal scene gate). Output:
#      <scene>/episode_NNNNNN/episode_NNNNNN.parquet
#   2. Reshape each scene's parquets into <scene>/data/chunk-000/
#      (the v2.1 layout combine_lerobot expects).
#   3. combine_lerobot merges both into the final dataset, with the
#      per-scene prompts assigned via --task ranges.
#
# Run from repo root:
#   source tools/env.sh
#   bash tools/render_dagger1_real_synth.sh

set -uo pipefail
cd "$(dirname "$0")/.."

DATASET_NAME=dagger-1_real_synth
FINAL_OUT=data/atomic_datasets/${DATASET_NAME}
TMP=/tmp/${DATASET_NAME}_intermediate

LEFT_RUN=runs/recovery_collection/nonhistory_real_synth_31ohxgxv_5000/left_gate/run-001-20260521_115122
RIGHT_RUN=runs/recovery_collection/nonhistory_real_synth_31ohxgxv_5000/right_gate/run-001-20260521_122910

rm -rf "$TMP"
mkdir -p "$TMP/left/data/chunk-000" "$TMP/right/data/chunk-000"
mkdir -p "$TMP/left/meta" "$TMP/right/meta"

echo "=========================================================="
echo "[$(date +%H:%M:%S)] Step 1a: render left_gate (50 recoveries, perturbed-gate)"
echo "=========================================================="
PYTHONPATH=src python scripts/recovery/render_recoveries_to_dataset.py \
    --recovery-run-dir "$LEFT_RUN" \
    --scene configs/scenes/left_gate.yaml \
    --frame configs/frames/carl_dual.yaml \
    --embodiment configs/embodiments/carl_dual_mocap.yaml \
    --out "$TMP/left_raw" \
    --episode-index-base 0 --index-offset 0
# Flatten <left_raw>/episode_NNNNNN/episode_NNNNNN.parquet → <left>/data/chunk-000/
mv "$TMP"/left_raw/episode_*/episode_*.parquet "$TMP/left/data/chunk-000/"
rm -rf "$TMP/left_raw"
echo '{"placeholder": "stub for combine_lerobot dataset-discovery"}' > "$TMP/left/meta/info.json"

echo
echo "=========================================================="
echo "[$(date +%H:%M:%S)] Step 1b: render right_gate (50 recoveries, perturbed-gate)"
echo "=========================================================="
PYTHONPATH=src python scripts/recovery/render_recoveries_to_dataset.py \
    --recovery-run-dir "$RIGHT_RUN" \
    --scene configs/scenes/right_gate.yaml \
    --frame configs/frames/carl_dual.yaml \
    --embodiment configs/embodiments/carl_dual_mocap.yaml \
    --out "$TMP/right_raw" \
    --episode-index-base 0 --index-offset 0
mv "$TMP"/right_raw/episode_*/episode_*.parquet "$TMP/right/data/chunk-000/"
rm -rf "$TMP/right_raw"
echo '{"placeholder": "stub for combine_lerobot dataset-discovery"}' > "$TMP/right/meta/info.json"

echo
echo "=========================================================="
echo "[$(date +%H:%M:%S)] Step 2: combine into ${FINAL_OUT}"
echo "=========================================================="
PYTHONPATH=src python -m falsify.cli.combine_lerobot \
    --src "$TMP" \
    --out "$FINAL_OUT" \
    --task "50:go through the gate on the left and hover over the stuffed animal" \
    --task "rest:go through the gate on the right and hover over the stuffed animal" \
    --overwrite

echo
echo "[$(date +%H:%M:%S)] cleanup: removing ${TMP}"
rm -rf "$TMP"

echo
echo "=========================================================="
echo "[$(date +%H:%M:%S)] done → ${FINAL_OUT}"
echo "=========================================================="
ls -la "$FINAL_OUT"
echo "---"
echo "parquets:"
ls "$FINAL_OUT/data/chunk-000/" | wc -l
echo "meta:"
ls "$FINAL_OUT/meta/"
