#!/usr/bin/env bash
# Drive a 4×4 eval sweep: 4 nonhistory policies × 4 scenarios.
# Outer loop = policy (minimises bridge swaps; bridge serves one policy
# at a time anyway). Inner loop = scenario.
#
# All per-campaign artifacts for this launch land under a shared
# sweep folder so the four policies stay aligned for A/B comparison:
#
#   runs/eval_campaigns/<policy_id>/sweep-<NNN>-<ts>[-<tag>]/run-NNN-<scenario>-<ts>/
#
# Sweep number is computed by scanning existing `sweep-*` dirs across ALL
# policy folders and taking max+1 — so a single launch produces folders
# named identically under each policy. An optional `--tag <label>` argument
# is appended to the sweep dir name for human-readable annotation
# (e.g. `--tag goal-fix` → `sweep-002-20260521_103408-goal-fix`).
#
# Top-level driver log lives at runs/eval_campaigns/sweep_<ts>.log.
# A `sweep_manifest.json` is written into each policy's sweep folder
# capturing the policy list, scenario list, tag, and start time.
#
# Run from repo root with PI_API_KEY set:
#   export PI_API_KEY="pi-jt-moraband-dev-001"
#   source tools/env.sh
#   bash tools/run_eval_sweep.sh                 # untagged sweep
#   bash tools/run_eval_sweep.sh --tag goal-fix  # tagged sweep
#
# Use `tail -f runs/eval_campaigns/sweep_*.log` to monitor.

set -uo pipefail
# SIGPIPE guard: if an outer pipe is closed (e.g. `bash sweep.sh | head -2`),
# write failures should NOT terminate the loop.
trap '' SIGPIPE
cd "$(dirname "$0")/.."

TAG=""
SCENARIOS_OVERRIDE=()
TRIALS_OVERRIDE=()
while [ $# -gt 0 ]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    # Override the default SCENARIOS list (one or more, space-separated):
    #   --scenarios pure gate_perturbed_small
    --scenarios)
      shift
      while [ $# -gt 0 ] && [[ ! "$1" =~ ^-- ]]; do
        SCENARIOS_OVERRIDE+=("$1"); shift
      done
      ;;
    # Restrict each campaign to a specific trial-index subset. Passed
    # through to run_eval_campaign.py --trials, applied uniformly to
    # every (policy, scenario) cell. Use to smoke-test changes without
    # paying for the full ~60-trial-per-scenario sweep:
    #   --trials 0 1
    --trials)
      shift
      while [ $# -gt 0 ] && [[ ! "$1" =~ ^-- ]]; do
        TRIALS_OVERRIDE+=("$1"); shift
      done
      ;;
    -h|--help) sed -n '2,28p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

POLICIES=(
  nonhistory_all_93sufwik_7500
  nonhistory_center_g3jt73md_3000
  nonhistory_real_center
  nonhistory_real_synth_31ohxgxv_5000
)

SCENARIOS=(
  pure
  gate_perturbed_small
  gate_perturbed_large
  compositional
)
# Override with --scenarios if provided.
if [ "${#SCENARIOS_OVERRIDE[@]}" -gt 0 ]; then
  SCENARIOS=("${SCENARIOS_OVERRIDE[@]}")
fi

STAMP=$(date +%Y%m%d_%H%M%S)
LOG="runs/eval_campaigns/sweep_${STAMP}.log"
mkdir -p "$(dirname "$LOG")"

# Compute the next sweep number by scanning existing sweep-NNN-* dirs
# across all policy folders. Using max+1 across policies (not per-policy)
# guarantees a single launch produces identical dir names under each
# policy, even if past sweeps were partial.
SWEEP_N=1
for POL in "${POLICIES[@]}"; do
  for D in runs/eval_campaigns/"${POL}"/sweep-*; do
    [ -d "$D" ] || continue
    BN=$(basename "$D")
    N_PART=${BN#sweep-}; N_PART=${N_PART%%-*}
    [[ "$N_PART" =~ ^[0-9]+$ ]] || continue
    if [ "$N_PART" -ge "$SWEEP_N" ]; then SWEEP_N=$((N_PART + 1)); fi
  done
done

SWEEP_DIR_NAME="sweep-$(printf '%03d' "$SWEEP_N")-${STAMP}"
if [ -n "$TAG" ]; then SWEEP_DIR_NAME="${SWEEP_DIR_NAME}-${TAG}"; fi

# Redirect ALL script output to the log file from this point on.
exec >>"$LOG" 2>&1
echo "[$(date +%H:%M:%S)] sweep started — tail this log for live progress"
echo "sweep_dir_name: ${SWEEP_DIR_NAME}"
echo "policies: ${POLICIES[*]}"
echo "scenarios: ${SCENARIOS[*]}"
echo "tag: ${TAG:-(none)}"
echo
echo "[sweep] log: $LOG" >/dev/tty 2>/dev/null || true
echo "[sweep] folder: ${SWEEP_DIR_NAME}" >/dev/tty 2>/dev/null || true

# Pre-create each policy's sweep folder and drop a manifest so a future
# reader (or the grid plotter) can identify which campaigns shipped
# together — independent of timestamps that drift across campaigns.
for POL in "${POLICIES[@]}"; do
  SWEEP_PATH="runs/eval_campaigns/${POL}/${SWEEP_DIR_NAME}"
  mkdir -p "$SWEEP_PATH"
  python -c "
import json, time
manifest = {
    'sweep_number': $SWEEP_N,
    'sweep_dir_name': '$SWEEP_DIR_NAME',
    'started_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    'tag': '$TAG' or None,
    'policy_id': '$POL',
    'cohort_policies': ${POLICIES[@]@Q},
    'scenarios': ${SCENARIOS[@]@Q},
    'driver_log': '$LOG',
}
with open('$SWEEP_PATH/sweep_manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)
" 2>/dev/null || true
done

TOTAL=$(( ${#POLICIES[@]} * ${#SCENARIOS[@]} ))
N=0

for POLICY in "${POLICIES[@]}"; do
  RUN_N=0
  SWEEP_PATH="runs/eval_campaigns/${POLICY}/${SWEEP_DIR_NAME}"
  for SCENARIO in "${SCENARIOS[@]}"; do
    N=$((N + 1))
    RUN_N=$((RUN_N + 1))
    TS=$(date +%Y%m%d_%H%M%S)
    OUT_DIR="${SWEEP_PATH}/run-$(printf '%03d' "$RUN_N")-${SCENARIO}-${TS}"

    echo
    echo "============================================================"
    echo "[$(date +%H:%M:%S)] [${N}/${TOTAL}] policy=${POLICY} scenario=${SCENARIO}"
    echo "                 → ${OUT_DIR}"
    echo "============================================================"

    TRIALS_ARGS=()
    if [ "${#TRIALS_OVERRIDE[@]}" -gt 0 ]; then
      TRIALS_ARGS=(--trials "${TRIALS_OVERRIDE[@]}")
    fi
    PYTHONPATH=src python scripts/run_eval_campaign.py \
        --scenario "configs/eval_suite/${SCENARIO}.yaml" \
        --policy-config "configs/policies/pi_gateway/${POLICY}.yaml" \
        --frame configs/frames/carl_dual.yaml \
        --out "$OUT_DIR" \
        --no-recovery --skip-flythrough "${TRIALS_ARGS[@]}"
    RC=$?
    if [ "$RC" -ne 0 ]; then
      echo "[$(date +%H:%M:%S)] [WARN] campaign exited rc=${RC}; continuing."
    fi
  done
done

# Patch finished_at into each sweep_manifest.
for POL in "${POLICIES[@]}"; do
  MF="runs/eval_campaigns/${POL}/${SWEEP_DIR_NAME}/sweep_manifest.json"
  [ -f "$MF" ] || continue
  python -c "
import json, time
with open('$MF') as f: m = json.load(f)
m['finished_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
with open('$MF','w') as f: json.dump(m, f, indent=2)
" 2>/dev/null || true
done

echo
echo "=== sweep finished $(date) ==="
echo "    sweep folder: ${SWEEP_DIR_NAME}"
