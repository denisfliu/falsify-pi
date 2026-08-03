#!/usr/bin/env bash
# One-shot launcher for the 2026-06-06 left/right finetune evals.
#
# 2 policies × 4 scenarios = 8 campaigns. Each cell restricts --scenes to
# the side that matches the policy (left-policy on left-facing scenes only,
# and vice versa). Bundles are shared with prior runs (master_seed=0), so
# trial cards are byte-identical to the corresponding cells of past sweeps.
#
# Outer loop = policy to minimise bridge swaps (the moraband multi-policy
# bridge serves one policy at a time).
#
# Usage:
#   export PI_API_KEY=pi-jt-moraband-dev-001
#   source tools/env.sh
#   bash tools/run_all_left_right_eval.sh
#
# Top-level log: runs/eval_campaigns/all_left_right_<ts>.log

set -uo pipefail
trap '' SIGPIPE
cd "$(dirname "$0")/.."

STAMP=$(date +%Y%m%d_%H%M%S)
LOG="runs/eval_campaigns/all_left_right_${STAMP}.log"
mkdir -p "$(dirname "$LOG")"

# Per (policy, scenario) -> space-separated scene_keys for --scenes.
# Bash assoc-array keyed by "policy|scenario".
declare -A FILTERS=(
  ["nonhistory_all_left|pure"]="left_gate center_gate_from_left"
  ["nonhistory_all_left|gate_perturbed_small"]="left_gate"
  ["nonhistory_all_left|gate_perturbed_large"]="left_gate"
  ["nonhistory_all_left|compositional"]="left_and_center"

  ["nonhistory_all_right|pure"]="right_gate center_gate_from_right"
  ["nonhistory_all_right|gate_perturbed_small"]="right_gate"
  ["nonhistory_all_right|gate_perturbed_large"]="right_gate"
  ["nonhistory_all_right|compositional"]="right_and_center"
)

POLICIES=(nonhistory_all_left nonhistory_all_right)
SCENARIOS=(pure gate_perturbed_small gate_perturbed_large compositional)

exec >>"$LOG" 2>&1
echo "[$(date +%H:%M:%S)] all-left/right eval launch — log $LOG"
echo "policies: ${POLICIES[*]}"
echo "scenarios: ${SCENARIOS[*]}"
echo

TOTAL=$(( ${#POLICIES[@]} * ${#SCENARIOS[@]} ))
N=0
for POLICY in "${POLICIES[@]}"; do
  for SCENARIO in "${SCENARIOS[@]}"; do
    N=$((N + 1))
    SCENES_STR="${FILTERS["${POLICY}|${SCENARIO}"]}"
    # shellcheck disable=SC2206
    SCENES_ARR=(${SCENES_STR})

    echo
    echo "============================================================"
    echo "[$(date +%H:%M:%S)] [${N}/${TOTAL}] policy=${POLICY} scenario=${SCENARIO} scenes=${SCENES_STR}"
    echo "============================================================"

    PYTHONPATH=src python scripts/eval/run_eval_campaign.py \
        --scenario "configs/eval_suite/${SCENARIO}.yaml" \
        --policy-config "configs/policies/pi_gateway/${POLICY}.yaml" \
        --frame configs/frames/carl_dual.yaml \
        --scenes "${SCENES_ARR[@]}" \
        --no-rtc --no-recovery --skip-flythrough --no-gripper-overlay
    RC=$?
    if [ "$RC" -ne 0 ]; then
      echo "[$(date +%H:%M:%S)] [WARN] cell exited rc=${RC}; continuing."
    fi
  done
done

echo
echo "=== all-left/right eval finished $(date) ==="
