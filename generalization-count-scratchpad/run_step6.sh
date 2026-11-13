#!/usr/bin/env bash
# Step 6: one given '#' before vs. after a fixed 40-character body (NoPE).
# Run with: bash run_step6.sh
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

RESULTS_CSV=${RESULTS_CSV:-results_step6_nope.csv}
PREDICTIONS_CSV=${PREDICTIONS_CSV:-predictions_step6_nope.csv}
RUN_PREFIX=${RUN_PREFIX:-out_step6_nope}

if [[ -e "$RESULTS_CSV" || -e "$PREDICTIONS_CSV" ]]; then
  echo "Refusing to overwrite or append to existing Step 6 results."
  exit 1
fi
for condition in prefix_one suffix_one; do
  for seed in 1337 1338 1339 1340 1341; do
    if [[ -e "${RUN_PREFIX}_${condition}_${seed}" ]]; then
      echo "Refusing to overwrite existing run ${RUN_PREFIX}_${condition}_${seed}."
      exit 1
    fi
  done
done

# Reuse the exact Step 5 raw data and union vocabulary; do not regenerate them.
DATA_DIR=data/count_scratchpad_L40_seen10_step5 \
BLOCK_SIZE=96 POS_TYPE=none CONDITIONS="prefix_one suffix_one" \
RUN_PREFIX="$RUN_PREFIX" RESULTS_CSV="$RESULTS_CSV" \
PREDICTIONS_CSV="$PREDICTIONS_CSV" bash run_multiseed.sh
