#!/usr/bin/env bash
# Train and evaluate conditions x 5 seeds on the counting task.
# Step 1 (default): NoPE, no scratchpad, length 20.
set -euo pipefail

RESULTS_CSV=${RESULTS_CSV:-results_step1_nope.csv}
PREDICTIONS_CSV=${PREDICTIONS_CSV:-predictions_step1_nope.csv}
POS_TYPE=${POS_TYPE:-none}
RUN_PREFIX=${RUN_PREFIX:-out_step1_nope}
CONDITIONS=${CONDITIONS:-no_scratchpad}
DATA_DIR=${DATA_DIR:-data/count_scratchpad}

if [[ -e "$RESULTS_CSV" || -e "$PREDICTIONS_CSV" ]]; then
  echo "Refusing to append to an existing result file. Set new RESULTS_CSV and PREDICTIONS_CSV paths."
  exit 1
fi

for condition in $CONDITIONS; do
  for seed in 1337 1338 1339 1340 1341; do
    run_dir="${RUN_PREFIX}_${condition}_${seed}"
    ../venv/bin/python train.py config/basic.py \
      --condition="$condition" --seed="$seed" --out_dir="$run_dir" \
      --pos_type="$POS_TYPE" --data_dir="$DATA_DIR"
    for iteration in 500 1000 1500 2000; do
      checkpoint_name=$(printf 'ckpt_iter%04d.pt' "$iteration")
      log_predictions=False
      if [[ "$iteration" -eq 2000 ]]; then
        log_predictions=True
      fi
      ../venv/bin/python evaluate.py config/basic.py \
        --out_dir="$run_dir" --checkpoint_name="$checkpoint_name" \
        --data_dir="$DATA_DIR" \
        --results_csv="$RESULTS_CSV" --predictions_csv="$PREDICTIONS_CSV" \
        --log_predictions="$log_predictions"
    done
  done
done
