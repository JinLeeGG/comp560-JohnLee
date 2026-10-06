#!/usr/bin/env bash
set -euo pipefail

RESULTS_CSV=${RESULTS_CSV:-results_multiseed.csv}
PREDICTIONS_CSV=${PREDICTIONS_CSV:-predictions_multiseed.csv}
POS_TYPE=${POS_TYPE:-none}
RUN_PREFIX=${RUN_PREFIX:-out_formal}

if [[ -e "$RESULTS_CSV" || -e "$PREDICTIONS_CSV" ]]; then
  echo "Refusing to append to an existing result file. Set new RESULTS_CSV and PREDICTIONS_CSV paths."
  exit 1
fi

../venv/bin/python data/distd_scratchpad/prepare.py

for condition in no_scratchpad dummy meaningful; do
  for seed in 1337 1338 1339 1340 1341; do
    run_dir="${RUN_PREFIX}_${condition}_${seed}"
    ../venv/bin/python train.py config/basic.py \
      --condition="$condition" --seed="$seed" --out_dir="$run_dir" \
      --pos_type="$POS_TYPE"
    for iteration in 500 1000 1500 2000; do
      checkpoint_name=$(printf 'ckpt_iter%04d.pt' "$iteration")
      log_predictions=False
      if [[ "$iteration" -eq 2000 ]]; then
        log_predictions=True
      fi
      ../venv/bin/python evaluate.py config/basic.py \
        --out_dir="$run_dir" --checkpoint_name="$checkpoint_name" \
        --results_csv="$RESULTS_CSV" --predictions_csv="$PREDICTIONS_CSV" \
        --log_predictions="$log_predictions"
    done
  done
done
