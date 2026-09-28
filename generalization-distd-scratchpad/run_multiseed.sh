#!/usr/bin/env bash
set -euo pipefail

RESULTS_CSV=${RESULTS_CSV:-results.csv}
PREDICTIONS_CSV=${PREDICTIONS_CSV:-predictions.csv}

../venv/bin/python data/distd_scratchpad/prepare.py

for condition in no_scratchpad dummy meaningful; do
  for seed in 1337 1338 1339 1340 1341; do
    run_dir="out_${condition}_${seed}"
    ../venv/bin/python train.py config/basic.py \
      --condition="$condition" --seed="$seed" --out_dir="$run_dir"
    ../venv/bin/python evaluate.py config/basic.py \
      --out_dir="$run_dir" --seed="$seed" \
      --results_csv="$RESULTS_CSV" --predictions_csv="$PREDICTIONS_CSV"
  done
done
