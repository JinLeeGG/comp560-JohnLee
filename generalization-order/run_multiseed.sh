#!/usr/bin/env bash
# Multi-seed PE sweep with per-seed data-split regeneration (half split).
#
# Motivation: the original 2026-06-20 sweep used 4 seeds on ONE fixed data split
# (SEED=1337 in prepare.py), so its spread reflected model-init/batch-order noise
# only. Here each seed regenerates the train/val/test split (DATA_SEED=$seed), so
# the reported variance also includes data-sampling variance. 10 seeds.
#
# Within a seed, all 5 PEs share that seed's data (PE is the only variable);
# across seeds, both data and model init vary.
#
# Results are logged to NEW csv files so the original results.csv is untouched:
#   results_multiseed.csv / predictions_multiseed.csv
#
# Run from generalization-order/ :  bash run_multiseed.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=../venv/bin/python

SEEDS="1337 1338 1339 1340 1341 1342 1343 1344 1345 1346"
PES="none learned sinusoidal rope t5"
RESULTS=results_multiseed.csv
PREDS=predictions_multiseed.csv

# start clean so a re-run does not append to stale rows
rm -f "$RESULTS" "$PREDS"

for seed in $SEEDS; do
  echo "=================== DATA_SEED=$seed : regenerating split ==================="
  DATA_SEED=$seed $PY data/order/prepare.py | tail -1
  for pe in $PES; do
    echo "--- seed=$seed pe=$pe : train ---"
    $PY train.py config/basic.py --pos_type="$pe" --seed="$seed" | tail -1
    echo "--- seed=$seed pe=$pe : eval ---"
    $PY evaluate.py config/basic.py --seed="$seed" --show_errors=0 \
        --results_csv="$RESULTS" --predictions_csv="$PREDS" | tail -1
  done
done

echo "=================== DONE : wrote $RESULTS / $PREDS ==================="
