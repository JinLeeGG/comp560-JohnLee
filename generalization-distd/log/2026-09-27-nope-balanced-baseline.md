# dist>=5 — Clean NoPE baseline with matched distance distributions

*2026-09-27 · based on commit `7afff2a` plus the changes recorded here · John Lee*

> **Main takeaway.** After matching the train, validation, and test distance
> distributions, NoPE learned the in-distribution task at 100% in all five seeds but
> remained near chance on held-out positions (mean 50.98%). The generalization failure is
> therefore not explained by the historical train/test distance-frequency mismatch.

## Question

The historical `dist>=5` dataset sampled random position pairs within each class for
train/validation but balanced individual distances within each class for test. This run
removes that extra distribution shift before choosing the task as a scratchpad baseline.

Task: a fixed-length 20-token input contains exactly two `X` symbols. Output `T` when
their position difference is at least 5 and `F` otherwise.

## Setup

- **Split:** train/validation place both `X` symbols in positions 0--9; held-out test
  places both in positions 10--19.
- **Distance distribution:** identical in every pool: each near distance 1--4 has
  12.5% of examples and each far distance 5--9 has 10%.
- **Sizes:** 50,000 train; 5,000 in-distribution validation; 2,000 held-out test.
- **Class balance:** exactly 50% `T` / 50% `F` in every pool; chance = 50%.
- **Model:** NoPE causal micro-transformer; 4 layers, 4 heads, embedding width 128,
  approximately 0.80M parameters.
- **Training:** 2,000 iterations; answer-token-only loss; CPU.
- **Model seeds:** 1337, 1338, 1339, 1340, 1341. Data are fixed at seed 1337, so this
  measures model-initialization and batch-order variance only.
- **Environment:** Python 3.9.6; PyTorch 2.8.0; NumPy 2.0.2.

Reproduce from `generalization-distd/`:

```bash
TRAIN_DISTANCE_SAMPLING=balanced OUTPUT_DIR=data/distd_balanced \
  ../venv/bin/python data/distd/prepare.py

for s in 1337 1338 1339 1340 1341; do
  ../venv/bin/python train.py config/nope_balanced.py --seed=$s
  ../venv/bin/python evaluate.py config/nope_balanced.py \
    --seed=$s \
    --results_csv=results_nope_balanced_20260927.csv \
    --predictions_csv=predictions_nope_balanced_20260927.csv \
    --show_errors=0
done
```

## Results

| Seed | ID validation | Held-out overall | Far `T` | Near `F` |
|---:|---:|---:|---:|---:|
| 1337 | 100.0% | 50.0% | 0.0% | 100.0% |
| 1338 | 100.0% | 50.0% | 0.0% | 100.0% |
| 1339 | 100.0% | 50.0% | 0.0% | 100.0% |
| 1340 | 100.0% | 50.0% | 0.0% | 100.0% |
| 1341 | 100.0% | 54.9% | 9.8% | 100.0% |
| **Mean +/- sample SD** | **100.0 +/- 0.0%** | **50.98 +/- 2.19%** | **1.96 +/- 4.38%** | **100.0 +/- 0.0%** |

The model did not fail to learn the task: every seed achieved 100% on held-in positions.
On unseen positions, however, every seed predicted almost all examples as near (`F`).

## Interpretation

This is a suitable baseline for a scratchpad pilot: the task is learnable in-distribution,
but NoPE does not reliably carry the learned rule to the held-out half. This does **not**
prove that NoPE is theoretically incapable of solving the task. It establishes a robust
empirical generalization gap for this model, training setup, and split.

## Caveats

- Five seeds are enough for a pilot but not a definitive statistical claim.
- All runs use one fixed data sample; a later confirmatory experiment should pair conditions
  within several data seeds as well as model seeds.
- NoPE has no explicit positional encoding, but the causal mask still supplies an ordering
  signal.
- The aggregate test is class- and distance-balanced. Per-position diagnostics are still
  needed before making a mechanistic claim about what the model learned.

## Next

Compare no scratchpad, dummy scratchpad, and meaningful distance-state scratchpad under
this exact NoPE dataset and model configuration, using paired seeds.
