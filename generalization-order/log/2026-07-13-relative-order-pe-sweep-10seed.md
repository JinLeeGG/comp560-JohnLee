# Relative-Order PE Sweep — 10-seed reinforced run (per-seed data regeneration)

*2026-07-13 · John Lee*

> **Bottom line.** Re-run of the [Phase 3 PE sweep](2026-06-20-relative-order-pe-sweep.md)
> with the statistics strengthened: **10 seeds** instead of 4, and each seed **regenerates the
> data split** (not one fixed split). The relative-vs-absolute separation holds in the mean —
> relative (`none` 99.2, `t5` 99.2, `rope` 95.7) sits far above absolute (`learned` 68.6,
> `sinusoidal` 31.9) — but the stronger run reveals variance the 4-seed run hid: **`rope` is
> not a clean 100%** (95.7 ± 6.8, one seed 79.6), and **`learned` occasionally generalizes
> fully** (68.6 ± 18.1, one seed 100%). All five reach **100% in-distribution val**.

### Why re-run

The 2026-06-20 sweep used 4 seeds on **one fixed data split** (`SEED=1337` in `prepare.py`), so
its spread was model-init / batch-order noise only and understated true variance (its own caveat).
This run fixes that: each seed regenerates the train/val/test split, so the reported spread now
includes **data-sampling variance**. Same task and `half` split (length-20, one `X` + one `Y`,
output `T` iff index(X) < index(Y); train both symbols in 0–9, test both in 10–19; 50/50, chance 50%).

### Setup

- **What changed vs 2026-06-20:** `prepare.py` `SEED` now reads env `DATA_SEED` (default 1337, so
  old behavior is unchanged). Driver `run_multiseed.sh` loops `for seed in 1337..1346: regenerate
  data with DATA_SEED=$seed; for pe in none learned sinusoidal rope t5: train --seed=$seed; eval`.
  Within a seed all 5 PEs share that seed's data (PE is the only variable); across seeds both data
  and init vary.
- **model / config:** unchanged — ~0.8M params (n_layer 4, n_head 4, n_embd 128, block_size 64),
  causal, CPU, 2000 iters, AdamW lr 1e-3, batch 64, answer-token-only loss.
- **logging:** new files `results_multiseed.csv` / `predictions_multiseed.csv` so the original
  `results.csv` is untouched.
- **reproduce** (from `generalization-order/`, `SPLIT='half'` in `prepare.py`):
  ```bash
  bash run_multiseed.sh
  ```
- **integrity check:** seed 1337 (same data seed as the original run) reproduced the published
  4-seed values exactly (`rope` 99.7, `learned` 69.0, `sinusoidal` 30.6).

---

## Results — `half` split, 10 seeds each

| PE | family | val | held-out (mean ± std) | min–max |
|----|--------|-----|-----------------------|---------|
| `none` | relative | 100% | **99.2 ± 1.7** | 94.7–100 |
| `t5` | relative | 100% | **99.2 ± 2.1** | 92.9–100 |
| `rope` | relative | 100% | **95.7 ± 6.8** | **79.6**–100 |
| `learned` | absolute | 100% | **68.6 ± 18.1** | **50–100** |
| `sinusoidal` | absolute | 100% | **31.9 ± 6.9** | 19.2–42.7 |

Per seed (held-out test):

| seed | `none` | `rope` | `t5` | `learned` | `sinusoidal` |
|------|--------|--------|------|-----------|--------------|
| 1337 | 100.0 |  99.7 | 100.0 |  69.0 | 30.6 |
| 1338 |  97.5 |  92.8 | 100.0 |  50.0 | 26.4 |
| 1339 | 100.0 | 100.0 | 100.0 |  89.7 | 36.6 |
| 1340 | 100.0 | 100.0 | 100.0 |  69.1 | 40.6 |
| 1341 |  94.7 |  79.6 | 100.0 | 100.0 | 33.2 |
| 1342 | 100.0 | 100.0 |  92.9 |  50.0 | 35.4 |
| 1343 | 100.0 |  86.5 | 100.0 |  92.7 | 29.1 |
| 1344 | 100.0 |  98.3 |  99.3 |  60.0 | 42.7 |
| 1345 | 100.0 | 100.0 | 100.0 |  50.0 | 24.6 |
| 1346 | 100.0 | 100.0 | 100.0 |  55.6 | 19.2 |
| **mean** | **99.2** | **95.7** | **99.2** | **68.6** | **31.9** |

---

## Why

- **Family separation holds in the mean.** Relative (96–99) vs absolute (32–69); the gap (≈27–67
  points) dwarfs the standard error of any mean, so the headline "relative predicts generalization"
  is safe.
- **But it is not a clean binary per seed** (the 4-seed run's "clean split" was partly an artifact
  of one lucky data split):
  - `rope` is the least stable relative method — 95.7 ± 6.8, dipping to 79.6 on seed 1341.
  - `learned` is highly variable — 50 (pure chance) up to 100 on one seed, with two more above 89.
- **The two absolute PEs fail differently, and the mechanism is now clearer.** `sinusoidal`'s
  position vectors for indices 10–19 are **fixed** (never trained) → no usable held-out signal →
  consistently below chance. `learned`'s 10–19 vectors **do** get gradient updates — the filler
  **digits** occupy those slots in every example, just never `X`/`Y` — so whether they develop
  usable features is init/data-dependent → the large variance. This asymmetry is invisible in a
  length-generalization setting (Kazemnejad et al. 2023), where held-out positions never exist
  during training at all.

## Caveats / limitations

- 10 seeds pin down the **family separation** comfortably, but not the **`learned` mean** precisely
  (std 18 → SEM ≈ 5.7). Treat `learned`'s occasional full generalization as a qualitative
  observation, not a quantified rate (that would need ~30+ seeds).
- Single task, single split, single fixed length — the relative/absolute line is not yet shown to
  hold across position-dependent tasks in general.
- The held-out-embedding mechanism above is consistent with the accuracy pattern but not directly
  verified; confirming it is a job for mechanistic analysis, not accuracy numbers.

## Next

- Regenerate the report figures from `results_multiseed.csv` (current committed PNGs are the 4-seed
  version and no longer match these numbers).
- Optionally add seeds for the two absolute conditions only, to tighten their error bars.
