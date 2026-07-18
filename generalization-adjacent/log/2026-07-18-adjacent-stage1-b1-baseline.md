# Adjacency — Stage 1: does the setup work at all? (length 6, b=1, T5)

*2026-07-18 · commit `00b0c0c` · John Lee*

> **Short version.** The model learns the task perfectly, but only **1 of 4 seeds** generalizes
> to the unseen positions. The reason is not the model and not the background — the training
> half is too small to define the rule. **Stage 2 (the b sweep) is on hold until we decide
> what to change.**

---

## The task

Is `X` immediately followed by `Y`?

```
XY0000  ->  T     (X and Y touch)
X0Y000  ->  F     (something sits between them)
```

`X` is always to the left of `Y`, so the only question is "touching, or a gap?"

`b` = how many different background characters exist. Here **b=1**, so every background slot
is `0`. (`b` is the knob this whole experiment is eventually meant to sweep, 1 → 10.)

---

## Run 1: length 6, b=1, T5, half split

**Purpose:** Check that the smallest case works before sweeping `b`. Train with X and Y only
in the **first half** (positions 0–2); test with them only in the **second half** (positions
3–5), which the model never sees during training.

**Config:** length 6, b=1, `pos_type=t5`, `half` split, 3 layers / 2 heads / 32 dim =
**38,464 params**, batch 64, lr 1e-3, 2000 iters, CPU, seeds 1337–1340.
Train 20,000 / val 2,000 / test 2,000, all 50/50 T/F, so chance = 50%.

**Results:**

| seed | did it LEARN? (val) | did it GENERALIZE? (held-out) |
|---|---|---|
| 1337 | 100% | **50%** |
| 1338 | 100% | **100%** |
| 1339 | 100% | **50%** |
| 1340 | 100% | **50%** |

- All 4 seeds learn the task instantly — 100% val by iteration 100.
- Only **1 of 4** generalizes. The other 3 answer `T` to *everything* in the unseen half.
- 50% here does **not** mean "half right". The test set is exactly 50/50 T/F, so answering
  all-`T` scores exactly 50%. Per-class accuracy shows it: T = 100%, F = 0%.

**Control (no held-out positions):** with the `none` split — X and Y anywhere — all 4 seeds
score **100%**. So the task, the model, and the engine are all fine. The problem is specific
to the held-out-position setup.

---

## Why this happened

In the first half there are only **3 possible arrangements**, and only **one** of them is F:

```
XY0000   ->  T
0XY000   ->  T
X0Y000   ->  F     <-- the ONLY F example the model ever sees
```

At b=1 the entire training set is literally these **3 strings**, repeated.

So "F" is taught by a *single* example. And several different rules explain all 3 examples
perfectly well:

- "they touch" → T  *(the rule we meant)*
- "X at position 0 and Y at position 2" → F  *(just memorizing)*
- "Y at position 2" → F

The model has no way to tell which one we meant, so **which rule it picks is basically luck**
— 1 in 4 here. The test half has the same weakness: its only F case is `000X0Y`, which is the
same shape (gap 2) as the single memorized example.

### Good news: the model reads the *gap*, not the positions

Fraction of `T` answers by gap, across all positions including untrained ones
(correct answer: `T` only for gap 1):

| seed | gap 1 | **gap 2** | gap 3 | gap 4 | gap 5 |
|---|---|---|---|---|---|
| 1337 | 1.00 ✅ | **0.75 ❌** | 0.00 ✅ | 0.00 ✅ | 0.00 ✅ |
| 1338 | 1.00 ✅ | **0.00 ✅** | 0.00 ✅ | 0.00 ✅ | 0.00 ✅ |
| 1339 | 1.00 ✅ | **0.50 ❌** | 0.00 ✅ | 0.00 ✅ | 0.00 ✅ |
| 1340 | 1.00 ✅ | **0.50 ❌** | 0.00 ✅ | 0.00 ✅ | 0.00 ✅ |

Every seed gets gaps 1, 3, 4 and 5 right **everywhere**, including position pairs never seen
in training. They disagree only on **gap 2** — the one case taught by a single example.

This is exactly what we hoped for: T5 is measuring the *distance* between X and Y, not their
absolute positions. The position shortcut the spec warned about **did not happen**.

**Output:**

<img src="figures/stage1_learnability_b1.png" alt="val accuracy vs iteration for 4 seeds, all reaching 100% by iteration 100" width="620">

<img src="figures/per_position_b1_half.png" alt="per-position accuracy heatmap, one panel per seed" width="900">

*(In the heatmap, red cells are errors. They sit on constant-gap diagonals, not in position
blocks — that is the "reads distance" signature.)*

---

## Conclusion

**What worked**

- The whole pipeline runs end to end: data → train → evaluate → figures.
- The model learns adjacency instantly, and gets 100% when training covers all positions.
- T5 reads relative distance, not absolute position — no shortcut. This is evidence *against*
  the original worry that T5 gets lost in background noise.

**What didn't work**

- Generalization to held-out positions is a **coin flip** (1 of 4 seeds), because the length-6
  training half only contains one F example.
- **Stage 1 does not pass its gate, so Stage 2 was not started.** Running the b sweep at
  length 6 would not answer anything: a drop from b=1 to b=10 would be indistinguishable from
  this seed lottery.

**Decision needed before continuing** — none of these are tested yet:

1. **Lengthen the input** (8/10/12) so each half contains several different gaps. Most obvious
   fix, but it reorders the staging agreed in the spec — worth asking the advisor.
2. **Keep length 6, change the split** — e.g. hold out one position at a time instead of a
   whole half, so training keeps more than one F arrangement.
3. **Keep length 6, run 10+ seeds per b** and treat the *rate* of generalization as the
   measurement, instead of any single run.

---

## Caveats

- **4 seeds is too few.** With a 1-of-4 outcome, the true rate could be anywhere from ~1% to
  ~81% (exact 95% interval). The coin-flip behaviour is solid; the *number* 1-in-4 is not.
- **The fix is untested.** This log explains why length 6 can't measure a background effect.
  It does not show what *can* — all three options above are guesses.
- **Only b=1 was run.** The b sweep hasn't started, so nothing here says anything yet about
  whether background diversity actually affects T5.
- Training is not perfectly stable: 3 of 4 seeds briefly dip to 50% once before recovering
  (visible as spikes in the curve). With only 3 distinct training strings that is expected,
  and best-val checkpointing means it doesn't affect the numbers above.
- Seeds change initialization and batch order only — the data split is fixed — so the real
  run-to-run variance is larger than shown.

---

## Reproduce

```bash
# main result
../venv/bin/python data/adjacent/prepare.py --b=1
for s in 1337 1338 1339 1340; do
  ../venv/bin/python train.py    config/basic.py --seed=$s
  ../venv/bin/python evaluate.py config/basic.py --seed=$s
done

# control: no held-out positions
../venv/bin/python data/adjacent/prepare.py --b=1 --split=none
for s in 1337 1338 1339 1340; do
  ../venv/bin/python train.py    config/basic.py --seed=$s
  ../venv/bin/python evaluate.py config/basic.py --seed=$s
done

# figures (restore the main data first)
../venv/bin/python data/adjacent/prepare.py --b=1
../venv/bin/python plot.py --curve --b=1 --length=6 --split=half \
    --out_dir=log/figures --out_name=stage1_learnability_b1.png
../venv/bin/python plot.py --split=half --b=1 --out_dir=log/figures
```

Raw data behind every number above: `results.csv`, `predictions.csv`, `curves.csv`.
Environment: python 3.9.6 · torch 2.8.0 · numpy 2.0.2 · matplotlib 3.9.4; data `SEED=1337`.
