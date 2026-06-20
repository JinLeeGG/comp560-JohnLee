# Relative-Order Task — held-out positions (the phenomenon)

*2026-06-20 · Phase 2 → motivates the Phase 3 PE sweep*

> **Bottom line.** Tested at symbol positions **never seen in training**, `none` (no positional
> encoding) still scores **~100%**, while `learned` (absolute PE) **collapses to ~58%** — barely
> above chance — even though *both* hit 100% on in-distribution validation.
> **Removing positional encoding *helps* generalization here**, the opposite of the naive guess.

### The task (one line)

Length-20 string with **one `X`** and **one `Y`** (rest are random digits). Output the last token:

```
T  if  X is before Y        ...X...Y...  : T
F  if  Y is before X        ...Y...X...  : F      rule (fixed): T ⇔ index(X) < index(Y)
```

The two settings we compare (only this changes between runs):
**`none` = NoPE** (no positional encoding) · **`learned` = APE** (learned absolute position embedding).

### What "held out" means here

On the full distribution (previous log) both settings scored 100%, so we now restrict **where**
`X`/`Y` may appear and test on positions the model never trained on:

| split | train / val sees | test = held out (never trained) |
|-------|------------------|---------------------------------|
| `single_pos` (P=12) | `X`,`Y` never at position 12 | exactly one of `X`/`Y` at position 12 |
| `half` | both `X`,`Y` in the **first** half (0–9) | both `X`,`Y` in the **second** half (10–19) |

Every pool is 50/50 T/F (chance = 50%). `prepare.py` asserts both the label rule and the split
rule on every example before writing data — all passed.

---

## Results at a glance

Both settings behave **identically** until the hard split, where `learned` breaks:

| split | `none` held-out | `learned` held-out |
|-------|-----------------|--------------------|
| full distribution (baseline) | 100% | 100% |
| `single_pos` (hold out 1 position) | 100% | 100% |
| **`half` (hold out 2nd half)** | **≈100%** | **≈58%** ← breaks |

The `half` split is the result. Per model-init seed (held-out test accuracy):

| seed | `none` | `learned` (per-class T / F) |
|------|--------|------------------------------|
| 1337 | 100%   | 69%  (T 38% / F 100%) |
| 1338 | 99.9%  | 50%  (T 100% / F 0%) |
| 1339 | 100%   | 61.5% (T 23% / F 100%) |
| 1340 | 100%   | 51.25% (T 2.5% / F 100%) |
| **mean** | **≈100%** | **≈58%** |
| in-dist **val** | 100% (all) | 100% (all) |

Two things to read off this table (*val* = accuracy on trained positions; *test* = accuracy on the
held-out positions):
1. **Both learn the task** (val = 100%); only `none` **generalizes** to the held-out positions.
2. **`learned` collapses to one label** — look at its T/F columns: each seed is near 0/100 or
   100/0, and *which* label it defaults to flips by seed. So ~58% is a collapse, not a clean 50%.

> ⚠️ **Caveat on these numbers.** The 4 seeds vary **model init / batch order only**, on **one
> fixed data split** (seed-1337 data). So the spread reflects initialization noise, **not**
> data-sampling noise, and understates true variance. The gap (100 vs 58) is far larger than that
> noise, so the conclusion holds — but the stronger check is to regenerate the split per seed (see
> *Next*).

---

## Figures

From `results.csv` / `predictions.csv` via `plot.py` (committed under [`figures/`](figures/);
regenerate with `../venv/bin/python plot.py --split=half --out_dir=log/figures`).

**1 — Held-out accuracy by method (the conclusion).** `none` at 100%, `learned` near the chance
line; both in-distribution val bars at 100%, so the gap is a *generalization* failure, not a
training failure.

<img src="figures/heldout_accuracy_half.png" alt="held-out accuracy: none 100% vs learned ~58% at the 50% chance line; both val 100%" width="620">

**2 — Per-position accuracy (the *why*).** Each cell = accuracy with `X` at that row, `Y` at that
column, over a sweep of all position pairs. `none` is green everywhere. `learned` is green in the
trained block (top-left, both in first half) but in the held-out block (bottom-right) it **fails
wherever the answer is `T`** (the red upper triangle, X before Y) while still getting `F` cases — i.e.
off the trained region it stops distinguishing order and falls back to one label.

<img src="figures/per_position_half.png" alt="per-position heatmap: none green everywhere; learned green in trained block, red in held-out block" width="820">

*(A per-class T/F bar chart is also produced (`heldout_perclass_half.png`) but not embedded — the
heatmap shows the label collapse more directly.)*

---

## Details

### Run 1 — `single_pos` (hold out one position): too easy
**Purpose.** Direct parallel to the detection task's position-12 hold-out.
**Config.** Baseline model (~0.8M params), 2000 iters, **one seed (1337)**.
**Result.** Both settings = **100%** (T and F) at the held-out position. A single held-out slot
generalizes trivially, same as detection. Saturated on one seed, so no sweep. → need a harder split.

### Run 2 — `half` (hold out the whole second half): the divergence
**Purpose.** The hard test: special symbols are seen **only** in the first half during training.
**Config.** Baseline model; **4 seeds (1337–1340)**. Each half has 10 positions = 90 ordered
(X,Y) pairs — enough to learn the task in-distribution.
**Result.** See *Results at a glance*. `none` ≈100%, `learned` ≈58% (collapse to one label).
**Output (seed 1337):**
```
none     val 100% | test:  T 100%  F 100%  → 100%
learned  val 100% | test:  T  38%  F 100%  →  69%
         (every learned error is gold=T predicted F, with X and Y both in the 2nd half)
```

### Why this happens
- **`none` (NoPE)** can only use the causal mask's *relative* order — "have I passed an `X`
  before reaching `Y`?" That is position-agnostic, so a circuit learned on the first half works
  unchanged on the second. → generalizes.
- **`learned` (APE)** *does* have trained position vectors for slots 10–19 (digits sit there in
  training), but it appears to tie the order computation to the **absolute positions where `X`/`Y`
  appeared** — the first half. Off-distribution it misfires and defaults to one label. → chance.
- This last point is a **hypothesis** from the behavior + heatmap, not yet proven; Phase 7
  interpretability is what would confirm it.

### Relation to prior work
This reproduces the NoPE-vs-APE finding of **Kazemnejad et al. (2023)**, *The Impact of Positional
Encoding on Length Generalization in Transformers* (NeurIPS 2023; abstract verified). They show it
for **length generalization** — testing on *longer* sequences. Here it appears in a **fixed-length,
held-out-position** setting: the shift is over *where* a symbol sits within a fixed length, not over
length. That fixed-length angle is the contribution.

---

## Next
1. **Phase 3 — the PE sweep.** Implement `sinusoidal` / `rope` / `t5` and run all five on the
   `half` split. Hypothesis: relative PEs (RoPE, T5) generalize like `none`; the absolute one
   (sinusoidal) fails like `learned`.
2. **Strengthen the sweep** — regenerate the data split per seed, so error bars include
   data-sampling variance (addresses the caveat above).
3. **Phase 7 interpretability** — test the absolute-position-reliance story directly.
4. Optional: a **distance-based** held-out split as a second generalization axis.
