# Generalization Experiment: Relative-Order Task
**Date:** June 20, 2026

This experiment trains a micro-transformer (<1M params) on a **relative-order** task
and (later) measures whether it generalizes to symbol *position pairs* it never saw
during training. It reuses the from-scratch swappable-PE engine built for the
detection task (`model.py` + `pos_encoding.py`); only the data and config are
task-specific.

> **Critical framing — this is NOT length generalization.** Every input is exactly
> `LENGTH = 20` characters. The distribution shift (later) is over *where* `X` and
> `Y` appear within that fixed length, never over sequence length.

---

## Task

A fixed-length string of digits contains the symbol `X` **exactly once** and the
symbol `Y` **exactly once** (the other 18 characters are random digits). The model
reads the string, sees `:`, and must output a **single** token:

```
LABEL MAPPING (fixed — never flip):  T  iff  index(X) < index(Y)

4829X017364Y19285746:T     ← X at 4, Y at 11 → X before Y → T
4829Y017364X19285746:F     ← Y at 4, X at 11 → Y before X → F
```

Unlike detection (*"is X present?"*, position-invariant), the answer here **depends
on position**. Vocab (size 16): digits `0`–`9`, plus `X`, `Y`, `T`, `F`, `:`, `\n`.

Classes are balanced 50/50 (T/F), so chance accuracy is 50%.

---

## The position split (the experiment knob)

`SPLIT` in [data/order/prepare.py](data/order/prepare.py) controls **where X and Y may
appear**, generalizing detection's hold-out to *two* symbols:

| `SPLIT`       | train/val rule                 | test rule                          |
|---------------|--------------------------------|------------------------------------|
| `none`        | X,Y anywhere (full dist.)      | X,Y anywhere (in-distribution)     |
| `single_pos`  | neither X nor Y at `P` (=12)   | exactly one of X/Y at `P`          |
| `half`        | both X,Y in first half (0–9)   | both X,Y in second half (10–19)    |

All pools stay 50/50 T/F (chance = 50%); `prepare.py` asserts the label mapping **and**
the split rule on every example before writing the bins.

## Headline result — *relative* PE generalizes, *absolute* PE doesn't

5-way PE sweep on the `half` split (held out the 2nd half), 4 seeds each. Every method reaches
**100% in-distribution val** (all *learn* the task); they split sharply on the held-out test, and
the split is exactly **relative vs absolute**:

| PE | family | held-out test |
|----|--------|---------------|
| `none` (NoPE) | relative (causal mask) | **≈100%** |
| `rope` | relative | **≈100%** |
| `t5` | relative | **100%** |
| `learned` | absolute | **≈58%** (≈ chance) |
| `sinusoidal` | absolute | **≈33%** (below chance) |

| `SPLIT` | result |
|---------|--------|
| `none` (full dist.) | all methods 100% |
| `single_pos` (P=12) | trivial — 100% (one held-out slot generalizes for free) |
| `half` (2nd half held out) | **relative ≈100% vs absolute ≤58%** (table above) |

The relative family expresses the answer through *relative* offsets (causal mask / rotation /
relative bias), so a rule learned in the first half transfers to the second. The absolute family
ties the computation to *absolute* slot identities seen in training, so off-distribution it
misfires and collapses to one label. This **reverses the naive "PE helps" intuition** — here the
*absolute-position representation* is exactly what breaks generalization.

This reproduces — across a whole PE family — the NoPE-vs-absolute-PE finding of **Kazemnejad et
al. (2023)**, *The Impact of Positional Encoding on Length Generalization in Transformers*
(NeurIPS 2023), who report it for **length generalization** (longer sequences); here it appears in
a **fixed-length, held-out-position** setting — the shift is over symbol *position* within a fixed
length, not over length.
See [log/2026-06-20-relative-order-pe-sweep.md](log/2026-06-20-relative-order-pe-sweep.md).

<img src="log/figures/heldout_accuracy_half.png" alt="held-out accuracy by PE: none/rope/t5 ~100%, learned ~58%, sinusoidal ~33%; all val 100%" width="900">

*(the per-position heatmap `log/figures/per_position_half.png` shows the mechanism — see the
[held-out log](log/2026-06-20-relative-order-heldout-splits.md).)*

> **Why `none` solves even the full distribution** (verified — not a data leak): a causal
> decoder is *not* permutation-invariant. The causal mask is itself an ordering signal, so
> even with no explicit PE the model learns *"have I already seen an X by the time I reach Y?"*.
> `verify_order.py` confirms 1000/1000 matched-pair flips where the bag of tokens is identical.

---

## Directory Structure
```
generalization-order/
├── README.md
├── model.py            (copied from generalization-detect, unchanged — task-agnostic)
├── pos_encoding.py     (copied from generalization-detect, unchanged)
├── train.py            (per-example batching, answer-token-only loss; saves seed in ckpt)
├── evaluate.py         (per-class T/F accuracy; appends results.csv + predictions.csv)
├── plot.py             (results.csv/predictions.csv -> bar chart + per-position heatmap)
├── verify_order.py     (matched-pair order probe — isolates order as the only variable)
├── results.csv         (generated; one aggregate row per run — the figure data record)
├── predictions.csv     (generated, gitignored; per-example sweep for the heatmap)
├── config/
│   └── basic.py
├── data/
│   └── order/
│       ├── prepare.py
│       ├── meta.pkl    (generated; stores vocab + split/split_detail)
│       ├── train.bin   (generated)
│       ├── val.bin     (generated)
│       └── test.txt    (generated)
├── log/                (per-experiment logs)
└── out/                (checkpoint + figure PNGs)
```

---

## Setup
From the repo root (`comp560-JohnLee`):
```bash
cd generalization-order      # the venv at ../venv has torch + numpy
```

## Prepare data
```bash
../venv/bin/python data/order/prepare.py
```
Prints class balance per pool and runs a **label-correctness assertion**
(`index(X) < index(Y)` iff label `T`) over every example before writing the bins.

## Train (the `pos_type` is the one knob that varies)
```bash
../venv/bin/python train.py config/basic.py --pos_type=none
../venv/bin/python train.py config/basic.py --pos_type=learned
```

## Evaluate (per-class T/F on the held-out test set)
```bash
../venv/bin/python evaluate.py config/basic.py --pos_type=none
```

## Verify (matched-pair order probe)
For random `(i<j)`, builds two byte-identical inputs differing only in whether `X` or
`Y` sits first, and checks the prediction flips `T`↔`F`. 100% proves the model reads
*order*, not content (the bag of tokens is identical between the two versions):
```bash
../venv/bin/python verify_order.py --n_trials=1000
```

## Results logging + figures
Every `evaluate.py` run **appends** one aggregate row to `results.csv`
(`timestamp, pos_type, split, split_detail, seed, val_acc, heldout_acc, heldout_T_acc,
heldout_F_acc`) and one row per diagnostic-sweep example to `predictions.csv`
(`pos_type, split, seed, x_pos, y_pos, gold, pred, correct`). The sweep spans **all**
position pairs (train *and* held-out regions) so the per-position figure can show the cliff.

`plot.py` turns those CSVs into PNGs in `out/` (matplotlib only). It plots only the methods
present, in a fixed order (`none, learned, sinusoidal, rope, t5`), so the figures extend
themselves automatically as the other PEs are added:
```bash
../venv/bin/python plot.py --split=half        # default; --split=single_pos also works
```
- `out/heldout_accuracy_<split>.png` — held-out acc by method, seed error bars, chance line,
  faded in-distribution val bars (the conclusion).
- `out/per_position_<split>.png` — accuracy over `(x_pos, y_pos)` pairs, one heatmap per
  method, train vs held-out region outlined (the mechanism / *why*).
- `out/heldout_perclass_<split>.png` — per-class T vs F (shows label collapse).

Committed figures for the log live in [`log/figures/`](log/figures/) (generate them there
with `plot.py --out_dir=log/figures`) and are embedded by **relative path**, so they render
directly on GitHub — no manual asset-URL step. Scratch copies in `out/` and `predictions.csv`
are gitignored; `results.csv` and `log/figures/*.png` are kept.

---

## Experiment Logs

| Date | Experiment | Status |
|------|------------|--------|
| 2026-06-20 | [Full-distribution baselines (none vs learned)](log/2026-06-20-relative-order-baselines.md) | ✅ both 100% (none surprising) |
| 2026-06-20 | [Held-out positions — the phenomenon (none vs learned)](log/2026-06-20-relative-order-heldout-splits.md) | ✅ `half`: NoPE ≈100% vs learned ≈58% |
| 2026-06-20 | [PE sweep — Phase 3 (all 5 encodings)](log/2026-06-20-relative-order-pe-sweep.md) | ✅ **relative (none/rope/t5) ≈100% vs absolute (learned 58% / sinusoidal 33%)** |
| _next_ | Strengthen sweep (data reseed) · more splits · Phase 4 minimal model | — |

To switch splits, edit `SPLIT` (and `HELDOUT_POS`) at the top of
[data/order/prepare.py](data/order/prepare.py), regenerate, then train+eval as above.

**New experiments:** copy [log/_TEMPLATE.md](log/_TEMPLATE.md) → `log/YYYY-MM-DD-<slug>.md`. It
bakes in the reproducibility fields (commit · exact command · config · seeds · env · caveats),
so every result is traceable to the code and data that made it. Keep day-to-day notes (decisions,
dead ends) in the separate running/activity log.

---

## Next Steps
- **Phase 3 — PE sweep: ✅ done.** All five PEs implemented and run on the `half` split;
  relative (none/rope/t5) ≈100%, absolute (learned/sinusoidal) ≤58%. Hypothesis confirmed.
- **Strengthen the seed sweep** — regenerate data per seed (current sweep varies only
  model init / batch order on one fixed half split), report mean/variance.
- **More splits** — even/odd and distance-based hold-outs, to test whether the
  relative/absolute line holds across generalization axes.
- **Phase 4 — minimal model** — shrink layers/heads, re-check the separation.
- **(Phase 7) interpretability** — confirm the relative-vs-absolute mechanism story.
