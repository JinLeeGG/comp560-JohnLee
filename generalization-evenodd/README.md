# Generalization Experiment: Even/Odd-Separation Task (distance task #1)
**Date:** June 25, 2026

This experiment trains a micro-transformer (<1M params) on an **even/odd-separation**
task and measures whether it generalizes to symbol *positions* it never saw during
training. It is the **distance** counterpart to the [relative-order
experiment](../generalization-order/README.md): where relative order tested whether the
model reads *which symbol comes first*, this tests whether it reads *how far apart* the
two symbols are. It reuses the from-scratch swappable-PE engine (`model.py` +
`pos_encoding.py`); only the data, config, and one new plot are task-specific.

> **Critical framing — this is NOT length generalization.** Every input is exactly
> `LENGTH = 20` characters. The distribution shift is over *where* the two `X`'s appear
> within that fixed length, never over sequence length.

---

## Task

A fixed-length string of digits contains the symbol `X` **exactly twice** (two identical
targets; the other 18 characters are random digits). The model reads the string, sees
`:`, and must output a **single** token:

```
DISTANCE  = |pos2 - pos1|            (absolute difference of the two X positions)
LABEL MAPPING (fixed — never flip):  T  iff  DISTANCE is EVEN,  else  F

482X5017X64019285746:F     ← X at 3, X at 8 → distance 5 (odd)  → F
482X50178X4019285746:T     ← X at 3, X at 9 → distance 6 (even) → T
```

**Why two identical `X`'s (not `X` and `Y`):** even/odd separation depends only on the
*distance* between the targets, not their order, so a second distinct symbol would add an
irrelevant order cue. Two `X`'s isolate distance cleanly — the deliberate contrast with
relative order (which used `X` and `Y` because order mattered there). Vocab (size 15):
digits `0`–`9`, plus `X`, `T`, `F`, `:`, `\n` (no `Y`).

Classes are balanced 50/50 (T/F), so chance accuracy is 50%.

---

## The position split (the experiment knob)

`SPLIT` in [data/evenodd/prepare.py](data/evenodd/prepare.py) controls **where the two
`X`'s may appear**:

| `SPLIT` | train/val rule | test rule |
|---------|----------------|-----------|
| `none`  | both `X`'s anywhere (full dist.) | both `X`'s anywhere (in-distribution) |
| `half`  | both `X`'s in first half (0–9) | both `X`'s in second half (10–19) |

All pools stay 50/50 T/F (chance = 50%); `prepare.py` asserts the label mapping **and** the
split rule on every example before writing the bins.

**Distance balance in the held-out pool (required for the separation graph).** Within the
held-out second half (positions 10–19) distances range 1–9, and larger distances have far
fewer position pairs (distance 9 = only the single pair `{10,19}`). Sampling positions
uniformly would starve the large distances, so the held-out test pool is built by sampling
a roughly-equal number of examples **per distance** (every even distance gets the same
count, every odd distance the same count, chosen so T/F stays exactly 50/50). With
`N_TEST=2000`: distances 2/4/6/8 → 250 each (T = 1000), distances 1/3/5/7/9 → 200 each
(F = 1000). `prepare.py` prints the per-distance counts to confirm coverage.

---

## The question this asks

On relative order, NoPE (`none`, no positional encoding) generalized to held-out positions
**≈100%**, because a causal decoder reads *order* straight off the causal mask. Even/odd
separation needs something the causal mask cannot give: the **exact distance** between two
identical symbols. So this is meant to be the first task where **NoPE *fails*** —
completing the other side of the task→method mapping:

> **order → relative methods win**  ·  **distance → distance/absolute methods win**

MacCormick's prediction for the accuracy-vs-separation figure:
- distance-aware methods (**RoPE**, **T5** relative bias) do well at short separations and
  may fall off as distance grows;
- **NoPE** fails — it can't count distance;
- **learned / sinusoidal** absolute PEs may *beat* NoPE here, since absolute positions can
  be subtracted to recover the distance.

### Headline result — *a capacity wall sits below the PE comparison*

The planned 5-PE × 4-seed sweep returned a **uniform null**: every method, every seed sat
at **exactly 50%** on the held-out test **and on in-distribution `val`**, collapsing to a
single label. The micro-transformer **never learned even/odd parity at all** — not even on
the training region — so the NoPE-vs-distance comparison can't be read at this scale.

It is *not* a degenerate task. Diagnostics pin it as a **capacity wall**:

| scale / setting | params | in-dist. val | held-out test |
|-----------------|-------:|--------------|---------------|
| micro, any PE, 2 000 iters (the sweep) | 0.80M | **50%** | 50% |
| micro `learned`/`rope`, **15 000** iters | 0.80M | **50%** | — |
| micro `learned`, lr **3e-3** | 0.80M | **50%** | — |
| **big `learned`** (`n_layer=6,n_head=8,n_embd=256`) | **4.76M** | **100%** (≈iter 2000) | **44.85%** (below chance) |

So even/odd distance-parity is genuinely learnable — just **beyond the <1M micro budget**
(threshold between 0.80M and 4.76M). And at the learnable 4.76M scale, **absolute PE learns
the task but fails to generalize** to held-out positions (val 100% → held-out 44.85%,
collapsed) — the same "learns-but-doesn't-generalize" signature as relative order, now on a
*distance* task. Full writeup:
[log/2026-06-25-evenodd-pe-sweep.md](log/2026-06-25-evenodd-pe-sweep.md).

<img src="log/figures/accuracy_vs_separation_half.png" alt="accuracy vs separation: all five PEs flat at chance (50%) across every held-out distance" width="640">

*The accuracy-vs-separation figure (MacCormick's note): at micro scale all five PEs lie on
the 50% chance line at every separation — the collapsed signature of nothing learned. The
distance-aware shape it was meant to reveal only becomes testable once the task is learned,
i.e. above the micro budget.*

> **Decision deferred (raised with advisor):** complete the comparison at the learnable
> ~4.76M scale · find the minimal capacity that learns it · add scratchpad (Phase 5) to
> crack it within <1M · or pivot to the easier `dist≥D` task. See the log's *Next* section.

---

## Directory Structure
```
generalization-evenodd/
├── README.md
├── model.py            (copied from generalization-order, unchanged — task-agnostic)
├── pos_encoding.py     (copied from generalization-order, unchanged)
├── train.py            (per-example batching, answer-token-only loss; saves seed in ckpt)
├── evaluate.py         (per-class T/F accuracy; appends results.csv + predictions.csv
│                        with x1_pos/x2_pos/distance)
├── plot.py             (bar chart + per-position heatmap + accuracy-vs-separation curve)
├── results.csv         (generated; one aggregate row per run — the figure data record)
├── predictions.csv     (generated, gitignored; per-example sweep for the figures)
├── config/
│   └── basic.py
├── data/
│   └── evenodd/
│       ├── prepare.py
│       ├── meta.pkl    (generated; stores vocab + split/split_detail)
│       ├── train.bin   (generated)
│       ├── val.bin     (generated)
│       └── test.txt    (generated)
├── log/                (per-experiment logs + committed figures)
└── out/                (checkpoint + figure PNGs)
```

> No matched-pair probe (`verify_*.py`) ships here: the order probe flipped `X`↔`Y` to flip
> the label while holding the token bag fixed, which isolated order cleanly. With two
> *identical* `X`'s there is no analogous label-flip that keeps the exact token multiset, so
> that probe does not translate. The label-correctness + split assertions in `prepare.py`
> and the per-position heatmap take its place.

---

## Setup
From the repo root (`comp560-JohnLee`):
```bash
cd generalization-evenodd      # the venv at ../venv has torch + numpy + matplotlib
```

## Prepare data
```bash
../venv/bin/python data/evenodd/prepare.py
```
Prints class balance per pool, per-distance coverage in the held-out pool, and runs a
**label-correctness assertion** (`|pos2−pos1|` even iff `T`) plus a **split assertion** over
every example before writing the bins.

## Train (the `pos_type` is the one knob that varies)
```bash
../venv/bin/python train.py config/basic.py --pos_type=none --seed=1337
../venv/bin/python train.py config/basic.py --pos_type=rope --seed=1337
```

## Evaluate (per-class T/F on the held-out test set)
```bash
../venv/bin/python evaluate.py config/basic.py --pos_type=none --seed=1337
```

## Results logging + figures
Every `evaluate.py` run **appends** one aggregate row to `results.csv` and one row per
diagnostic-sweep example to `predictions.csv` (`pos_type, split, seed, x1_pos, x2_pos,
distance, gold, pred, correct`). The sweep spans **all** unordered position pairs (train
*and* held-out regions) so the per-position figure shows the cliff and the separation curve
has coverage at every distance.

`plot.py` turns those CSVs into PNGs in `out/`:
```bash
../venv/bin/python plot.py --split=half               # bar + per-class + per-position heatmap
../venv/bin/python plot.py --split=half --separation  # accuracy-vs-separation curve
```
- `out/heldout_accuracy_<split>.png` — held-out acc by method, seed error bars, chance line,
  faded in-distribution val bars (the conclusion).
- `out/heldout_perclass_<split>.png` — per-class T (even) vs F (odd) (shows label collapse).
- `out/per_position_<split>.png` — accuracy over `(x1_pos, x2_pos)` pairs (upper triangle,
  X1 < X2), train vs held-out region outlined (the mechanism).
- `out/accuracy_vs_separation_<split>.png` — **the figure from MacCormick's note:** the
  held-out block of the heatmap collapsed along the diagonal — mean accuracy vs distance,
  one line per method.

Committed figures for the log live in [`log/figures/`](log/figures/) (generate them there
with `plot.py --out_dir=log/figures`). Scratch copies in `out/` and `predictions.csv` are
gitignored; `results.csv` and `log/figures/*.png` are kept.

---

## Run plan (the full sweep)
```bash
../venv/bin/python data/evenodd/prepare.py
for pe in none learned sinusoidal rope t5; do
  for seed in 1337 1338 1339 1340; do
    ../venv/bin/python train.py    config/basic.py --pos_type=$pe --seed=$seed
    ../venv/bin/python evaluate.py config/basic.py --pos_type=$pe --seed=$seed
  done
done
../venv/bin/python plot.py --split=half
../venv/bin/python plot.py --split=half --separation
```

**Caveat (same as the order sweep):** the seed sweep varies only model init / batch order
on **one fixed data split** (`prepare.py` is run once). Regenerating the data per seed
(`SEED` in `prepare.py`) is the stronger protocol and is the natural next step.

---

## Experiment Logs

| Date | Experiment | Status |
|------|------------|--------|
| 2026-06-25 | [Even/odd PE sweep — half split (all 5 encodings)](log/2026-06-25-evenodd-pe-sweep.md) | ✅ uniform null at micro scale → **capacity wall** (4.76M learns it; absolute PE doesn't generalize) |

---

## Next Steps
- **dist(X,Y) ≥ D task** — the other binary distance task (stays single-token T/F), after
  even/odd. The non-binary "output the exact distance" variant is intentionally excluded:
  its output isn't a single T/F token, so it breaks the answer-token-loss / per-class
  structure the pipeline is built on.
- **Distance-based held-out split (method B)** — train on near distances, test on far
  distances (a different generalization axis than position held-out).
- **Reseed the data per seed** — strengthen the variance estimate (see caveat).
