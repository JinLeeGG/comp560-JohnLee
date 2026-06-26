# Generalization Experiment: dist≥D Threshold Task (distance task #2)
**Date:** June 26, 2026

This experiment trains a micro-transformer (<1M params) on a **distance-threshold** task —
"are the two `X`'s at least `D` apart?" — and measures whether it generalizes to symbol
*positions* it never saw during training. It is the **coarse** counterpart to the
[even/odd-separation experiment](../generalization-evenodd/README.md): where even/odd asked
for the *exact parity* of the X–X distance (and hit a capacity wall at micro scale — every
PE/seed stuck at 50% in-distribution val), this asks only a **monotone, far-vs-near**
question, which needs a less precise distance representation and may fit under the same
micro budget. It reuses the from-scratch swappable-PE engine (`model.py` + `pos_encoding.py`)
unchanged; only the label rule and one constant (`D`) differ.

> **Critical framing — this is NOT length generalization.** Every input is exactly
> `LENGTH = 20` characters. The distribution shift is over *where* the two `X`'s appear
> within that fixed length, never over sequence length.

---

## Task

A fixed-length string of digits contains the symbol `X` **exactly twice** (two identical
targets; the other 18 characters are random digits). The model reads the string, sees `:`,
and must output a **single** token:

```
DISTANCE  = |pos2 - pos1|              (absolute difference of the two X positions)
THRESHOLD = D = 5                       (the one constant that defines this task)
LABEL MAPPING (fixed — never flip):     T  iff  DISTANCE ≥ 5 (far),  else  F (near)

482X50178X4019285746:T     ← X at 3, X at 9 → distance 6 ≥ 5 → T (far)
482X5X1786401928574 6:F    ← X at 3, X at 5 → distance 2 < 5 → F (near)
```

**Why two identical `X`'s (not `X` and `Y`):** distance is order-invariant, so a second
distinct symbol would add an irrelevant order cue. Two `X`'s isolate distance cleanly. Vocab
(size 15): digits `0`–`9`, plus `X`, `T`, `F`, `:`, `\n` (no `Y`).

**Why `D = 5`:** it splits the held-out region's possible distances (1–9, see below) into
`F = {1,2,3,4}` and `T = {5,6,7,8,9}` — roughly half and half — so the classes can be
balanced exactly. Classes are forced 50/50 (T/F) in every pool, so chance accuracy is 50%.

**Why this might clear the even/odd wall (honest expectation: a coin flip, not a safe bet).**
dist≥D depends on the *magnitude* of the gap, a monotone signal — there is no "off-by-one
flips the label" sensitivity that parity has. So it should need a coarser distance
representation. But it still has to read distance at all: if distance-*reading* (not parity
specifically) is the micro bottleneck, dist≥D hits the same wall. Both outcomes are
informative (see "Relation to even/odd" below).

---

## The position split (the experiment knob)

`SPLIT` in [data/distd/prepare.py](data/distd/prepare.py) controls **where the two `X`'s may
appear**:

| `SPLIT` | train/val rule | test rule |
|---------|----------------|-----------|
| `none`  | both `X`'s anywhere (full dist.) | both `X`'s anywhere (in-distribution) |
| `half`  | both `X`'s in first half (0–9) | both `X`'s in second half (10–19) |

All pools stay 50/50 T/F (chance = 50%); `prepare.py` asserts the label mapping **and** the
split rule on every example before writing the bins.

**Distance balance in the held-out pool (required for the separation graph).** Within the
held-out second half (positions 10–19) distances range 1–9, and larger distances have far
fewer position pairs (distance 9 = only the single pair `{10,19}`). The held-out test pool is
built distance-balanced — equal examples per distance — with **exact 50/50 class balance
prioritized** when the two can't both be exact. For `D = 5`, `N_TEST = 2000` it is clean:
distances 1/2/3/4 → 250 each (F = 1000), distances 5/6/7/8/9 → 200 each (T = 1000).
`prepare.py` prints the per-distance counts to confirm coverage.

---

## The question this asks

even/odd separation was meant to be the task where **NoPE fails** (the causal mask gives
order but not exact distance) — but it never got there: a micro model couldn't learn parity
at all. dist≥D is the retry at one notch easier. The comparison even/odd couldn't deliver:

- Does **NoPE** (`none`) fail on the held-out half — can read order, not distance? Or is
  coarse distance approximable from the causal mask (a finding either way)?
- Do **rope / t5** (distance-aware) hold up, at least near the threshold?
- Do **learned / sinusoidal** (absolute, which can subtract positions) beat NoPE? If absolute
  beats NoPE on a *distance* task, that's the flip side of the relative-order result:

  > **order → relative methods win**  ·  **distance → distance/absolute methods win**

- Read val alongside held-out: high val + low held-out = learned-but-didn't-generalize; both
  low = didn't learn (the even/odd outcome).

**Confound to check (more exposed here than in even/odd).** On the `half` split, "dist ≥ 5"
correlates with "the two `X`'s sit at opposite ENDS of 10–19", so a model could win by a
*position pattern* ("one `X` left, one right") instead of measuring distance. The
**per-position heatmap distinguishes them**: genuine distance → **bands parallel to the
diagonal**; a position shortcut → **blocks**. Read the heatmap before claiming the model
reads distance.

### Headline result — *coarse distance clears the even/odd wall; generalization is partial*

The learnability gate **passed**: at the micro baseline (0.80M) both `learned` and `rope`
reach **100% in-dist val by ~iter 250** (vs even/odd's flat 50% wall). The full 5-PE × 4-seed
sweep then shows **all 20 runs at 100% in-dist val** — so dist≥D *is* micro-learnable,
**localizing the even/odd wall to parity's exactness, not distance-reading in general.**

Generalization to held-out positions, however, is **partial and strongly seed-dependent**:

| PE | family | in-dist val | held-out mean | per seed | generalized |
|----|--------|------------:|--------------:|----------|:-----------:|
| `none` | NoPE | 100% | 58.8% | 50/50/**85**/50 | 1/4 |
| `learned` | absolute | 100% | **50.0%** | 50/50/50/50 | **0/4** |
| `sinusoidal` | absolute | 100% | 61.2% | 50/**66**/50/**79** | 2/4 |
| `rope` | relative | 100% | 55.0% | **72**/50/50/48 | 1/4 |
| `t5` | relative | 100% | 67.5% | **85**/50/50/**85** | 2/4 |

The one robust separation: **absolute `learned` (APE) never generalizes** (collapses to
"near" on all 4 seeds); every other PE generalizes on ≥1 seed. But the held-out means are
**not a ranking** at n=4 (the middle four are within noise), and what transfers across
positions is mainly the **near** class: pooled over seeds the far (T) side stays **at/below
chance for every distance d5–9** — only a *minority* of seeds (e.g. t5/1337, t5/1340,
none/1339) recover the far side. Even on those seeds accuracy at a *fixed* distance depends on
position (correct near the trained boundary, wrong deeper in), so **distance-reading vs a
position shortcut is unresolved and leans toward the shortcut** — the per-position heatmaps
look more block-like (position) than band-like (distance). Method B (distance-held-out) is the
clean test. Full writeup: [log/2026-06-26-distd-pe-sweep.md](log/2026-06-26-distd-pe-sweep.md).

<img src="log/figures/accuracy_vs_separation_half.png" alt="accuracy vs separation: ~100% at distances 1-4, collapse at the D=5 threshold, far side stays at/below chance for all PEs in the pool; learned flat at 0" width="620">

*Accuracy-vs-separation (held-out region, pooled over seeds): near distances (1–4) solved,
then at/beyond the D=5 threshold every PE sits **at or below chance** on the far side —
generalization in the pool is mostly the near class, not recovered far. `learned` is flat at
0% (always "near").*

---

## Directory Structure
```
generalization-distd/
├── README.md
├── model.py            (copied from generalization-evenodd, unchanged — task-agnostic)
├── pos_encoding.py     (copied from generalization-evenodd, unchanged)
├── train.py            (per-example batching, answer-token-only loss; saves seed in ckpt)
├── evaluate.py         (per-class T/F accuracy; appends results.csv + predictions.csv
│                        with x1_pos/x2_pos/distance; reads threshold D from meta)
├── plot.py             (bar chart + per-position heatmap + accuracy-vs-separation curve)
├── results.csv         (generated; one aggregate row per run — the figure data record)
├── predictions.csv     (generated, gitignored; per-example sweep for the figures)
├── config/
│   └── basic.py
├── data/
│   └── distd/
│       ├── prepare.py
│       ├── meta.pkl    (generated; stores vocab + split/split_detail + threshold D)
│       ├── train.bin   (generated)
│       ├── val.bin     (generated)
│       └── test.txt    (generated)
├── log/                (per-experiment logs + committed figures)
└── out/                (checkpoint + figure PNGs)
```

---

## Setup
From the repo root (`comp560-JohnLee`):
```bash
cd generalization-distd        # the venv at ../venv has torch + numpy + matplotlib
```

## Prepare data
```bash
../venv/bin/python data/distd/prepare.py
```
Prints class balance per pool, per-distance coverage in the held-out pool, and runs a
**label-correctness assertion** (`|pos2−pos1| ≥ 5` iff `T`) plus a **split assertion** over
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
- `out/heldout_perclass_<split>.png` — per-class T (far) vs F (near) (shows label collapse).
- `out/per_position_<split>.png` — accuracy over `(x1_pos, x2_pos)` pairs (upper triangle,
  X1 < X2), train vs held-out region outlined; **bands = distance, blocks = shortcut**.
- `out/accuracy_vs_separation_<split>.png` — **the figure from MacCormick's note:** the
  held-out block of the heatmap collapsed along the diagonal — mean accuracy vs distance, one
  line per method, with the threshold `D` marked.

Committed figures for the log live in [`log/figures/`](log/figures/) (generate them there
with `plot.py --out_dir=log/figures`). Scratch copies in `out/` and `predictions.csv` are
gitignored; `results.csv` and `log/figures/*.png` are kept.

---

## Run plan (the full sweep)
```bash
../venv/bin/python data/distd/prepare.py
for pe in none learned sinusoidal rope t5; do
  for seed in 1337 1338 1339 1340; do
    ../venv/bin/python train.py    config/basic.py --pos_type=$pe --seed=$seed
    ../venv/bin/python evaluate.py config/basic.py --pos_type=$pe --seed=$seed
  done
done
../venv/bin/python plot.py --split=half
../venv/bin/python plot.py --split=half --separation
```

**Caveat (same as the even/odd sweep):** the seed sweep varies only model init / batch order
on **one fixed data split** (`prepare.py` is run once). Regenerating the data per seed
(`SEED` in `prepare.py`) is the stronger protocol and the natural next step.

---

## Relation to even/odd

Same distance family, one notch easier. dist≥D being micro-learnable (gate passed) while
even/odd was not **localizes the even/odd wall to parity's exactness, not to distance-reading
in general** — a clean contrast. (Had dist≥D *also* failed, that would have pointed at
distance-reading itself as the micro bottleneck.)

---

## Not in scope
- **Distance-based held-out split (method B)** — train on near distances, test on far. A
  different generalization axis, later.
- **Sweeping D** — one threshold (5) for now; other D shifts class balance and difficulty.

---

## Experiment Logs

| Date | Experiment | Status |
|------|------------|--------|
| 2026-06-26 | [dist≥5 PE sweep — half split (all 5 encodings)](log/2026-06-26-distd-pe-sweep.md) | ✅ micro-learnable (all val 100%) → clears even/odd wall; held-out partial & seed-dependent; absolute `learned` never generalizes |
