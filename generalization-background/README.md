# generalization-background — background-token diversity (`b`) sweep

A T5 stress test. Across the earlier tasks (`generalization-order` / `-evenodd` / `-distd`)
T5 kept underperforming, and it was never clear why. Every one of those tasks used a **busy
background**: filler slots were random digits 0–9, i.e. `b=10`. Hypothesis: T5 was not
failing at the *task*, it was failing to find `X` and `Y` inside all that digit noise.

This folder isolates that. The task is kept deliberately **easy** so it is never the
bottleneck, and one knob is swept: **`b`, the number of distinct background token types.**

## The task

Fixed-length string containing `X` exactly once and `Y` exactly once, with **X always to the
left of Y**. Every other slot is background. Answer token last:

```
XY0000:T     X@0 Y@1  distance 1  -> T
0XY000:T     X@1 Y@2  distance 1  -> T
X0Y000:F     X@0 Y@2  distance 2  -> F
X00Y00:F     X@0 Y@3  distance 3  -> F
```

**Label: `T` iff the distance from `X` to `Y` is exactly 1, else `F`.** Since X is always to
the left of Y, the whole task is "is the distance 1, or larger?"

## The knob

| `b` | background token set |
|---|---|
| 1 | `{0}` |
| 2 | `{0,1}` |
| … | … |
| 10 | `{0,…,9}` — the busy background every earlier task used |

Background slots are filled uniformly at random from that size-`b` set. Vocab size grows
with `b` and is read from the built data, never hardcoded.

## The split

Held-out **position**, not length — every input is exactly `length` characters.

- **`half`** — train/val put both `X` and `Y` in the first half; the held-out test puts both
  in the second half, never seen in training.
- **`none`** — full distribution, a plain learnability baseline.

50/50 T/F in every pool.

## Structure

Engine (`model.py`, `pos_encoding.py`) is reused **unchanged** from `generalization-distd`.
Only the data and one figure are task-specific.

```
data/background/prepare.py   # data gen; b, length, split are all arguments
config/basic.py            # small model (n_embd=32, n_head=2, n_layer=3), pos_type=t5
train.py                   # per-example batching, answer-token-only loss; logs curves.csv
evaluate.py                # logs results.csv (with a b column) + predictions.csv
plot.py                    # accuracy-vs-b, per-position heatmaps, learnability curves
log/                       # one polished report per stage, figures under log/figures/
```

`pos_type=t5` is fixed by design — this is a T5 stress test, not a PE comparison.

## Run

```bash
# generate data for a given b (b, length, split are all overridable)
../venv/bin/python data/background/prepare.py --b=1
../venv/bin/python data/background/prepare.py --b=4 --length=12 --split=half

# train + evaluate one run
../venv/bin/python train.py    config/basic.py --seed=1337
../venv/bin/python evaluate.py config/basic.py --seed=1337

# sweep b
for b in 1 2 3 4 5 6 7 8 9 10; do
  ../venv/bin/python data/background/prepare.py --b=$b
  for s in 1337 1338 1339 1340; do
    ../venv/bin/python train.py    config/basic.py --seed=$s
    ../venv/bin/python evaluate.py config/basic.py --seed=$s
  done
done

# figures
../venv/bin/python plot.py --split=half --vs_b --out_dir=log/figures   # headline
../venv/bin/python plot.py --split=half --out_dir=log/figures          # per-position heatmaps
../venv/bin/python plot.py --curve --b=1 --length=6 --split=half       # learnability curve
```

**Lengths other than 6 need a bigger `block_size`** (`length + 2`): pass
`--block_size=14` for length 12. `train.py` asserts this and tells you what to set.

## Success = learned AND generalized

Every `results.csv` row carries **both** numbers, and they must be read together:

| val (in-dist) | held-out | reading |
|---|---|---|
| high | high | T5 fully succeeds at this `b` |
| high | low | learned but did not generalize — background hurts **generalization** |
| low | — | never learned it — background hurts **learning** |

## Status

- **Stage 1 (clean background, b=1) — ✅ ran, ❌ did not pass.**
  See [`log/2026-07-18-background-stage1-clean-baseline.md`](log/2026-07-18-background-stage1-clean-baseline.md).
  Run at lengths 6, 8, 10 and 12, **10 seeds each (40 runs)**.
- The model **always learns** the task: validation accuracy is 100% in all 40 runs. The problem
  is only carrying it to unseen positions.
- At **length 6**, 9 of 10 runs fail completely — they answer `T` to everything, scoring exactly
  50%. Only seed 1338 reaches 100%, and nothing lands in between. The training half has just 3
  positions, so it contains exactly **one** way to place an `F` pair (X at 0, Y at 2); one
  placement is not enough to pin the rule down.
- **Longer inputs fix the complete failure but not the perfect-score rate:** complete failures go
  9 → 4 → 6 → 0 out of 10 across lengths 6/8/10/12 (length 6 vs 12, Fisher p = 0.0001), while
  perfect runs stay flat at 1, 4, 1, 4 (p = 0.30, no trend). Length 12 is the best-behaved —
  nothing collapses, worst run 75%, narrowest spread (sd 8.9) — but still only 4/10 perfect.
- **The background sweep (b = 1…10) has not been started.** Step 1 was meant to establish a
  reliable clean-background baseline to compare against, and it did not.

## Caveats

- One fixed data split per (b, seed); seeds vary init + batch order only. Regenerate data per
  seed later to strengthen the variance estimate.
- T5 only by design. If a result looks T5-specific, a spot-check on `rope` can say whether it
  is a T5 property or general — but the focus stays T5.
