# generalization-adjacent — adjacency task + background-diversity (`b`) sweep

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
XY0000:T     X@0 Y@1  adjacent  -> T
0XY000:T     X@1 Y@2  adjacent  -> T
X0Y000:F     X@0 Y@2  gap of 1  -> F
```

**Label: `T` iff `Y` sits immediately after `X` (gap 1), else `F`.** Since X is always before
Y, this is purely "adjacent, or is there a gap".

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
data/adjacent/prepare.py   # data gen; b, length, split are all arguments
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
../venv/bin/python data/adjacent/prepare.py --b=1
../venv/bin/python data/adjacent/prepare.py --b=4 --length=12 --split=half

# train + evaluate one run
../venv/bin/python train.py    config/basic.py --seed=1337
../venv/bin/python evaluate.py config/basic.py --seed=1337

# sweep b
for b in 1 2 3 4 5 6 7 8 9 10; do
  ../venv/bin/python data/adjacent/prepare.py --b=$b
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

- **Stage 1 (length 6, b=1) — ✅ ran, ❌ did not pass the gate.**
  See [`log/2026-07-18-adjacent-stage1-b1-baseline.md`](log/2026-07-18-adjacent-stage1-b1-baseline.md).
  Val hits 100% within ~100 iters, but held-out is **bimodal across seeds: 100% / 50% / 50% /
  50%**. The cause is not T5 and not the background — it is that **length 6 with a half split
  is a degenerate measurement**. Each half is 3 positions, so the training region contains
  exactly **one** F configuration (`X@0,Y@2`), and the held-out region's only F configuration
  is the same gap-2 case. The diagnostic sweep shows all four seeds read the **gap**
  correctly (gap ≥ 3 → F everywhere, including untrained positions — no position shortcut);
  they differ only on whether gap-2 → F transfers away from the single memorized instance.
- **Confirmation:** at **length 12** the same config gives held-out 85 / 95 / 100 / 85%
  (no bimodal collapse), and the `none` split at length 6 gives 100% on 4/4 seeds.
- **Therefore: run the `b` sweep at length 10–12, not length 6.** The staging in the original
  spec has Stage 3 (length scaling) after Stage 2 (the b sweep); for this task that order
  needs to be inverted, because the length-6 "debugging anchor" cannot measure a background
  effect — its held-out score is dominated by a single fragile configuration.

## Caveats

- One fixed data split per (b, seed); seeds vary init + batch order only. Regenerate data per
  seed later to strengthen the variance estimate.
- T5 only by design. If a result looks T5-specific, a spot-check on `rope` can say whether it
  is a T5 property or general — but the focus stays T5.
