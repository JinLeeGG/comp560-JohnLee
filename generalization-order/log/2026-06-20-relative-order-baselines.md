# Relative-Order Task — full-distribution baselines (`none` vs `learned`)

*2026-06-20 · commit `8213fcd` · John Lee*

> **Bottom line.** On the full position distribution, **both** settings solve the task at
> **100%** — including `none`, which has **no positional encoding**. A causal decoder gets the
> order of its tokens "for free" from its causal mask, so this baseline cannot tell the PE methods
> apart. The real test therefore moves to **held-out positions** (next log).

### The task

Length-20 string with **one `X`** and **one `Y`** (rest are random digits). Output the last token:

```
T  if  X is before Y        4829X017364Y19285746 : T
F  if  Y is before X        4829Y017364X19285746 : F      rule (fixed): T ⇔ index(X) < index(Y)
```

The two settings compared (only this changes between runs):
**`none` = NoPE** (no positional encoding) · **`learned` = APE** (learned absolute position embedding).

---

## Setup

- **data / split:** length 20, one `X` + one `Y` + 18 random digits; **full distribution** (`X`,`Y`
  at any positions); pools 50k / 5k / 2k (train/val/test), 50/50 T/F, chance = 50%; vocab 16 read
  from the data. **Integrity check:** every pool exactly 50/50, and the label assertion
  (`index(X) < index(Y)` ⇔ `T`) passed on all **57,000** examples before training.
- **model / config:** ~0.8M params (n_layer 4, n_head 4, n_embd 128), CPU, 2000 iters, AdamW
  lr 1e-3, batch 64, answer-token-only loss (the loss looks only at the final T/F token, not the digits).
- **seeds:** single seed 1337. The result is already at the 100% ceiling, so extra seeds would add
  nothing; the held-out experiment (where results are *not* maxed out) uses a multi-seed sweep.
- **env:** python 3.9.6 · torch 2.8.0 · numpy 2.0.2 · data `SEED=1337`.
- **reproduce** (from `generalization-order/`):
  ```bash
  # baselines use the FULL distribution: set SPLIT='none' in data/order/prepare.py, then
  ../venv/bin/python data/order/prepare.py
  ../venv/bin/python train.py    config/basic.py --pos_type=none    --seed=1337   # also --pos_type=learned
  ../venv/bin/python evaluate.py config/basic.py --pos_type=none    --seed=1337   # appends results.csv
  ../venv/bin/python verify_order.py --n_trials=1000                              # the order-only probe (control)
  ```

---

## Results

Both settings solve the task at 100% (per-class T and F):

| pos_type | T (X before Y) | F (Y before X) | overall |
|----------|----------------|----------------|---------|
| `none` (NoPE) | 100% (1000/1000) | 100% (1000/1000) | **100%** |
| `learned` (APE) | 100% (1000/1000) | 100% (1000/1000) | **100%** |

`none` was *expected* near chance (50%): with no position information it shouldn't be able to tell
which symbol came first. It scored **100%** — the surprise that drives the rest of the project. Val reached 100% early in
training (val loss ≈ 0) for both.

**Control — is `none`'s 100% a data leak?** Isolate *order* as the only variable: for 1000 trials,
take one digit body + one position pair (i < j) and build two **byte-identical** inputs differing
only in which symbol is first — `X@i, Y@j` (→ T) vs `Y@i, X@j` (→ F). The two inputs contain the
exact same characters — only the order differs — so **only order** can flip the prediction. Result: **1000/1000 flips correct** → the model genuinely
reads order. **Not a data leak.**

Example data (full distribution — `X`,`Y` anywhere):
```
X@ 3 Y@11 -> T   869X7678304Y87307021:T
X@13 Y@ 9 -> F   987570430Y204X382685:F
```

---

## Why

A causal decoder is **not order-blind**. The causal mask lets each position attend only to earlier
tokens, so the model can learn *"by the time I reach `Y`, have I already passed an `X`?"* — the mask
itself is the order signal. So `none` does **not** actually mean "no position information" for a
decoder, and on the full distribution the task can't tell apart methods that rely on positional
encoding from those that don't: the causal mask alone solves it. (Detection was *position-invariant*
— its answer didn't depend on where the symbol sat — so it too was trivially solvable, but for a
different reason: here it's the causal mask's implicit order, not invariance.)

## Caveats / limitations

- **Single seed.** Runs use seed 1337 only — fine when the result is already 100%/100%, but the
  held-out experiment (where results are not maxed out) uses a 4-seed sweep.

## Relation to prior work

The "causal mask supplies order" point is the documented NoPE behavior of decoder-only
transformers: **Kazemnejad et al. (2023)**, *The Impact of Positional Encoding on Length
Generalization in Transformers* (NeurIPS 2023; abstract verified — they note a NoPE transformer can
reconstruct absolute position from the causal mask).

## Next

1. **Held-out splits** — hold out positions and re-run `none` vs `learned`; this is where a real
   generalization gap can appear → [2026-06-20-relative-order-heldout-splits.md](2026-06-20-relative-order-heldout-splits.md).
2. **Full PE sweep** afterward (`sinusoidal` / `rope` / `t5`) on the split that separates methods.
