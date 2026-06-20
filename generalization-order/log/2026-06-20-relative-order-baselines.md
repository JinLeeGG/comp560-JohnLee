# Relative-Order Task — full-distribution baselines (none vs learned)

*2026-06-20 · both PEs → 100%, including `none` (NoPE) · surprise: causal mask alone solves order · roadmap: Phase 2 → held-out splits next*

---

This log records the first run of **Task 2 (relative order)**: one `X` and one `Y` in a
fixed-length-20 string, output `T` if `X` comes before `Y`, else `F`. The goal for this
milestone was narrow — build a clean **full-distribution** dataset and measure two
baselines before the meeting: `none` (expected ~chance) and `learned` (expected high).
Held-out position splits are deliberately **not** built yet.

The engine is unchanged from the detection milestone (`model.py` + `pos_encoding.py`
copied verbatim); only the data (`data/order/prepare.py`) and config are task-specific.
Per-example batching and **answer-token-only loss** carry over unchanged because the
answer is still the last token.

## What was built

| file | role |
|------|------|
| `data/order/prepare.py` | relative-order generator; 50/50 T/F; label-correctness assertion |
| `config/basic.py`, `train.py`, `evaluate.py` | detection structure, `data_dir=data/order`, per-class T/F |
| `verify_order.py` | matched-pair order probe (isolates *order* as the only variable) |

**Label mapping (fixed, never flipped):** `T iff index(X) < index(Y)`.

---

## Run 1: Data generation + built-in checks

**Purpose:** Catch any placement/label bug *before* training — now that position
determines the answer, a flipped label or misplaced symbol would silently corrupt the
whole experiment.

**Config:** `prepare.py`, length 20, seed 1337, 50k / 5k / 2k (train/val/test), all from
the full position distribution. Vocab size **read from the built vocab** (not hardcoded).

**Results:** vocab_size **16** (`0`–`9`, `X`, `Y`, `T`, `F`, `:`, `\n`); every pool
exactly 50/50 T/F; **label-correctness assertion passed for all 57,000 examples**
(`index(X) < index(Y)` iff `T`).

**Output:**
```
=== Relative-order data prepared ===
vocab_size=16
train:  50000 examples  | T=25000 F=25000  (50.0% T)
val  :   5000 examples  | T=2500 F=2500  (50.0% T)
test :   2000 examples  | T=1000 F=1000  (50.0% T)
label-correctness assertion passed for all pools  (index(X)<index(Y) iff T)

First 5 train examples (X/Y positions marked):
  X@13 Y@ 9 -> F   987570430Y204X382685:F
  X@ 3 Y@11 -> T   869X7678304Y87307021:T
  X@11 Y@14 -> T   42101049019X49Y56391:T
```

---

## Run 2: Baseline `pos_type='none'` (NoPE) — full distribution

**Purpose:** Sanity expectation was **~50% (chance)**: with no positional information,
a model "shouldn't" be able to tell which symbol comes first. A result near 50% would
confirm the task is genuinely position-dependent (unlike detection). The spec flagged
the opposite outcome as a stop-and-look condition.

**Config:** `n_layer=4 n_head=4 n_embd=128 block_size=64` (~0.8M params), CPU; 2000 iters,
AdamW lr 1e-3 (warmup 100 + cosine), batch 64, answer-token-only loss, seed 1337.

**Results — surprise: 100%, not 50%.**
- T (X before Y): **100%** (1000/1000)
- F (Y before X): **100%** (1000/1000)
- Overall: **100%** (2000/2000); val acc hit 100% by iter ~250, loss → 0.

**Output:**
```
=== Relative-order Evaluation ===
pos_type : none
T (X before Y): 1000/1000 = 100.00%
F (Y before X): 1000/1000 = 100.00%
OVERALL       : 2000/2000 = 100.00%
```

---

## Run 3: Order probe on the `none` model — is it a data leak?

**Purpose:** The spec says "if `none` scores well above 50%, something is leaking
position info." Before concluding, isolate *order* as the only variable: for 1000 random
trials, draw one digit body and one position pair `(i<j)`, then build two **byte-identical**
inputs differing only in which symbol is first — `X@i,Y@j` (gold T) vs `Y@i,X@j` (gold F).
The bag of tokens is identical between the two, so the only thing that can flip the
prediction is *order*.

**Config:** `verify_order.py --n_trials=1000` on the `none` checkpoint.

**Results:** **1000/1000 matched-pair flips correct.** The model predicts `T` on the
X-first version and `F` on the Y-first version of the *same* body — so it genuinely reads
order. **This rules out a content/data leak:** the data design is clean.

**Output:**
```
=== Relative-order PROBE (order is the only variable) ===
pos_type : none
both-correct flips: 1000/1000 = 100.00%
```

**Diagnosis (not a data bug — an architecture property):** a **causal decoder is not
permutation-invariant**. The causal mask lets the representation at each position depend
only on tokens at or before it, so the model can learn *"by the time I reach `Y`, have I
already seen an `X`?"* — i.e., the causal mask itself supplies the ordering signal that
explicit positional encoding was supposed to provide. So `none` ≠ "no position info" for a
decoder. This is the documented NoPE behavior of decoder-only transformers (cf. the
positional-encoding paper on the reading list — to review before citing).

---

## Run 4: Baseline `pos_type='learned'` — full distribution

**Purpose:** Confirm the task is solvable and the engine works on it (expected high).

**Config:** identical to Run 2 except `pos_type=learned`.

**Results:** T **100%**, F **100%**, overall **100%** — as expected.

**Output:**
```
=== Relative-order Evaluation ===
pos_type : learned
T (X before Y): 1000/1000 = 100.00%
F (Y before X): 1000/1000 = 100.00%
OVERALL       : 2000/2000 = 100.00%
```

---

## Conclusion

The intended meeting story was **none ≈ 50%, learned ≈ high ⇒ positional embedding is
decisive**. The data refuted it: **both score 100%**, and the probe proves the `none`
result is real (genuine order computation, no content leak). On the **full distribution**,
relative-order does *not* separate "needs PE" from "doesn't" — the **causal mask alone**
gives a decoder enough ordering to solve it.

This does not kill the task; it relocates the interesting question. Like detection (which
was position-*invariant*), relative-order is *trivially solvable on the full distribution* —
but for a different reason (implicit causal-mask order, not invariance). The real test is
the **held-out-position split**: does that implicit ordering, and do the explicit PEs,
**generalize to position pairs never seen in training**? That is where the methods may
finally diverge, and it is the next experiment.

## Next

1. **Design the held-out position-pair split** (single pair / even-odd / first-half vs
   second-half) — the open question deferred to the meeting, since both `X` and `Y` carry
   positions.
2. **Re-run none vs learned on that split** — the first chance for a genuine generalization
   gap to appear.
3. **Fill in `sinusoidal` / `rope` / `t5`**, then run the 5-way PE comparison on the split.
4. **Add a 3–5 seed sweep** once a config sits near a generalization boundary.
