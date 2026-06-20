# Relative-Order Task — full-distribution baselines (`none` vs `learned`)

*2026-06-20 · Phase 2 · sets up the held-out experiment (next log)*

> **Bottom line.** On the full position distribution, **both** settings solve the task at
> **100%** — including `none`, which has **no positional encoding**. A causal decoder gets the
> order of its tokens "for free" from its causal mask, so this baseline cannot tell the PE methods
> apart. The
> real test therefore moves to **held-out positions** (next log).

### The task (one line)

Length-20 string with **one `X`** and **one `Y`** (rest are random digits). Output the last token:

```
T  if  X is before Y        4829X017364Y19285746 : T
F  if  Y is before X        4829Y017364X19285746 : F      rule (fixed): T ⇔ index(X) < index(Y)
```

The two settings we compare (only this changes between runs):
**`none` = NoPE** (no positional encoding) · **`learned` = APE** (learned absolute position embedding).

---

## Summary

| run | setting | question | result |
|-----|---------|----------|--------|
| 1 | — | is the data clean? | ✅ 50/50 T/F, labels correct (57k examples) |
| 2 | `none` | can a model with **no PE** do it? | **100%** (T 100% / F 100%) |
| 3 | `none` | is run 2 just a data leak? | ✅ no leak — 1000/1000 order-flips |
| 4 | `learned` | can APE do it? (sanity) | **100%** (T 100% / F 100%) |

All runs: **single seed 1337**, full distribution (`X`,`Y` at any positions), 2000 balanced test
examples (chance = 50%). Model ~0.8M params (n_layer 4, n_head 4, n_embd 128), CPU, 2000 iters,
AdamW lr 1e-3, batch 64, answer-token-only loss.

---

## Run 1 — Data + integrity checks
**Purpose.** Position now determines the answer, so a flipped label or misplaced symbol would
silently corrupt the experiment. Catch it before training.
**Result.** Vocab size **16** (read from the data, not hardcoded); every pool exactly 50/50 T/F;
the label assertion (`index(X) < index(Y)` ⇔ `T`) passed on all **57,000** examples.
```
train 50000 | val 5000 | test 2000      (each 50.0% T)
label-correctness assertion passed for all pools
  X@ 3 Y@11 -> T   869X7678304Y87307021:T
  X@13 Y@ 9 -> F   987570430Y204X382685:F
```

## Run 2 — `none` (NoPE): can a model with no PE do it?
**Purpose.** Expected ~50% (chance): "no position info ⇒ can't tell which symbol came first." The
spec flagged the opposite as a stop-and-check condition.
**Result — surprise, 100% not 50%.** T 100% (1000/1000), F 100% (1000/1000); val reached 100%
early in training (val loss ≈ 0).

## Run 3 — Is that a data leak? (order-only probe)
**Purpose.** Isolate *order* as the only variable. For 1000 trials, take one digit body + one
position pair (i < j) and build two **byte-identical** inputs differing only in which symbol is
first: `X@i, Y@j` (→ T) vs `Y@i, X@j` (→ F). Same multiset of tokens, so **only order** can flip
the prediction.
**Result.** **1000/1000** flips correct → the model genuinely reads order. **Not a data leak.**
```
=== order-only probe (none model) ===   both-correct flips: 1000/1000 = 100%
```

> **Why NoPE still works.** A causal decoder is *not* order-blind. The causal mask lets each
> position attend only to earlier tokens, so the model can learn *"by the time I reach `Y`, have I
> already passed an `X`?"* — the mask itself is the order signal. (Documented NoPE behavior:
> Kazemnejad et al. 2023, NeurIPS.)

## Run 4 — `learned` (APE): sanity check
**Purpose.** Confirm the task is solvable and the engine is sound. **Config.** As Run 2, with
`pos_type=learned`. **Result.** T 100% / F 100% — as expected (see Summary).

---

## Conclusion

The intended story was *"none ≈ 50%, learned ≈ high ⇒ positional encoding is decisive."* The data
**refuted** it: both score 100%, and the probe shows `none`'s result is genuine order computation,
not a leak. On the **full distribution**, relative order does not separate "needs PE" from
"doesn't" — the **causal mask alone** gives a decoder enough order to solve it.

This doesn't kill the task; it **relocates** the interesting question. Like detection (which was
position-*invariant*), relative order is trivially solvable on the full distribution — but for a
different reason (implicit causal-mask order, not invariance). The real test is whether that
implicit order, and the explicit PEs, **generalize to positions never seen in training**.

*Single-seed caveat:* runs 2 and 4 use seed 1337 only. That is fine for a 100%/100% saturated
baseline, but the held-out experiment (where results are not saturated) uses a multi-seed sweep.

## Next
1. **Held-out splits** — hold out positions and re-run `none` vs `learned` (next log; this is
   where a real generalization gap can appear).
2. **Fill in `sinusoidal` / `rope` / `t5`**, then run the 5-way PE sweep on the split that separates
   methods.
3. **Seed sweep** wherever results are not saturated.
