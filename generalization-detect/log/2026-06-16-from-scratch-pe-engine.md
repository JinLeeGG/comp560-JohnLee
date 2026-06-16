# From-Scratch Engine — swappable positional encoding

*2026-06-16 · gate passed: learned PE → 100% (matches the nanoGPT baseline) · roadmap: Phase 3 (engine)*

---

This log records the move **off nanoGPT** onto a from-scratch micro-transformer whose
positional encoding is a swappable module — the experiment platform for the PE sweep. The
milestone goal was deliberately narrow: build the engine with **all five PE branches wired
in**, fully implement only `learned`, and prove it reproduces the known detection baseline
before filling in the rest. Nothing about the task changed; only the engine did.

## What was built

New files in `generalization-detect/` (no nanoGPT import anywhere):

| file | role |
|------|------|
| `pos_encoding.py` | all 5 PE variants behind one interface |
| `model.py` | task-agnostic decoder transformer (returns logits `(B,T,vocab)`) |
| `train.py` | per-example batching, **loss only on the answer token** |
| `test_engine.py` | Stage-1 sanity suite (below) |
| `evaluate.py`, `verify_eval.py` | adapted to load the new model instead of nanoGPT's GPT |

**Design contract — `pos_type` is the only independent variable.** Each PE plugs into
exactly one of three branch points and is a no-op at the other two, so `model.py` has *no*
`pos_type` branching: it calls all three hooks unconditionally and the PE object decides
what runs.

| pos_type | A: add to embedding | B: rotate q,k | C: attention-logit bias |
|----------|---------------------|---------------|--------------------------|
| `none` | — | — | — |
| `learned` | learned vector | — | — |
| `sinusoidal` | sin/cos vector | — | — |
| `rope` | — | rotate q,k | — |
| `t5` | — | — | relative bias |

This milestone: **`none` and `learned` are fully implemented**; `sinusoidal`, `rope`, `t5`
are wired-in stubs that raise `NotImplementedError` on their active branch (next milestone
fills in the formulas — "fill in the blank," not "restructure").

**One deliberate change from the baseline:** training now uses per-example rows with
cross-entropy on the **answer token only** (the token after `:`), not nanoGPT's
flat-stream LM loss. The 20 digits are random, so supervising them is wasted signal — this
removes the flat-stream misalignment *and* makes the loss curve meaningful (see Run 2).

---

## Run 1: Stage-1 sanity checks (no training)

**Purpose:** Catch engine bugs cheaply before spending any compute. These verify the
wiring is correct independent of whether the model has learned anything.

**Config:** `test_engine.py`, five checks:
1. **PE interface contract** — every `pos_type` builds and exposes all three hooks;
   inactive branches are true no-ops; stub modes raise `NotImplementedError` on their
   active branch only.
2. **Forward shape + no NaN** — `learned`: `(8,22)` ids → `(8,22,15)` logits, all finite.
3. **Causal masking holds** — altering the *last* token must leave all earlier positions'
   logits bit-identical (no future-token leakage), while the last position *does* change.
4. **No duplicated PE params** — the learned position embedding appears exactly once in
   `state_dict()` (checks the non-registered-reference trick in attention).
5. **Param count** — micro range (<1M), near the ~0.79M baseline.

**Results:** all 30 assertions PASS. Param count **0.803M** (matches baseline); learned PE
appears once as `pe.pos_emb.weight`; causal masking confirmed.

**Output (screenshots):**

**Stage-1 suite** (`test_engine.py`)
<!-- paste test_engine.py screenshot here -->
```
=== 1. PE interface contract ===
  [PASS] none/learned/sinusoidal/rope/t5: builds + exposes 3 hooks; correct no-ops; stubs raise
  ... (22 checks)
=== 2. Forward shape + no NaN (learned) ===  [PASS] (8, 22, 15), finite
=== 3. Causal masking holds ===              [PASS] earlier positions unchanged; last changes
=== 4. No duplicated PE params (learned) ===  [PASS] keys=['pe.pos_emb.weight']
=== 5. Param count sanity ===                 [PASS] 0.803M (803,456)
ALL STAGE-1 CHECKS PASSED
```

---

## Run 2: Verification gate — learned PE, full distribution (SPLIT=none)

**Purpose:** The real pass condition. Train `pos_type='learned'` on the full-distribution
detection data (X allowed in every position) and reproduce the Phase-0 nanoGPT baseline of
100% on both classes. Only if this passes do we fill in the other PE modes.

**Config:**
- **Model:** `n_layer=4, n_head=4, n_embd=128, block_size=64` → **0.80M params**, CPU
- **Training:** per-example batches of 64, **answer-token-only loss**, 2000 iters, AdamW
  lr 1e-3 (warmup 100 + cosine decay) — same hyperparameters as Phase 0
- **Data:** 50k / 5k / 2k (train / val / test), balanced Y/N, length 20, seed 1337, X anywhere

**Example data** (from the run — X may appear at any of the 20 positions):
```
TRAIN (X anywhere, 0–19):
  13209659612789771407:N     ← no X
  6906549923X674605480:Y     ← X at position 10
  005X5926483998683163:Y     ← X at position 3
TEST (same distribution):
  8906078089431X282016:Y     ← X at position 13
  36630763653087640211:N     ← no X
```

**Results:**
- Y (X present): **100%** (1000/1000)
- N (X absent): **100%** (1000/1000)
- Overall: **100%** (2000/2000) — matches the nanoGPT Phase-0 baseline.
- **Loss is now meaningful:** answer-token loss dropped **2.66 → ~0.00 by iter ~250**,
  hitting 100% val accuracy — unlike the old flat-stream loss that sat flat at ~2.10.

**Output (screenshots):**

**1. Training** (`train.py` — loss drops to ~0)
<!-- paste train.py screenshot here -->
```
iter     0: val loss 2.6496 | val acc 12.32%
iter   250: val loss 0.0008 | val acc 100.00%
iter  2000: val loss 0.0000 | val acc 100.00%
done in 93.2s | best val acc 100.00%
```

**2. Evaluation — result** (`evaluate.py`)
<!-- paste evaluate.py screenshot here -->
```
=== Detection Evaluation ===
pos_type : learned
Y (X present): 1000/1000 = 100.00%
N (X absent) : 1000/1000 = 100.00%
OVERALL      : 2000/2000 = 100.00%
```

---

## Run 3: Held-out re-check — learned PE, single split (SPLIT=single, position 12)

**Purpose:** Confirm the new engine behaves *identically to nanoGPT on the known case* —
the Phase-1 result that detection generalizes to a held-out position. This is the whole
point of the gate: same answer on the case we already understand.

**Config:** same model/training as Run 2; train X in positions 0–11 & 13–19, test X only
at position 12 (never seen in training).

**Results:**
- Y (X present, all at position 12): **100%** (1000/1000)
- N (X absent): **100%** (1000/1000)
- Controlled probe (`verify_eval.py`): flipping X in/out at position 12 flips the answer
  N↔Y correctly — same as every trained position.

Reproduces Phase 1 exactly: detection is position-invariant, so the held-out position is
solved automatically.

**Output (screenshots):**
<!-- paste train.py + evaluate.py screenshots here -->
```
evaluate.py (test X only at position 12):
  Y (X present): 1000/1000 = 100.00%
  N (X absent) : 1000/1000 = 100.00%
verify_eval.py:  X@12 (held-out) -> Y   (expect Y)   --> probes ALL PASS
```

---

## Conclusion

The from-scratch engine reproduces the nanoGPT baseline exactly — 100% Y / 100% N on the
full distribution, and the same trivial-but-correct generalization to a held-out position.
With the suspects ruled out by the Stage-1 checks (interface, masking, shapes, param
sharing) and the gate green, the engine is **trusted**. As a bonus, the answer-token-only
loss makes the training curve informative for the first time (2.66 → 0), so future
non-saturated runs will have a usable loss signal alongside accuracy.

The single independent variable — `pos_type` — is in place, with `none`/`learned` working
and the other three wired but stubbed. The architecture is now a fill-in-the-blank away
from the PE comparison, with no restructuring needed.

## Next

1. **Fill in `sinusoidal`, `rope`, `t5`** per their textbook definitions (Vaswani 2017
   table; standard q/k rotation; bucketed relative bias). Re-run this same gate for each so
   every PE clears the baseline before being compared.
2. **Add a 3–5 seed sweep** — needed once results stop saturating.
3. **Run the 5-way PE comparison on a position-dependent task** (relative order, Task 2),
   where held-out positions can genuinely break generalization — detection cannot, so it
   only ever served as the engine's sanity check.
