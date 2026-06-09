# Detection Baseline — X at all positions
*Phase 0 (reorient infrastructure) · June 8, 2026 · Status: ✅ 100% baseline*

---

## Run 1: SPLIT=none, full position distribution

**Purpose:** Verify the detection pipeline end-to-end and confirm a high baseline when
`X` is allowed in *every* position. This calibrates the measuring "ruler" before any
held-out-position split.

**Config:**
- Data: `SPLIT='none'`, `LENGTH=20`, 50k train / 5k val / 2k test, balanced 50/50, `SEED=1337`
- Model: `n_layer=4, n_head=4, n_embd=128, block_size=64, dropout=0` → **0.79M params** (<1M ✓)
- Training: `batch_size=64, gradient_accumulation_steps=1, max_iters=2000, lr=1e-3, device=cpu`

**Results:**
- **Y (X present): 100% (1000/1000)**
- **N (X absent): 100% (1000/1000)**
- **Overall: 100% (2000/2000)** — baseline goal (99%+) met.
- Training loss plateaus at ~2.10 and barely moves, *even though accuracy is 100%*.
  This is expected: loss averages next-token prediction over all 23 tokens of an
  example, ~20 of which are random digits (irreducible loss ≈ ln(10) ≈ 2.30 each). The
  Y/N decision is only 1 token in 23, so solving it shifts the average loss by ~0.03.
  **This task must be judged by `evaluate.py` accuracy, not by val loss.**

**Output:**

<!-- paste screenshot of evaluate.py output here -->
<img width="444" height="125" alt="image" src="https://github.com/user-attachments/assets/3628e4a8-7a78-4630-9494-c5011c537c88" />


---

## Conclusion

**What worked:**
- Pipeline verified end-to-end; clean **100%** baseline on the full position distribution.
- Building a separate, task-specific evaluator (per-class Y/N accuracy on a held-out
  `test.txt`) was the right call — it is the only trustworthy signal for this task.

**What didn't work / caveats:**
- **Training loss is not a usable signal here** (flat ~2.10 regardless of learning).
  Unlike the Korean-translation task — where the loss curve was meaningful — a Wandb
  *loss* graph would be flat and uninformative for detection. The meaningful curve for
  this task is **per-position accuracy** (see Phase 1).
- Examples are stored as a flat `\n`-joined stream, and nanoGPT's `get_batch` samples
  random windows, so examples can be misaligned to absolute positions. Harmless for
  this `none` baseline (all positions trained), but the held-out-*position* experiment
  may need aligned examples so "X never seen at absolute position p" is a clean
  condition. Revisit before trusting Phase 1 numbers.

**Changes made:**
- Built `generalization-detect/` (`data/detect/prepare.py`, `config/basic.py`, `evaluate.py`).
- Set `gradient_accumulation_steps=1` — nanoGPT defaults to 40, which made each CPU
  iteration ~2.5s (~80 min for 2000 iters); 1 → ~2 min total.

---

## Next

**Phase 1:** set `SPLIT='single'` (hold out position 12), re-run prepare → train →
evaluate, and measure whether held-out-position accuracy drops. Report the per-position
accuracy graph, and run 3–5 seeds (mean/variance).
