# Detection Baseline — X at all positions
*Phase 0 · June 8, 2026 · ✅ 100%*

---

## Run 1: SPLIT=none (X allowed in all 20 positions)

**Purpose:** Verify the pipeline works end-to-end and hits a high baseline, before
holding any position out.

**Config:** 0.79M-param model (`n_layer=4, n_head=4, n_embd=128, block_size=64`), CPU,
`max_iters=2000`. Data: 50k / 5k / 2k train / val / test, balanced Y/N, length 20, seed 1337.

**Example data** (from the run — X may appear at any of the 20 positions, in both train and test):
```
TRAIN (X anywhere, 0–19):
  6906549923X674605480:Y     ← X at position 10
  005X5926483998683163:Y     ← X at position 3
  13209659612789771407:N     ← no X
TEST (same distribution — X anywhere):
  8906078089431X282016:Y     ← X at position 13
  36630763653087640211:N     ← no X
```

**Results:**
- Y (X present): **100%** (1000/1000)
- N (X absent): **100%** (1000/1000)
- Overall: **100%** (2000/2000) — baseline met.

**Output:**

<img width="444" height="125" alt="evaluate.py output" src="https://github.com/user-attachments/assets/3628e4a8-7a78-4630-9494-c5011c537c88" />

---

## Takeaways

- **Ignore training loss for this task.** It sits at ~2.10 no matter what, because ~20
  of every 23 tokens are random digits the model can't predict — the Y/N answer is just
  1 token. Judge by `evaluate.py` accuracy, not loss.
- **Speed fix:** set `gradient_accumulation_steps=1` (nanoGPT defaults to 40 → ~40×
  slower on CPU).
- **Watch for Phase 1:** examples are stored as one flat stream, so X positions can
  blur across the training window. Fine for this baseline; may need aligned examples
  once we hold a position out.

## Next

**Phase 1** — `SPLIT='single'` (hold out position 12): does accuracy drop there? Report
a per-position accuracy graph, run 3–5 seeds.
