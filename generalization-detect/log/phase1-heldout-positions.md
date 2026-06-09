# Held-Out Positions — detection is position-invariant
*Phase 1 · June 9, 2026 · result: generalizes 100%, but for a trivial reason*

---

## Run 1: hold out a single position (SPLIT=single, position 12)

**Purpose:** First generalization test. Train with X in every position except 12, then
test whether the model detects X at position 12 — a position it never saw during training.

**Config:**
- **Model:** same as Phase 0 (0.79M params, 2000 iters, CPU)
- **Split:** train/val X in positions 0–11 & 13–19 · test X only at position 12
- **Test set:** 1,000 Y / 1,000 N

**Example data** (illustrating the split — X position annotated):
```
TRAIN (X anywhere except 12):
  482X9501736490285746:Y     ← X at position 3
  90345678120948573610:N     ← no X
TEST  (X only at position 12):
  482950173649X2857460:Y     ← X at position 12  (never trained here)
  66103948572610495837:N     ← no X
```

**Results:**
- Y (X present, all at position 12): **100%** (1000/1000)
- N (X absent): **100%** (1000/1000)

**Output (screenshots):**

**1. Data prep** (`prepare.py` — confirms test X-positions = [12])
<!-- paste prepare.py screenshot here -->

**2. Training** (`train.py`)
<!-- paste train.py screenshot here -->

**3. Evaluation — result** (`evaluate.py`)
<!-- paste evaluate.py screenshot here -->

---

## Run 2: hold out the whole second half (SPLIT=half, train 0–9 / test 10–19)

**Purpose:** Stress test. Position 12 alone sits between trained positions 11 and 13, so
Run 1 only needed easy "interpolation." Here the model never sees X anywhere in positions
10–19, so it must extrapolate to an entire unseen region.

**Config:**
- **Model:** same as Phase 0
- **Split:** train/val X in positions 0–9 only · test X in positions 10–19

**Example data** (from the actual run — note X is in the left half for train, right half for test):
```
TRAIN (X only in 0–9):
  0X595926483998683163:Y     ← X at position 1
  69065X99234674605480:Y     ← X at position 5
  77291033086568562700:N     ← no X
TEST  (X only in 10–19):
  8906078089431228X016:Y     ← X at position 16  (never trained in this half)
  91186054015461848763:N     ← no X
```

**Results:**
- Y (X present, all in positions 10–19): **100%** (1000/1000)
- N (X absent): **100%** (1000/1000)

**Output (screenshots):**

**1. Data prep** (`prepare.py` — confirms test X-positions = [10–19])
<!-- paste prepare.py screenshot here -->

**2. Training** (`train.py`)
<!-- paste train.py screenshot here -->

**3. Evaluation — result** (`evaluate.py`)
<!-- paste evaluate.py screenshot here -->

---

## Conclusion

The model generalizes to held-out positions perfectly — even with an entire half of the
positions unseen. But this is **not** deep position generalization. **Detection is
position-invariant:** "is X present?" does not depend on *where* X is, so the model learns
a single position-general X-presence detector instead of per-position detectors, and any
held-out position transfers for free.

This is still useful: it pins down the **"trivially generalizes" end** of the spectrum. To
see *when* generalization fails, the next task's answer must **depend on position**.

**Caveats:**
- This isn't a data artifact — even with perfectly aligned examples it would still be
  ~100%, because the model keys on the X token's *presence*, not its location. But the
  flat-stream misalignment (Phase 0 open item) does make "held out" less clean and must be
  fixed for position-dependent tasks.
- No seed sweep here: the result is 100% with an understood cause, so there's no variance
  to measure. Seed sweeps matter once a task sits near a generalization boundary.

## Next

Move to **Task 2 (relative order):** one X and one Y, output Y if X comes before Y. The
answer depends on the two symbols' positions, so held-out positions can genuinely break
generalization. Build its `prepare.py` with **aligned examples** so the held-out-position
condition is clean.
