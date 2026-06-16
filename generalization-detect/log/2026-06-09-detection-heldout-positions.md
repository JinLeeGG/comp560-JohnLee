# Held-Out Positions — detection is position-invariant
*2026-06-09 · result: generalizes 100%, but for a trivial reason · roadmap: Phase 1*

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
<img width="559" height="366" alt="image" src="https://github.com/user-attachments/assets/0c752b66-c8de-40fd-893e-a36be0085007" />


**2. Training** (`train.py`)
<!-- paste train.py screenshot here -->
<img width="358" height="536" alt="image" src="https://github.com/user-attachments/assets/a48e6d6d-4e1e-4950-a266-b9600cba8916" />


**3. Evaluation — result** (`evaluate.py`)
<!-- paste evaluate.py screenshot here -->
<img width="276" height="89" alt="image" src="https://github.com/user-attachments/assets/77c5a101-1c8f-4ea2-a6e6-725e2e0761cb" />


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
<img width="559" height="349" alt="image" src="https://github.com/user-attachments/assets/aa797db7-bffa-4409-8cdd-f62d6b043fd1" />


**2. Training** (`train.py`)
<!-- paste train.py screenshot here -->
<img width="386" height="538" alt="image" src="https://github.com/user-attachments/assets/8d8d5d9e-9469-488e-9617-a1553834ebec" />


**3. Evaluation — result** (`evaluate.py`)
<!-- paste evaluate.py screenshot here -->
<img width="288" height="89" alt="image" src="https://github.com/user-attachments/assets/3d8ec581-0240-4fdc-8974-44f8aec811cb" />


---

## Conclusion

The model scored 100% even on positions it never trained on — at first glance, perfect
generalization. But the reason is almost a trick: **to answer "is there an X?", you don't
need to know *where* the X is.** Detection is **position-invariant** (the answer doesn't
depend on the X's location), so the model just learns to spot the X token *anywhere*, and a
position it never trained on works automatically. The held-out split never really
challenged it.

**A PE *was* active here.** These runs used nanoGPT's default **learned absolute PE**
(`wpe = nn.Embedding(block_size, n_embd)`, added to the token embeddings), so the 100% is
*despite* a positional encoding being present — not because there was none. A
position-invariant task simply makes whatever PE is present irrelevant to the answer.
(Confirmed 2026-06-16 on the from-scratch engine: re-running with `pos_type='none'` — no
positional encoding at all — also scores 100% / 100%, including at held-out position 12.
Removing the PE changes nothing here, because the task never needed it.)

So detection sits at the **easy end**: it always generalizes, because position doesn't
matter to the answer. To find out *when* generalization actually **fails**, we need a task
whose answer **depends on position** — that's the next step (relative order: is X before Y?).

**Two objections this rules out:**
- *Not a data-alignment artifact.* Even with the flat-stream misalignment (Phase 0 open
  item) fixed, this would still be ~100%, because the model only cares whether X appears,
  not where. (We still fix alignment before the position-dependent task.)
- *No seed sweep here.* A saturated 100% with a known cause won't change across seeds —
  seed sweeps earn their keep when results are noisy or sit near a generalization
  boundary, which is exactly what Task 2 should produce. I'll run multiple seeds there.

## Next

Move to **Task 2 (relative order):** one X and one Y, output Y if X comes before Y. The
answer depends on the two symbols' positions, so held-out positions can genuinely break
generalization. Build its `prepare.py` with **aligned examples** so the held-out-position
condition is clean.
