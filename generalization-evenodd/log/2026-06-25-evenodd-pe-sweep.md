# Even/odd separation — a capacity wall below the PE comparison (Phase 3, distance task #1)

*2026-06-25 · commit `<pending>` · John Lee*

> **Bottom line.**
>
> 1. A micro-transformer (<1M params) **cannot learn even/odd distance-parity at all** —
>    every PE, every seed sits at exactly 50% on *in-distribution* val (loss locked at
>    ln 2), so the intended NoPE-vs-distance held-out comparison is moot at micro scale.
> 2. The task is genuinely learnable, though: a deliberately oversized 4.76M model reaches
>    **100% in-distribution** in ~2000 iters — making this a **capacity wall**, not a
>    degenerate task.
> 3. At that learnable scale, absolute PE (`learned`) **learns the task but fails to
>    generalize** to held-out positions (44.85%, below chance) — the same
>    "learns-but-doesn't-generalize" signature seen on relative order, now on a *distance*
>    task.

### The task

- **Input** — a fixed string of **20 characters**. The symbol `X` appears
  **exactly twice** (no `Y`; the two targets are identical); the other 18 are
  random digits.
- **Distance** — `|pos2 − pos1|`, how far apart the two `X`'s are.
- **Label** — `T` if that distance is **even**, `F` if it is **odd**.

```
482X50178X4019285746 : T     X at 3 and 9  →  distance 6 (even)  →  T
482X5017X64019285746 : F     X at 3 and 8  →  distance 5 (odd)   →  F
```

The five positional encodings compared (the one knob that changes, `pos_type`):

- `none` — **NoPE**: no positional encoding at all
- `learned`, `sinusoidal` — **absolute** PE (encodes each fixed slot)
- `rope`, `t5` — **relative / distance-aware** PE (encodes the gap between positions)

Even/odd of a difference is the XOR of the two position-parities, so the task reduces to
*"are the two X's on positions of the same parity?"* — a parity-flavored function.

---

## Setup  <!-- R1/R3/R6 -->

- **data:** length 20, two `X`'s; pools 50k train / 5k val / 2k test, all **50/50 T/F**
  (chance = 50%).
- **split (`half`):** train/val both X in 0–9; held-out test both X in 10–19. The held-out
  test is **distance-balanced** (distances 2/4/6/8 → 250 each, 1/3/5/7/9 → 200 each).
- **checks:** `prepare.py` asserts label-correctness (`|pos2−pos1|` even iff T) and the
  split rule.
- **model:** the micro baseline — `n_layer=4, n_head=4, n_embd=128` (**0.80M params**),
  `block_size=64`.
- **optimizer / schedule:** AdamW · `lr=1e-3` cosine · `max_iters=2000` · `batch_size=64`.
- **loss / device:** answer-token-only loss · CPU.
- **seeds:** 1337–1340 (vary model init + batch order only; **one fixed data split**).
- **env:** python 3.9.6 · torch 2.8.0 · numpy 2.0.2 · matplotlib 3.9.4; data `SEED=1337`.
- **reproduce:**
  ```bash
  ../venv/bin/python data/evenodd/prepare.py
  for pe in none learned sinusoidal rope t5; do for s in 1337 1338 1339 1340; do
    ../venv/bin/python train.py    config/basic.py --pos_type=$pe --seed=$s
    ../venv/bin/python evaluate.py config/basic.py --pos_type=$pe --seed=$s
  done; done
  ../venv/bin/python plot.py --split=half --out_dir=log/figures
  ../venv/bin/python plot.py --split=half --separation --out_dir=log/figures
  ```

---

## Results  <!-- R5/R7/R9 -->

### 1. Micro-scale PE sweep (the planned experiment) — **uniform null**

All 20 runs (5 PE × 4 seeds) collapse to a single label; **in-distribution val is also
50%**, so nothing was learned even on the training region.

| PE | in-dist. val acc | held-out test acc | held-out per-class |
|----|------------------|-------------------|--------------------|
| `none` | 50.0% | 50.0% | 100/0 or 0/100 (seed-dependent collapse) |
| `learned` | 50.0% | 50.0% | 100/0 or 0/100 |
| `sinusoidal` | 50.0% | 50.0% | 100/0 or 0/100 |
| `rope` | 50.0% | 50.0% | 100/0 or 0/100 |
| `t5` | 50.0% | 50.0% | 100/0 or 0/100 |

Because val never exceeds chance, the held-out columns carry no generalization signal —
the held-out accuracy is just *which label the run collapsed to* crossed with the test
class balance.

### 2. Is it learnable? Diagnostics (single seed 1337, in-distribution val)

| diagnostic | params | iters | lr | best in-dist. val |
|------------|-------:|------:|----|-------------------|
| micro `learned` | 0.80M | **15 000** | 1e-3 | **50.0%** |
| micro `rope` | 0.80M | **15 000** | 1e-3 | **50.0%** |
| micro `learned` | 0.80M | 8 000 | **3e-3** | **50.0%** |
| **big `learned`** (`n_layer=6, n_head=8, n_embd=256`) | **4.76M** | 8 000 | 1e-3 | **100.0%** (by iter ~2000, val loss → 0.0000) |

More iterations (7.5×) and a higher learning rate do **not** move the micro model off 50%;
a ~6× larger model learns it cleanly. **→ capacity wall**, between 0.80M and 4.76M.

### 3. Does the learnable (4.76M) model *generalize*? — **no**

Evaluating the 4.76M `learned` checkpoint on the held-out second half:

| | in-dist. val | held-out test | T (even) | F (odd) |
|--|--------------|---------------|----------|---------|
| 4.76M `learned` (absolute PE) | **100.0%** | **44.85%** (below chance) | 5.9% | 83.8% |

High val + below-chance held-out = **learned the task, did not generalize** — and collapsed
toward one label off-distribution. This mirrors the relative-order result (absolute PE ties
the computation to absolute slot identities seen in training) — here on a *distance* task.

**Figures** (micro sweep; from `results.csv` / `predictions.csv` via `plot.py`):

<img src="figures/accuracy_vs_separation_half.png" alt="accuracy vs separation: all five PEs flat at 50% across every held-out distance" width="640">

**How to read it.** The x-axis is the distance between the two `X`'s (1–9); the y-axis is
accuracy; each line is one PE; the dashed line at 50% is pure guessing. (Each point pools the
4 seeds, n = 80–720 per distance.)

**What it shows.** Every line lies flat on the 50% line at every distance. This was meant to be
*the* graph of the experiment — distance-aware methods (`rope`, `t5`) doing well at short gaps
and tailing off as the gap grows. But because no model learned the task, all five sit flat at
chance; the shape it was designed to reveal only appears once a model actually learns.

The other three figures tell the same story:

<img src="figures/heldout_accuracy_half.png" alt="bar chart: every PE at 50% on both held-out test and in-distribution val" width="640">

**How to read it.** For each PE, the blue bar is accuracy on the held-out positions (10–19,
never trained on); the gray bar is accuracy on in-distribution val (0–9, the same positions it
*did* train on).

**What it shows.** The gray bar is *also* 50% — the model never even got the training region
right. So this is not "learned the task but failed to generalize"; it is "never learned at
all." (A true generalization failure would be a tall gray bar beside a short blue one.)

<img src="figures/heldout_perclass_half.png" alt="per-class held-out accuracy: T and F bars at 50% mean, with per-seed dots at the 0 and 100 extremes" width="640">

**How to read it.** The same held-out accuracy, split by the correct answer: green = T (even
distance) cases, orange = F (odd distance) cases. Bars are the mean over seeds; each dot is one
seed.

**What it shows.** The dots sit at 0% and 100%, not scattered around 50%. Each run does not
guess randomly — it answers the *same* label for everything: some seeds say T to every input
(T = 100%, F = 0%), others say F to every input. Averaged together this looks like 50%, but no
single run is ever "half right."

<img src="figures/per_position_half.png" alt="per-position accuracy heatmaps for all five PEs: every sampled cell uniformly around 50%" width="640">

**How to read it.** One heatmap per PE. A cell is the accuracy when the first `X` sits at the
row position and the second `X` at the column position; color runs red (0%) → yellow (50%) →
green (100%). The black box is the train region (both `X`'s in 0–9), the blue box the held-out
region (both in 10–19); gray cells are position pairs that never occur.

**What it shows.** Every cell is yellow (~50%) — inside the black train box and the blue
held-out box alike. The model solves no placement of the two `X`'s anywhere, which is why even
the training region sits at chance.

---

## Why / interpretation

- **Capacity wall, not a bad task.** The loss snaps to ln 2 within 1000 iters and never
  moves at micro scale, regardless of PE / iters / lr — the optimizer finds the trivial
  50/50 predictor and the gradient toward the parity solution is too weak to escape. A
  4.76M model finds the solution easily, so the function is well-defined and learnable; the
  <1M budget is simply below the threshold for *this* computation.
- **Why even/odd is so much harder than relative order.** Order is readable straight off
  the causal mask ("have I already seen an X?") — a monotone signal. Parity-of-distance
  requires representing position-parity and an exact equality/XOR over the two X's; it has
  no monotone shortcut. *Hypothesis* (parity is known-hard for transformers): worth checking
  against the literature before relying on it — **not yet read, so not cited here.**
- **Absolute PE doesn't generalize across positions (4.76M datapoint).** Consistent with the
  relative-order finding: absolute slot identities learned in the first half don't transfer
  to the unseen second half. The *relative* family (none/rope/t5) at the learnable scale is
  the missing comparison — see Next.

## Caveats / limitations

- **One fixed data split**; the 4 seeds vary only init + batch order. (Same caveat as the
  order sweep.) For the micro null this barely matters — nothing learned under any seed.
- The 4.76M results are **single-seed, single-PE diagnostics** (`learned` only), run to
  answer "is it learnable / does absolute PE generalize," **not** a controlled sweep. They
  intentionally **break the project's <1M micro constraint** and are labeled as such.
- The intended scientific comparison — does **NoPE fail** (can't count distance) while
  **rope/t5** hold up, at the scale where the task is learnable — is **deferred, not
  answered.** MacCormick's predicted accuracy-vs-separation shape cannot be read from a model
  that never learned the task.

## Relation to prior work

The "absolute-PE-learns-but-doesn't-generalize-to-held-out-positions" pattern reproduces the
relative-order result in this repo. The broader parity-hardness framing is left as a
hypothesis pending a literature check (no unread papers cited).

## Next

1. **Decide the path** (raised with advisor): (a) run the PE sweep at the learnable ~4.76M
   scale — above-micro, clearly flagged — to finally test NoPE-fails / distance-wins and
   produce the real separation figure; (b) **find the minimal capacity** that learns even/odd
   (binary-search 0.8M↔4.76M; a Phase-4 "minimal model" result); (c) **scratchpad tokens**
   (Phase 5) to crack it *within* <1M; or (d) pivot to the easier **dist≥D** task, which may
   fit under the micro budget.
2. Whichever path: regenerate data **per seed** to upgrade the variance estimate.
