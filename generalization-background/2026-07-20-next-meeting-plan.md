# Next-Meeting Plan — Debugging the length-6 / b=1 held-out failure (T5)

*2026-07-20 · John Lee · follows the 2026-07-20 meeting with Prof. MacCormick*

## One-sentence diagnosis

The length-6, b=1 held-out failure is (very likely) **not** a T5/PE bug but a **readout**
problem. The answer is always read from the **last position** — a fixed absolute anchor — so
"distance from that fixed slot to X/Y" secretly encodes X/Y's **absolute** position. That lets
even a relative-PE model take the position-lookup shortcut ("F iff Y is at slot 2"), which does
not transfer to the held-out second half → the model collapses to answering `T` everywhere
(held-out F accuracy = 0).

> This is the last place absolute position leaks in even when the PE is relative — which is
> exactly what "only relative positioning, remove absolute positioning" (the meeting goal) is
> about. Relative PE is **necessary but not sufficient**; the readout must be position-neutral too.
>
> **Status — leading hypothesis, not yet proven.** The decisive test (swap in a position-neutral
> readout and see if held-out jumps to 100%) is an architecture change, hence advisor-gated
> (Part B). The pre-meeting diagnostics below build a *strong plausible case*, not proof.

## Evidence we already have

- All 40 runs **learn** the task (val = 100%). The failure is *purely* generalization to
  held-out positions.
- Length 6, b=1, no-colon: **bimodal** — 2/10 perfect (100%), 8/10 collapse (exactly 50%, with
  held-out F acc = 0.0 = "answer T to everything"). Nothing in between → two discrete solutions:
  the real gap rule (generalizes) vs. the position-lookup shortcut (does not).
- Colon vs. no-colon barely differs (9/10 → 8/10 collapses at length 6) → the colon marker is
  **not** the cause. Ruled out.
- The earlier version that **reportedly** hit 100% at b=1 used a mean-pool readout (a classifier
  head that averages all positions), not next-token-at-last-position. ⚠️ This is a **one-line log
  caveat, not reproduced here** — it may have differed in other ways too, so treat it as a *lead to
  verify* (Part B), not established fact. It is the reason the readout is the prime suspect.
- `q, k = self.pe.rotate_qk(q, k)` in `CausalSelfAttention` is a **no-op** under `pos_type='t5'`
  (`T5RelativeBias` does not override `rotate_qk`, so it inherits the identity). The suspected
  "RoPE bug" therefore cannot affect the t5 runs. **Closed.**

## Goal for the next meeting

Run the professor's suggested diagnostics, use what they show to build **and pressure-test** the
readout hypothesis, and prepare a fix proposal. Be explicit at the meeting that the pre-meeting
result is a *plausible case*, not proof — the decisive experiment (a readout swap) is an
architecture change and therefore advisor-gated (Part B).

---

## Part A — Diagnostics to run (all from the professor's suggestions)

### A1. T5 bias table + position heatmap — first-cut PE-vs-readout test (NOT decisive)

**Purpose.** Narrow the cause using two cheap views on the *current* model: (a) what the PE
*learned* about distance, and (b) what the model *actually does* across positions.

**Steps.**
1. **Bias table.** Small script: load a trained checkpoint, read `model.pe.rel_bias.weight`
   (`(num_buckets=32, n_head)`), map bucket → relative distance via `T5RelativeBias._bucket`, and
   plot bias vs. distance, one line per head. Do it for a **generalizing seed (1337)** and a
   **failing seed (1338)**, side by side.
2. **Position heatmap (already implemented — use it).** Run the evaluate.py diagnostic sweep
   (`predictions.csv`) → plot.py per-position heatmap for the **failing seed**. Accuracy that
   tracks the **gap** (constant along diagonals) = the model reads relative distance; accuracy
   that tracks **where Y sits** = a position shortcut. This visualizes the shortcut more directly
   than the bias table does.

**How to read it (neutral).**
- Bias clean in *both* seeds **and** the failing seed's heatmap tracks Y-position → consistent with
  "PE learned distance, model still took a position shortcut" → points downstream (readout /
  attention routing). *(Our leading hypothesis.)*
- Failing seed's bias is **flat** → the PE itself didn't learn distance → points at the PE, not the
  readout.

**⚠️ First cut, not decisive.** The bias shows what was *learned*; the heatmap shows position-vs-gap
*behavior*. Together they strongly narrow the cause, but they still cannot fully separate a
**readout** problem from an **attention-routing (Q/K)** problem — both live downstream of the bias.
The decisive test is the mean-pool readout swap (Part B), which is advisor-gated.

**Notes.** Plot **per seed**, never pooled (earlier lesson: pooled figures can fake a
distance-looking pattern out of a collapsed-to-one-class model). Requires the 1337/1338
checkpoints — if `train.py` overwrote them across the seed sweep, re-run just those two seeds
(~15 s each) to regenerate.

### A2. Shrink the model — make the diagnostics legible (keep depth ≥ 2) — LOWEST PRIORITY / DO LAST

**Priority note.** This is a *legibility aid for A1*, not a standalone diagnostic, so it cannot
precede A1 and is **contingent**: only needed if A1's bias plot on the current model (3 layers /
2 heads) is hard to read — the 2-head plot is likely legible enough on its own. It also changes
the model, which would break the link to the documented bimodal behavior (seed 1337 = 100% vs
1338 = 50%) that A1 relies on, so it must be a *separate later pass*. In spirit it belongs to the
Phase-7 mechanistic "why" work more than to the readout diagnosis. Keep it in the plan (the
advisor raised it) but run it **last**, and only if useful.

**Purpose.** A small model (few heads, small dim) is simple enough to read attention + bias **by
eye**, which supports the eventual "why" analysis.

**⚠️ Keep `n_layer ≥ 2`.** The *generalizing* rule ("is X immediately before Y?") is a **two-hop**
relational computation and cannot be represented in one layer:
- **Layer 1:** a distance-1-biased head lets **Y detect "X is right before me"** — a *local*,
  relative check that works at any position → Y now carries an "adjacent-to-X" flag.
- **Layer 2:** the **readout reads that flag off Y**.

A **1-layer** model can only implement the **absolute shortcut** — its readout is pinned to the
last slot, so the one thing it can do is a distance-`d`-peaked head reading a *fixed* slot
(e.g. distance 3 from slot 5 = slot 2). So a 1-layer run would fail for a **depth** reason,
confounding the readout diagnosis. Do not use 1 layer as the minimal model.

**Steps.**
1. Set `n_layer=2, n_head=1` (down from 3 / 2), keep dim; re-run length-6 b=1, 10 seeds.
2. **Confirm val still = 100%.** If a single head can't learn it, bump to `n_head=2`.
3. Re-plot A1 on this model (a single head → one legible bias curve).

**Expected.** Same failure shape (learns, fails held-out) but now human-readable.

**Optional separate probe — 1 layer.** Run `n_layer=1` *on its own* to demonstrate the depth
argument: it should **only ever do the shortcut** (always fail held-out), which is itself a clean
proof that "the shortcut is a 1-layer, fixed-slot reader." Keep it labeled as a depth probe, not
the minimal-model diagnostic.

### A3. Switch the task to X..X — the cleanest "can't distinguish" probe

**Purpose.** Remove the token-identity cue so that distinguishing `X0X000` from `000X0X` can
**only** be absolute-position leakage (there is no other cue to grab).

**Steps.**
1. Add an X..X variant of the data generator: two identical `X` markers; label = `T` iff the two
   X's are distance 1 apart. Same length / split / b as the X..Y task.
2. **First confirm it is learnable (val = 100%)**, exactly as for X..Y, before reading any held-out
   number. ⚠️ X..X is *not* simply "X..Y minus the identity cue": matching two **identical** markers
   and measuring their gap is a somewhat different (possibly harder) computation, so a low held-out
   could mean "X..X didn't learn cleanly" rather than position leakage. Rule that out first.
3. Run length-6 b=1, 10 seeds; compare the held-out pattern against X..Y.

**Expected.** If learnable, the same qualitative failure (position shortcut) with no identity
confound — a razor-sharp version of the professor's main objective ("the model should be unable to
distinguish `X0X000` from `000X0X`").

---

## Part B — Prepare the proposal (bring to the meeting; do NOT switch unilaterally)

The fix is a **readout** change, which touches the "micro-**LLM** / next-token" framing, so it is
the professor's call. Bring two options with the evidence above:

- **Option 1 — mean-pool readout.** Average the final hidden states over all positions before the
  classifier. Fully relative; departs from decoder-LM style. *(This is the version that
  historically reached 100% at b=1.)*
- **Option 2 — bidirectional attention only (`causal=False`).** Keeps the LM-style readout; removes
  the causal mask's absolute signal but **not** the fixed-anchor signal → at best a **partial** fix.
  ⚠️ **T5 caveat:** an earlier ablation found `causal=False` can *break T5 learnability* (the
  bidirectional buckets), so for T5 specifically this may not even learn. Try `t5_bias_mode='causal'`
  with bidirectional attention to keep decoder-style buckets. (The RoPE 55→80% precedent under
  `causal=False` on dist≥5 was RoPE, not T5.)

**The decisive experiment (advisor-gated) — mean-pool readout.** This is the *actual* test of the
hypothesis, not an "if time" extra. Swap the fixed last-position readout for a position-neutral one
(mean-pool the final hidden states) and re-run length-6 b=1, 10 seeds. Prediction: collapse
8/10 → 0/10, held-out F acc rises off 0. This also **reproduces/verifies** the historical
"mean-pool = 100%" claim (currently only a log caveat). Run it first once the advisor approves the
architecture change.
⚠️ **b-sweep risk:** mean-pool averages *all* positions, so at high `b` the busy background may
dilute the X/Y signal — it could help at b=1 but hurt at b=10, and the b-sweep is this folder's real
goal. Keep max-pool / attention-pool (a learned query) as alternatives, and re-validate the chosen
readout **across the b-sweep**, not just at b=1.

## Why this stays on-purpose (talking points for the meeting)

- **Fixed length + position-based split: untouched.** Still position generalization, not length
  generalization.
- **PE stays the object of study.** Fixing the readout removes a confound that was dragging down
  the *relative* PEs, so the PE comparison gets **cleaner, not erased**: absolute PE is expected
  to still fail even with mean-pool, because absolute position is baked into its hidden states
  upstream (averaging confused notes gives a confused average).
- **The readout finding is itself in-scope.** "Relative PE is necessary but not sufficient; the
  readout must also be position-neutral" is a direct answer to the central *why does it
  generalize* question.
- **It completes the professor's stated aim**: "only relative positioning, remove absolute
  positioning."

## Anticipated objections (and answers)

1. **"Isn't this just the length-6 degeneracy you already documented (only one F config in the first
   half)?"** Partly — the single F config is what *makes the shortcut available*, so data sparsity
   and the readout are two levers on the *same* failure. But they are separable: growing the
   training region (longer inputs) reduces complete failures (9→4→6→0 across lengths 6/8/10/12) yet
   **never** gets all seeds to 100% (best 4–6/10); the mean-pool readout reportedly reached 100% at
   the *hardest* case (length 6). So the readout is the stronger, more complete lever, and A1's
   heatmap checks whether the failure tracks the gap or the position. Own this argument, don't dodge
   it.
2. **"Is the bias plot decisive?"** No — see A1. It is a first cut; the decisive test is the
   mean-pool swap (advisor-gated). Pre-meeting = strong plausible case, not proof.
3. **"Does the fix survive the background sweep?"** Unverified — mean-pool may dilute the signal at
   high `b`. Flagged as a risk in Part B; validate across `b` and keep pooling alternatives open.

## Open questions for the professor

1. **Readout:** OK to move to mean-pool (Option 1), or stay LM-style and accept the partial
   bidirectional fix (Option 2)?
2. **Framing:** does "micro-LLM" require next-token readout, or is a pooled classifier acceptable
   for these binary classification tasks?
3. **Task:** should X..X *replace* X..Y going forward, or serve only as a one-off diagnostic?

## Closed / ruled out

- **RoPE "bug" in `CausalSelfAttention`:** no-op under t5; cannot affect these runs. Closed.
- **Colon marker as the cause:** ruled out (colon vs. no-colon barely differs).

---

### Suggested order

1. **A1** (T5 bias + position heatmap, seed 1337 vs 1338, on the current 3-layer / 2-head model) —
   narrow the cause (PE vs. downstream). Highest value, lowest cost; reuses the documented runs.
2. **A3** (X..X) — reconfirm the "can't distinguish" objective with no identity confound.
3. Carry the evidence + **Part B proposal** into the meeting.
4. **A2** (smaller model, `n_layer=2, n_head=1`) — *last / optional.* Only if A1's figures need it;
   otherwise defer to the Phase-7 mechanistic work.
