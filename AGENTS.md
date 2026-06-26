# AGENTS.md — Project Context

This file gives an AI coding agent the context needed to work on this project.
It is a research codebase for a CS honors thesis. Read it fully before editing code.

## Project

**Working title:** Generalization of Algorithmic Tasks in Micro-LLMs
**Author:** John Lee (rising senior, CS major / Math minor, Dickinson College)
**Advisor:** Prof. John MacCormick
**Timeline:** Summer internship June 4 – July 17, 2026; thesis completion Spring 2027.
**Constraint:** all models are micro-transformers, under ~1M parameters.

### Central question

When and why does a micro-transformer (<1M params) generalize to input
*configurations* it never saw during training, on fixed-length inputs — and how
do positional encoding, model depth, and scratchpad tokens affect this?

Here "configuration" means **position**. Input length is held fixed (e.g. 20).
Generalization is created by splitting the data on *where* the special symbol
appears: train on some positions, test on held-out positions.

### Critical framing (do not violate)

This is **NOT length generalization.** All training and test examples have the
**same length.** The distribution shift is over symbol *position* within a fixed
length, not over sequence length. Any design choice that drifts toward varying
input length is off-target. (The previous copy-task pipeline used variable-length
inputs and hit truncation problems — that is exactly the trap to avoid here.)

## Tasks (from the project spec)

1. **Symbol detection (start here).** Fixed-length string of digits; output `Y`
   if the letter `X` is present, `N` if absent.
2. **Relative order.** String contains `X` and `Y` exactly once each; output `Y`
   if `X` comes before `Y`, else `N`.
3. **Variants (later).** Counting occurrences of `X`; distance between two `X`s;
   distance-exceeds-threshold (binary); span-replace and move-to-end transforms.

The generalization experiment for each task: train with the special symbol(s) in
all positions except a held-out set, then test on the held-out positions. Three
split styles of interest: single held-out position (e.g. position 12), even vs.
odd positions, first half vs. second half.

## Repository layout

The engine and the experiment live in **two sibling repos**:

```
comp560-nanoGPT/          # the nanoGPT engine (train.py, model.py, sample.py, configurator.py)
comp560-JohnLee/
└── generalization-copy/  # last semester's experiment (copy task) — reference only
    ├── README.md
    ├── evaluate.py
    ├── config/basic.py
    └── data/basic/prepare.py
```

New work for the detection task should live in its own experiment directory
(e.g. `generalization-detect/`) mirroring this structure, so the copy-task work
is preserved untouched.

### What the existing (copy-task) pipeline does

- `data/basic/prepare.py`: generates `word:word\n` pairs (lowercase, length 3–6),
  concatenates into one text stream, splits 90/10 by character position into
  `train.bin` / `val.bin`. Vocab size 54 (a–z, A–Z, `:`, `\n`).
- `config/basic.py`: n_layer=4, n_head=4, n_embd=128, block_size=64, CPU,
  max_iters=2000, lr=1e-3, warmup_iters=100.
- `evaluate.py`: prompts `word:`, greedily generates `len(word)` tokens
  (temp 0.1, top_k=1), checks exact-match accuracy. Supports `--test_alphabet`.
- Standard nanoGPT configurator override pattern via `NANOGPT_CONFIG`.

This pipeline trains as a plain autoregressive LM over the concatenated stream.
Engine + workflow are reusable as-is. The two files that must change are
`prepare.py` and `evaluate.py`.

## Phase roadmap

- **Phase 0 — Reorient infrastructure. ✅ done.** Detection-task `prepare.py`
  (fixed length 20, position-controlled, 3-way split) + single-token Y/N
  `evaluate.py`. Result: 100% baseline on the full position distribution.
- **Phase 1 — Confirm the phenomenon. ✅ done.** Held-out-position split, no
  fine-tuning. Result: detection **generalizes 100% even with a whole half held
  out**, because detection is **position-invariant** ("is X present?" doesn't depend
  on where). Detection therefore cannot exhibit the phenomenon — a position-dependent
  task is needed. This is why the task ladder was moved ahead of the PE sweep below.
- **Phase 2 — Task difficulty ladder (moved up; current).** Add Task 2 (relative
  order: is `X` before `Y`?) and variants whose answer depends on position. Goal: find
  a task where held-out-position generalization can actually fail. Build with aligned
  examples (fix the flat-stream misalignment from Phase 0).
- **Phase 3 — Positional encoding sweep (primary summer deliverable).** Compare
  learned-absolute / sinusoidal / RoPE / NoPE on a task that *does* exhibit the
  phenomenon (from Phase 2) and a held-out split. Natural point to reconsider the
  engine (see Key decisions).
- **Phase 4 — Minimal model.** Shrink layers/heads to the smallest model that
  still solves the base task at ~99%, then rerun the generalization tests.
- **Phase 5 — Scratchpad tokens.** Add non-output tokens for intermediate
  computation; measure effect on generalization and the cost tradeoff vs. adding
  layers/heads.
- **Phase 6 — Fine-tuning axis (optional).** Compare fine-tuning effort for the
  generalized task vs. training it from scratch. (Absorbs last semester's
  copy+finetune work.)
- **Phase 7 — Explain "why" + write-up.** Mechanistic interpretability
  (Nanda-style progress measures) to explain why certain PEs/tasks generalize.
  Thesis writing.

## Phase 0 spec (✅ done — kept for reference)

### New `prepare.py` requirements

- Fixed input length **20**.
- Vocab: digits `0`–`9`, `X`, `Y`, `N`, `:`, `\n` (~15 tokens). Drop the letter
  alphabet from the copy task.
- Sample format (detection): `48295017364X19285746:Y` (one `X` present) or
  `48295017364019285746:N` (no `X`). Output is a **single** token after `:`.
- **Position control:** the position(s) where `X` is placed must be a parameter,
  so a single function covers single-held-out, even/odd, and half/half splits.
- **3-way split — generate three separate pools (not a 90/10 stream slice):**
  1. `train` — `X` in allowed positions.
  2. `in-distribution val` — allowed positions, held-out examples; for monitoring
     training (the nanoGPT `val.bin` role).
  3. `generalization test` — `X` in the excluded position(s); evaluated separately
     via `evaluate.py`, never seen during training.
- Keep Y/N classes balanced and report class balance.

### `evaluate.py` changes

- Output is one token. Drop the `len(word)`-token generation; generate 1 token
  after `:` and check Y vs. N.
- Binary task → chance is 50%. Balance classes and report per-class accuracy, not
  just overall, so the baseline is meaningful.
- Add an option to point evaluation at the held-out generalization-test set.

## Key decisions

- **Engine: keep nanoGPT through the task ladder; the big reassessment is at the PE
  sweep (Phase 3).** Get the phenomenon confirmed on a verified engine first, before
  investing in a rewrite. (Phase 2's aligned-examples fix is a smaller data-layout /
  `get_batch` change, not a rewrite.)
- **From-scratch transformer is on the table for Phase 3+.** The advisor suggested
  (not mandated) eventually moving off nanoGPT to a custom from-scratch program
  for better code ownership. Phase 3 (where PE must be swapped, requiring deep
  edits to `model.py`) and Phase 7 (interpretability hooks) are where this pays
  off. Decide fork-vs-rewrite at the Phase 3 boundary.
- nanoGPT default PE is learned-absolute; the PE sweep will require editing the
  engine's `model.py` (not yet reviewed in detail).

## Working principles

- **Seeds matter.** Small-model generalization is seed-sensitive. Run 3–5 seeds
  per config and report mean/variance from Phase 0 onward; build config-sweep +
  seed-repeat + logging structure in early.
- **Read literature just-in-time, superficially.** The advisor explicitly said to
  read the list superficially and not worry about the math. Read each paper when a
  specific design decision needs it, not all upfront.
- **Don't record unread papers.** John does not list papers in logs or documents
  until he has personally reviewed them. There is an open gap in the literature on
  the fixed-length case (the advisor hasn't found canonical references either);
  candidates exist but must be reviewed before being cited or relied on.

## Confirmed reading list (from the honors intent form)

1. Vaswani et al. 2017, *Attention Is All You Need* (NeurIPS).
2. Weiss et al. 2021, *Thinking Like Transformers* (ICML).
3. Zhou et al. 2024, *What Algorithms Can Transformers Learn? A Study in Length
   Generalization* (ICLR).
4. Kazemnejad et al. 2023, *The Impact of Positional Encoding on Length
   Generalization in Transformers* (NeurIPS).
5. Nye et al. 2021, *Show Your Work: Scratchpads for Intermediate Computation
   with Language Models* (arXiv:2112.00114).

Note: most of this list targets length generalization; relevance to the
fixed-length setting is partial and should be read through that lens.