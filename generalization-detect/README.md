# Generalization Experiment: Detection Task
**Date:** June 8, 2026

This experiment trains a micro-transformer (<1M params) on a **symbol-detection**
task and measures whether it generalizes to symbol *positions* it never saw during
training.

> **Critical framing — this is NOT length generalization.** Every input is exactly
> `LENGTH = 20` characters. The distribution shift is over *where* the special
> symbol `X` appears within that fixed length, never over sequence length.

---

## Task

A fixed-length string of digits either contains the symbol `X` (exactly once) or
does not. The model reads the string, sees `:`, and must output a **single** token:
`Y` if `X` is present, `N` if absent.

```
48295017364X19285746:Y     <- X present (at some allowed position)  -> Y
48295017364019285746:N     <- no X (all digits)                     -> N
```

Vocab (size 15): digits `0`–`9`, plus `X`, `Y`, `N`, `:`, `\n`.

---

## The position split (the experiment knob)

Generalization is created by controlling **which positions `X` may occupy**. One
parameter, `SPLIT` in `data/detect/prepare.py`, selects the split style:

| `SPLIT`     | train/val X positions        | test X positions            |
|-------------|------------------------------|-----------------------------|
| `none`      | all 0–19 (full distribution) | all 0–19 (in-distribution)  |
| `single`    | all except `HELDOUT_SINGLE`  | only `HELDOUT_SINGLE`       |
| `even_odd`  | even positions               | odd positions               |
| `half`      | first half (0–9)             | second half (10–19)         |

`prepare.py` writes **three pools**:
- `train.bin` — training stream (X in allowed positions)
- `val.bin` — in-distribution val, for monitoring during training
- `test.txt` — the generalization test set (X in the held-out positions),
  evaluated separately by `evaluate.py`, never seen during training

Classes are kept balanced 50/50 (Y/N), so chance accuracy is 50%.

---

## Directory Structure
```
generalization-detect/
├── README.md
├── evaluate.py
├── verify_eval.py          (correctness / sanity check)
├── config/
│   └── basic.py
├── data/
│   └── detect/
│       ├── prepare.py
│       ├── meta.pkl        (generated)
│       ├── train.bin       (generated)
│       ├── val.bin         (generated)
│       └── test.txt        (generated)
├── log/                    (per-phase experiment logs)
└── out/                    (checkpoint)
```

---

## Setup
From the repo root (`comp560-JohnLee`):
```bash
source venv/bin/activate          # torch + numpy already installed here
cd generalization-detect
```

## Prepare data
```bash
python data/detect/prepare.py
```

## Training
```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python -u ../../comp560-nanoGPT/train.py config/basic.py
```

## Evaluation
```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python -u evaluate.py config/basic.py
```

## Verify (sanity check)
`verify_eval.py` checks that the headline accuracy is real, independent of `test.txt`:
1. **Causal probes** — takes a fixed digit string and flips `X` in/out at chosen
   positions, confirming the model's answer flips `N`↔`Y` (including at *held-out*
   positions). This proves the model genuinely detects `X`, not that it memorized labels.
2. **Label integrity** — every `test.txt` line is length 20 and `(X present) iff (label Y)`.
3. **Position coverage** — reports where `X` actually lands in the test set.

Run it after any change to the data or model:
```bash
python verify_eval.py
```

---

## Experiment Logs

Detailed records live in [`log/`](log/) — one file per experiment, named
`YYYY-MM-DD-<task>.md` (dated so they sort chronologically as the roadmap shifts):

| Date | Experiment | Status |
|------|------------|--------|
| 2026-06-08 | [Detection baseline — X at all positions](log/2026-06-08-detection-baseline.md) | ✅ 100% |
| 2026-06-09 | [Held-out positions — detection is position-invariant](log/2026-06-09-detection-heldout-positions.md) | ✅ 100% (trivial) |
| 2026-06-16 | [From-scratch engine — swappable positional encoding](log/2026-06-16-from-scratch-pe-engine.md) | ✅ gate passed (learned) |
| _next_ | Relative order — position-dependent task | — |

This README stays a stable overview; each new experiment adds a `log/YYYY-MM-DD-<task>.md`.

---

## Next Steps
- **Phase 2 — relative order:** one `X` and one `Y`; output `Y` if `X` comes before `Y`.
  The answer depends on position, so held-out positions can genuinely break generalization
  (unlike detection, which is position-invariant). Build it with **aligned examples**.
- Add a seed sweep (3–5 seeds, mean/variance) once a task sits near a generalization boundary.
