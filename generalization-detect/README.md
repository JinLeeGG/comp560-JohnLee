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
├── config/
│   └── basic.py
├── data/
│   └── detect/
│       ├── prepare.py
│       ├── meta.pkl        (generated)
│       ├── train.bin       (generated)
│       ├── val.bin         (generated)
│       └── test.txt        (generated)
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

---

## Experiment Logs

Detailed records live in [`log/`](log/) — one file per phase, titled by **what the
experiment does** (the phase number is kept as a tag for roadmap tracking):

| Phase | Experiment | Status |
|-------|------------|--------|
| 0 | [Detection baseline — X at all positions](log/phase0-baseline.md) | ✅ 100% |
| 1 | Held-out position 12 _(next)_ | — |

This README stays a stable overview; each new phase adds a `log/phaseN-<name>.md`.

---

## Next Steps
- **Phase 1:** set `SPLIT='single'` (hold out position 12), re-run prepare → train →
  evaluate, and measure whether held-out-position accuracy drops. First real result.
- Run 3–5 seeds per config and report mean/variance.
