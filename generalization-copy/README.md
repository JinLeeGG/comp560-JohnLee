
# Generalization Experiment: Copy Task
**Date:** April 29, 2026
**Experiment #1**

This experiment trains a character-level GPT model on a lowercase-only copy task, then tests whether it can transfer that knowledge to uppercase letters through fine-tuning.

---

## Data Format
The model learns from a stream of copy pairs:
```
abc:abc
hello:hello
zrpmj:zrpmj
```
Only lowercase a-z is used in Phase 1. Vocab size: 28 (a-z, colon, newline).

---

## Directory Structure
```
generalization-copy/
├── README.md
├── config/
│   └── basic.py
├── data/
│   └── basic/
│       ├── prepare.py
│       ├── meta.pkl
│       ├── train.bin
│       └── val.bin
└── out/
```

---

## Setup
**1. Create Python virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**2. Install dependencies:**
```bash
pip install torch numpy tqdm tiktoken
```

**3. Prepare data:**
```bash
cd generalization-copy
python data/basic/prepare.py
```

---

## Training
From the `generalization-copy` directory:
```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python -u ../../comp560-nanoGPT/train.py config/basic.py
```

---

## Sampling
After training:
```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python -u ../../comp560-nanoGPT/sample.py config/basic.py --num_samples=3 --max_new_tokens=20 --seed=2345
```

---

## Experiment Log

### Run 1: max_iters=2000
**Purpose:** Verify workflow and train baseline lowercase copy model
**Config:** max_iters=2000, n_layer=4, n_head=4, n_embd=128, block_size=32, device=cpu, compile=False
**Results:**
- Training completed successfully, final val loss: 1.7350
- Model partially learned the copy task
- Short strings copied correctly (e.g., ngq:ngq, ipbx:ipbx)
- Longer strings had occasional errors (e.g., xxcm:xxc, wryd:wry)
- Loss plateaued around 1.7, suggesting more iterations needed

**Issues:**
- First run hit `ZeroDivisionError` at iter 1900 due to missing `warmup_iters`
- Fixed by adding `warmup_iters = 100` to config

---

## Next Steps
- Increase `max_iters` to 5000 and retrain until loss drops below 0.1
- Once lowercase copy is reliable, test on uppercase-only inputs to measure baseline accuracy
- Fine-tune with small amount of uppercase data and compare learning speed to a model trained from scratch
