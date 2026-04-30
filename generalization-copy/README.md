
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
Training data uses only lowercase a-z, but the vocab includes both lowercase and uppercase letters (vocab size: 54) so the model has the uppercase tokens available for Phase 2 and 3.

---

## Directory Structure
```
generalization-copy/
├── README.md
├── evaluate.py
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

## Evaluation
Automated accuracy measurement on random test inputs:
```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python -u evaluate.py config/basic.py
```

To test on uppercase inputs:
```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python -u evaluate.py config/basic.py --test_alphabet=uppercase
```

---

## Experiment Log

### Run 1: max_iters=2000, block_size=32
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

### Run 2: max_iters=5000, block_size=32
**Purpose:** See if more iterations help
**Config:** max_iters=5000, block_size=32, others same as Run 1
**Results:**
- Train loss kept dropping to 1.34, but val loss rose to 2.30
- Clear overfitting, sampling actually got worse than Run 1

---

### Run 3: max_iters=2000, block_size=64
**Purpose:** Test if larger context window fixes the truncation problem
**Config:** max_iters=2000, block_size=64, others same as Run 1
**Results:**
- Sampling looked clean, strings copied correctly
- Wrote evaluate.py to measure accuracy systematically
- **Lowercase copy accuracy: 98% (98/100 correct)**

---

### Run 4: vocab size update with uppercase tokens
**Purpose:** Add uppercase letters to vocab so the model has those tokens available for Phase 2 and 3, without changing training data
**Changes:**
- Updated prepare.py so vocab includes a-z, A-Z, colon, and newline (vocab size: 54)
- Training data still uses only lowercase
- Retrained the model with the new vocab

**Results:**
- **Lowercase copy accuracy: 100% (100/100 correct)**
- **Uppercase copy accuracy: 0% (0/100 correct)**
- This is exactly the baseline expected before any fine-tuning
- Looking at errors, the model outputs lowercase-like patterns (e.g., "ywz", "GGGGGG") when given uppercase inputs, suggesting it falls back on what it knows

---

## Next Steps
- Write a new prepare.py to generate a small uppercase fine-tuning dataset
- Set up fine-tuning config with init_from pointing to the lowercase checkpoint
- Run both the fine-tuning experiment and a from-scratch control experiment
- Compare learning speed to measure knowledge transfer
