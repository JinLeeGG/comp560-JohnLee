# comp560-JohnLee

Research code by Gyujin (John) Lee for the Deep Network Understanding (DNU) Lab at
Dickinson College, advised by Prof. John MacCormick. The project asks when and why a
micro-transformer (under 1M parameters) generalizes to symbol positions it never saw in
training, and how the positional encoding and scratchpad tokens change that.

> **This is position generalization, not length generalization.** Within a task every
> input has the same length; training and test differ only in *where* the special symbols
> appear.

## Start here

[`generalization-order/`](generalization-order/) is the code for the summer 2026 report,
*Relative vs. Absolute Positional Encoding Predicts Fixed-Length Position Generalization
in Micro-Transformers* ([report](https://dnulab.org/student-reports/summer2026/lee.pdf),
[poster](https://dnulab.org/student-reports/summer2026/lee-poster.pdf)). The report's
LaTeX source is in [`report/`](report/).

## Experiments

Each folder has its own README with the task, results, and run commands.

### Scratchpad tokens (honors thesis, fall 2026)

| Folder | What it tests |
|---|---|
| [`generalization-distd-scratchpad/`](generalization-distd-scratchpad/) | Scratchpad tokens for the distance-threshold task (are two `X`s at least five apart?) |
| [`generalization-count-scratchpad/`](generalization-count-scratchpad/) | Counting `X`s at unseen positions: running-count scratchpad vs. length-matched dummy tokens |
| [`generalization-order-scratchpad/`](generalization-order-scratchpad/) | Scratchpad plan for the relative-order task (planning only) |

### Positional encoding (summer 2026)

| Folder | What it tests |
|---|---|
| [`generalization-detect/`](generalization-detect/) | Symbol detection (is `X` present?); where the from-scratch transformer with swappable positional encodings was built |
| [`generalization-order/`](generalization-order/) | Relative order (does `X` come before `Y`?) across five positional encodings and 10 seeds |
| [`generalization-evenodd/`](generalization-evenodd/) | Distance parity: is the distance between two `X`s even or odd? |
| [`generalization-distd/`](generalization-distd/) | Distance threshold: are two `X`s at least `D` apart? |
| [`generalization-evenodd-distance-measurement-small/`](generalization-evenodd-distance-measurement-small/) | Length-6 distance-parity check, including the colon-anchor ablation |
| [`generalization-background/`](generalization-background/) | Whether a busy digit background is what makes T5 fail, by sweeping the number of background token types |

### Coursework (COMP 560, spring 2026)

| Folder | What it tests |
|---|---|
| [`generalization-copy/`](generalization-copy/) | Copy task: train on lowercase words, then fine-tune to uppercase |
| [`translate-korean/`](translate-korean/) | Character-level model learning Korean number words |
| [`wandb_tutorial/`](wandb_tutorial/) | Weights & Biases tutorial |

## Setup

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

The experiment READMEs run their commands from inside each folder with
`../venv/bin/python`. `generalization-copy/`, `translate-korean/`, and the first detection
runs use the course's [nanoGPT engine](https://github.com/dson-comp560-sp26/comp560-nanoGPT)
instead, checked out next to this repo as `../comp560-nanoGPT`.
