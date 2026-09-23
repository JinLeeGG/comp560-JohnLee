# Experiment Plan: Scratchpad Tokens for Relative Order

**Status:** Planning only -- no implementation yet
**Meeting:** John Lee and Prof. John MacCormick, September 18, 2026

## Goal

Test whether explicit intermediate-state supervision helps a micro-transformer
generalize the relative-order rule to symbol positions that were not seen during
training.

This is a small proof-of-concept experiment. It will reuse the existing
from-scratch `MicroTransformer` and change only the data representation, training
targets, and evaluation procedure.

> This is **not length generalization**. Every raw input has exactly 20 characters.
> The distribution shift is over the positions of `X` and `Y` within that fixed
> length. Scratchpad tokens increase the model sequence length, but raw input length
> remains fixed across all examples.

## Task

Each input contains 18 random digits, one `X`, and one `Y`:

```text
53X4Y381902746105827:T
```

The label is:

- `T` if `X` appears before `Y`.
- `F` if `Y` appears before `X`.

The classes are balanced 50/50.

## Scratchpad concept

After each raw input token, the model predicts a token representing its current
state:

| Token | Meaning |
|---|---|
| `i` | Neither `X` nor `Y` has been seen |
| `x` | `X` was seen first; `Y` has not been seen |
| `y` | `Y` was seen first; `X` has not been seen |
| `t` | The model can conclude that `X` came first |
| `f` | The model can conclude that `Y` came first |

Example with a shortened raw input:

```text
Raw example:         53X4Y3:T
Scratchpad example:  5i3iXx4xYt3t:tT
```

The intended state transitions are:

```text
5 -> i
3 -> i
X -> x
4 -> x
Y -> t
3 -> t
: -> t
t -> T
```

This experiment uses supervised finite-state scratchpad tokens rather than
free-form natural-language chain-of-thought. During training, the correct state
token at each step is provided as a target; during testing, the model must generate
those state tokens itself.

## Pilot setup

The pilot will use the existing `half` position split because the current
relative-order experiment already shows a clear generalization failure for learned
absolute positional encoding:

- Train and in-distribution validation: both `X` and `Y` are in positions 0--9.
- Held-out test: both `X` and `Y` are in positions 10--19.
- Positional encoding: `learned` only for the initial pilot.
- Model architecture and optimizer: copy the current `generalization-order` setup.

All conditions must use the same underlying raw examples, split, model size,
optimizer settings, and seeds.

## Experimental conditions

### A. No scratchpad

The existing relative-order task. The model predicts only the final answer.

```text
53X4Y3:T
```

### B. Dummy scratchpad

A constant state token `d` is inserted after every raw token.

```text
5d3dXd4dYd3d:dT
```

This controls for the longer sequence and the additional autoregressive steps.

### C. Meaningful scratchpad

The inserted tokens represent the correct intermediate state.

```text
5i3iXx4xYt3t:tT
```

The meaningful condition must outperform the dummy condition before an improvement
can be attributed to state supervision.

All three conditions should use the same union vocabulary so that their embedding
and output-layer parameter counts remain identical.

## Training

Training uses teacher forcing. One complete interleaved example is processed in a
single forward pass; the model is not trained through multiple separate calls.

Loss is applied only where the model must produce an output:

- Scratchpad loss on `i/x/y/t/f` or dummy `d` targets.
- Final-answer loss on the final `T/F` target.
- No loss when the next raw input token is externally supplied.

The two losses should be reported separately:

```text
state_loss
answer_loss
total_loss = state_loss + answer_loss
```

Taking the mean state loss before combining it with answer loss prevents the many
state targets from overwhelming the single final-answer target.

The no-scratchpad condition has no `state_loss`; its loss is `answer_loss` only.

## Chained inference

Inference is iterative. The evaluator supplies one raw token, the model predicts one
state token, and that prediction is included in the context for the next step:

```text
"5"           -> predict i
"5i3"         -> predict i
"5i3iX"       -> predict x
"5i3iXx4"     -> predict x
"5i3iXx4xY"   -> predict t
...
```

After the delimiter state is generated, the model predicts the final `T/F` label.
Because predicted states are reused, an early state error may propagate through the
rest of the chain.

## Validation and evaluation

### During training: teacher-forced in-distribution validation

Measure on the existing in-distribution validation pool:

- State loss and state-token accuracy.
- Final-answer loss and accuracy.

Use only this in-distribution validation set for checkpoint selection. The held-out
test set must not influence model selection.

### After training: free-running chained evaluation

Run the full chained procedure on both:

- In-distribution validation examples.
- Held-out-position test examples.

Report:

- Final `T/F` accuracy overall and per class.
- State-token accuracy.
- Full state-trace exact-match accuracy.
- Invalid state-token rate.
- Teacher-forced versus free-running final-answer accuracy.

The primary result is free-running final-answer accuracy on the held-out test set.

## Run plan

1. Implement and debug all three conditions with one seed.
2. Confirm that every condition reaches approximately 99% in-distribution accuracy.
3. Run three seeds per condition: `3 conditions x 3 seeds = 9 runs`.
4. Report mean and standard deviation for held-out accuracy.
5. Inspect state-transition errors if chained accuracy is poor.

## Interpretation

| Result | Interpretation |
|---|---|
| Meaningful > dummy and no scratchpad | Intermediate-state supervision helps generalization |
| Meaningful approximately dummy > no scratchpad | Extra tokens or computation steps may explain the gain |
| All conditions approximately 100% held-out | The pilot is too easy; use a harder split or smaller model |
| Teacher-forced high, free-running low | State errors compound during chained inference |
| All conditions low in-distribution | Fix optimization or implementation before interpreting generalization |

A promising pilot result is a substantial and repeatable held-out improvement for
the meaningful scratchpad over both controls while all conditions solve the
in-distribution task.

## Proposed implementation layout

```text
generalization-order-scratchpad/
├── README.md
├── model.py                 # reuse the current task-agnostic MicroTransformer
├── pos_encoding.py
├── train.py                 # masked state/answer losses
├── evaluate.py              # chained inference and diagnostics
├── config/
│   └── basic.py
├── data/
│   └── order_scratchpad/
│       └── prepare.py
├── log/
└── out/
```

## Decisions to review before implementation

- [ ] Confirm `half` as the first position split.
- [ ] Confirm learned absolute PE as the only PE in the pilot.
- [ ] Confirm the three conditions: no scratchpad, dummy, and meaningful.
- [ ] Confirm that a state token is generated after `:` as well as after each body token.
- [ ] Confirm equal weighting of mean `state_loss` and `answer_loss`.
- [ ] Choose the in-distribution checkpoint-selection metric.
- [ ] Confirm three seeds after a successful one-seed smoke test.
