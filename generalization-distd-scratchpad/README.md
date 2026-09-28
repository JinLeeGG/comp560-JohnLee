# Experiment Plan: Scratchpad Tokens for Distance-Threshold Generalization

**Status:** Implemented and smoke-tested; one-seed gate passed; formal five-seed sweep pending

**Date:** September 27, 2026

## Research question

Do structured scratchpad tokens help a NoPE micro-transformer generalize a distance-threshold rule to symbol positions not seen during training?

The experiment isolates the scratchpad intervention:

- Explicit positional encoding is fixed to **NoPE** in every condition.
- The model, optimizer settings, and seeds are fixed across conditions.
- All three conditions use the exact same underlying raw examples, in the same
  train, validation, and held-out splits.
- The dummy and meaningful scratchpad conditions are matched for sequence length
  and inference steps; the no-scratchpad condition serves as the practical baseline.

NoPE means that no position embedding, rotation, or attention bias is added. The
causal attention mask still provides an ordering signal, so this should be described
as **no explicit positional encoding**, not as a model with no positional information.

> This is **not length generalization**. Every raw input contains exactly 20
> characters. Train and test differ only in the positions where the two `X` symbols
> appear. Scratchpad conditions have longer model sequences, but their raw inputs
> remain fixed at length 20.

## Verified baseline

The pilot uses the existing [`dist >= 5` distance-threshold task](../generalization-distd/README.md). Before
selecting it, the dataset was cleaned so that train, validation, and test have
identical class and distance distributions.

NoPE baseline results over five model seeds:

| Evaluation group | Mean +/- sample SD |
|---|---:|
| In-distribution validation: all examples | 100.00 +/- 0.00% |
| Held-out test: all examples | 50.98 +/- 2.19% |
| Held-out test: examples whose correct answer is `T` (far) | 1.96 +/- 4.38% |
| Held-out test: examples whose correct answer is `F` (near) | 100.00 +/- 0.00% |

The overall held-out accuracy of approximately 51% does not mean that the models
partially distinguished far from near examples. Four seeds scored 0% on the far
class, and one scored 9.8%, while all five seeds scored 100% on the near class.
Therefore, the models predicted `F` for almost every held-out example.

All five models reached 100% validation accuracy on positions seen during training,
but failed to transfer the distance rule to the held-out positions. This consistent
failure leaves clear room to test whether meaningful scratchpad states improve
generalization. However, it does not show that every possible NoPE model is
theoretically incapable of solving the task.

Full baseline record:
[../generalization-distd/log/2026-09-27-nope-balanced-baseline.md](../generalization-distd/log/2026-09-27-nope-balanced-baseline.md).

## Task

A fixed-length 20-character string contains exactly two `X` symbols. The other 18
characters are random digits.

```text
distance = |position(X2) - position(X1)|

T if distance >= 5   (far)
F if distance < 5    (near)
```

Example:

```text
482X50178X4019285746:T
```

The two `X` symbols are at positions 3 and 9, so their distance is 6 and the answer
is `T`.

### Position split

- Training and in-distribution validation: both `X` symbols are in positions 0--9.
- Held-out test: both `X` symbols are in positions 10--19.
- Every pool is exactly 50/50 `T/F`.
- Every pool has the same distance frequencies:
  - Distances 1--4: 12.5% each.
  - Distances 5--9: 10% each.

Thus, the intended distribution shift is symbol position, not input length, class
balance, or distance frequency.

## Meaningful scratchpad

After each externally supplied raw token, the model predicts one token representing
its current counting state.

| State | Meaning after processing the current raw token |
|---|---|
| `i` | The first `X` has not been seen |
| `a` | The first `X` was just seen; zero intervening tokens |
| `b` | One token has appeared after the first `X` |
| `c` | Two tokens have appeared after the first `X` |
| `d` | Three tokens have appeared after the first `X` |
| `e` | Four or more tokens have appeared after the first `X` |
| `t` | The second `X` was seen and the distance is at least 5 |
| `f` | The second `X` was seen and the distance is less than 5 |

The `e` state saturates because exact distances above the threshold are irrelevant.
After the second `X`, the terminal `t` or `f` state is retained through the delimiter.

Shortened example:

```text
Raw:         2X1234X7:T
Scratchpad:  2iXa1b2c3d4eXt7t:tT
```

The two `X` symbols are five positions apart. Four raw tokens occur between them, so
the state reaches `e`; the second `X` then transitions to `t`.

This is supervised finite-state counting, not a free-form natural-language chain of
thought.

## Experimental conditions

### A. No scratchpad

The model reads the complete raw input and predicts only the final answer.

```text
2X1234X7:T
```

### B. Dummy scratchpad

The constant token `z` is generated after every raw token and after the delimiter.

```text
2zXz1z2z3z4zXz7z:zT
```

This controls for the longer sequence and additional autoregressive calls without
providing a useful intermediate state.

### C. Meaningful scratchpad

The model generates the correct counting states.

```text
2iXa1b2c3d4eXt7t:tT
```

The primary controlled comparison is meaningful versus dummy scratchpad. No scratchpad
shows the total practical improvement, but it is not matched for sequence length or
inference computation.

All three conditions will use the same union vocabulary, including all state and dummy
tokens, so embedding and output-layer parameter counts remain identical. The formal
no-scratchpad baseline will therefore be retrained rather than copied directly from
the task-selection run above.

## Training

Training uses teacher forcing. One complete example is processed in a single forward
pass; training is not performed as a sequence of separate model calls.

Loss is applied only at tokens the model must generate during inference:

- State loss on `i/a/b/c/d/e/t/f` or dummy `z` targets.
- Answer loss on the final `T/F` target.
- No loss on raw digits, `X`, or `:` because those tokens are externally supplied.

For scratchpad conditions:

```text
state_loss  = mean cross-entropy over state targets
answer_loss = cross-entropy on the final T/F target
total_loss  = state_loss + answer_loss
```

Taking the mean over state targets before combining the losses prevents the many
state positions from numerically overwhelming the single answer target. The two loss
components and their accuracies will be logged separately.

All conditions use the same number of examples, batches, and 2,000 optimization
iterations. The final checkpoint at the fixed training budget is used for the pilot;
the held-out test set is never used for checkpoint selection.

## Chained inference

Scratchpad inference is iterative:

1. The evaluator appends the next raw token.
2. The model greedily generates one state token.
3. That predicted state remains in the context.
4. Steps 1--3 repeat through the delimiter.
5. The model generates the final `T/F` answer.

Example prefixes:

```text
"2"             -> i
"2iX"           -> a
"2iXa1"         -> b
"2iXa1b2"       -> c
...
```

This free-running evaluation allows early state errors to propagate, matching actual
use. A teacher-forced evaluation is also reported to distinguish local transition
errors from accumulated chain errors.

## Metrics

Report on both in-distribution validation and held-out test:

- Free-running final-answer accuracy.
- Per-class `T/F` accuracy.
- Teacher-forced final-answer accuracy.
- State-token accuracy.
- Decision-state accuracy at the second `X`, reported overall and separately for
  far (`t`) and near (`f`) examples.
- Full state-trace exact-match accuracy.
- Invalid state-token rate.
- `state_loss` and `answer_loss` on in-distribution validation.

The primary metric is **free-running final-answer accuracy on the held-out-position
test set**.

## Run plan

1. Verify data formatting, loss masks, and chained inference with one seed.
2. Require at least 99% in-distribution final-answer accuracy before interpreting
   held-out results.
3. Run all three conditions with paired model seeds
   `1337, 1338, 1339, 1340, 1341`.
4. Report every seed plus mean and sample standard deviation.
5. If the pilot is positive, repeat with multiple paired data seeds.

Formal pilot size:

```text
3 conditions x 5 model seeds = 15 training runs
```

## Preliminary implementation gate

The end-to-end pipeline was checked with model seed `1337` using 500 training
iterations per condition. This shortened run used the same architecture and data as
the formal plan, with `lr_decay_iters=500` to match the smaller optimization budget.

| Condition | ID answer | Held-out answer | Held-out `T` | Held-out `F` | Held-out state / trace |
|---|---:|---:|---:|---:|---:|
| No scratchpad | 100% | 50% | 0% | 100% | n/a |
| Dummy scratchpad | 100% | 50% | 0% | 100% | 100% / 100% |
| Meaningful scratchpad | 100% | 100% | 100% | 100% | 100% / 100% |

The meaningful condition also achieved 100% decision-state accuracy at the second
`X` on the held-out test, and all scratchpad predictions were valid state tokens.
The dummy result shows that sequence length and extra autoregressive calls alone did
not fix the baseline failure in this seed.

This is an implementation gate, not the experiment's conclusion. It establishes
that the state machine, masked losses, and free-running evaluator can produce the
intended contrast. The preregistered five paired seeds at 2,000 iterations are still
required before interpreting the effect as robust. See
[`log/2026-09-27-one-seed-gate.md`](log/2026-09-27-one-seed-gate.md) for the run record.

## Interpretation

| Result | Interpretation |
|---|---|
| Meaningful > dummy and no scratchpad | Intermediate-state supervision improves generalization |
| Meaningful approximately dummy > no scratchpad | Extra sequence length or computation may explain the gain |
| Teacher-forced high, free-running low | State errors compound during chained inference |
| All conditions near chance in-distribution | Implementation or optimization failed; do not interpret OOD results |
| All conditions high on held-out positions | The intervention is not distinguishable in this setup |

A promising pilot result requires the meaningful condition to outperform both controls
across paired seeds while all conditions solve the in-distribution task.

## Known limitation and follow-up ablation

After the second `X`, meaningful scratchpads repeat the answer-bearing `t/f` state.
Therefore a positive pilot demonstrates a benefit from meaningful process supervision,
but does not by itself separate counting-state supervision from repeated answer
supervision. If the pilot is positive, the next ablation should replace the repeated
post-decision `t/f` states with a neutral `g` state while retaining a single final
`T/F` target.

## Implementation layout

```text
generalization-distd-scratchpad/
├── README.md
├── engine.py              # imports the verified distance-task model
├── scratchpad.py          # finite-state algorithm and target masks
├── dataset.py             # raw-data loading and condition-specific tensorization
├── train.py
├── evaluate.py
├── test_scratchpad.py       # state transitions, loss masks, and chained contexts
├── run_multiseed.sh         # formal 3-condition x 5-seed sweep
├── config/
│   └── basic.py
├── data/
│   └── distd_scratchpad/
│       └── prepare.py
├── log/
└── out/
```

`engine.py` reuses the verified model and NoPE implementation from
`generalization-distd` without modifying it. The raw dataset is generated once and
then tensorized into each condition, ensuring that all comparisons use exactly the
same underlying examples. Only the sequence representation, masked objectives, and
chained evaluator are specific to this experiment.
