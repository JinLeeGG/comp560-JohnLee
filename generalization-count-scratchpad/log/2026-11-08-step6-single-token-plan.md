# Step 6 plan: a single given token before or after the input

**Date:** November 8, 2026

## Question and fixed comparisons

Does a single supplied `#` change fixed-length position generalization, and does its
placement matter? Step 5 found gains from dummy tokens, but placement and the
additional token-prediction objective affected the result. This step isolates two
short, answer-only formats. The conditions and interpretations below are recorded
before starting training.

| Condition | Record | Model input length | Loss |
|---|---|---:|---|
| Existing Step 5 baseline | `body:count` | 41 | answer only |
| `prefix_one` | `#body:count` | 42 | answer only |
| `suffix_one` | `body#:count` | 42 | answer only |

In both new conditions the `#` is supplied during training and inference. The model
does not predict it, and no state-token loss is applied. Both formats place the
answer after the same delimiter index. The 40 raw body characters, labels, and split
membership are identical to Step 5. Raw length is fixed within every train/test
comparison; this is position generalization.

## Configuration

- Exact existing data folder: `data/count_scratchpad_L40_seen10_step5`.
- Data seed 1337; 50,000 train / 5,000 validation / 2,000 test examples.
- Counts 0, 1, 2, 3 are balanced within each split.
- `X` positions: 0-9 for train/validation, 10-39 for unseen-position test.
- Same 43-token vocabulary; no new embeddings or output classes.
- NoPE, causal attention, 4 layers, 4 heads, embedding 128, dropout 0,
  `block_size=96`, under 1M parameters.
- Same optimizer settings and 2,000 iterations as Step 5.
- Paired model seeds 1337-1341; checkpoints 500 / 1,000 / 1,500 / 2,000.
- New training: 2 conditions x 5 seeds = 10 runs.
- Baseline: reuse only `no_scratchpad` from `results_step5_nope.csv` and
  `predictions_step5_nope.csv`, after checking data/model provenance.

Pre-run audit confirmed all five baseline checkpoint model/data metadata match and
their saved test `(body, label)` sequences equal the current data in file order.
The training program and optimizer configuration are unchanged from Step 5. The
checkpoint format does not itself store optimizer settings or data hashes; the
settings are supported by the Step 5 run record and unchanged source files.

SHA256 of the reused raw data, checked before the new sweep:

```text
train.bin 4929370e776961faa949b3c8dbd8a1a5ae37d5a63dc8cf080b47d50c3f9860f2
val.bin   af8172898cfbb273f85fe4fa78c63d15c10c0e05267cde6dce7d2b0fa51ac78d
test.bin  d58c8a6874aba88a5069d49210422cd8c90a0cd20efadec90d2a6364ce8205de
meta.pkl  5ebc43f5de17443184ecb5414c4f8bf0035a0b1ae7f14cfb301dcce6c7896c3c
```

## Metrics and analysis

The primary metric is unseen-position accuracy on counts 1-3 at the fixed final
2,000-iteration checkpoint. Report every seed, mean, sample SD, and paired changes
against the existing baseline and between the two new conditions. Accuracy on count
0 is reported separately because it has no target-symbol position shift.

Also report seen-position validation accuracy, each count separately, invalid answers,
single-X accuracy by raw position, and the four checkpoint trajectories. Report
optimization failures explicitly if a final run is below 99% seen-position accuracy.
Do not select checkpoints using unseen-position scores. Following Step 5, describe a
direction as consistent when at least 4 of 5 paired seeds have that direction; this is
a descriptive criterion, not a statistical-significance test.

Check final predictions against the same raw examples/labels and verify that each
new condition/seed has 5,000 validation and 2,000 test predictions. The original data
and baseline results must not be overwritten.

## Interpretation fixed before training

- If `prefix_one` improves over baseline, a single supplied prefix can reproduce
  some improvement without a long scratchpad or an auxiliary prediction loss.
- If the prefix consistently beats the suffix, token placement matters even with
  model sequence length, answer position, and objective matched.
- If both improve, the result remains compatible with several explanations; it does
  not establish hidden computation in the added token.
- If neither improves, one supplied token is insufficient in this setting. It does
  not establish that many tokens are necessary: Step 5's long prefix also used an
  auxiliary prediction loss.

Even a successful prefix changes causal context and raw-token sequence positions
together. These comparisons cannot establish an attention sink or isolate a unique
internal mechanism. A fixed ordinary-letter prefix is a potential later control;
it is not included in this run.

## Command

```bash
bash run_step6.sh
```

Outputs use new Step 6 paths: `out_step6_nope_*`, `results_step6_nope.csv`, and
`predictions_step6_nope.csv`. Per-example predictions and model checkpoints remain
local under the repository's ignore rules.
