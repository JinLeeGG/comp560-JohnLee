# Tiny Relative-Order Sanity Experiment

This directory is the 6/30 advisor sanity check, separated from
`generalization-order/` so the June 20 length-20 results stay untouched.

## Question

Before running larger PE sweeps, verify the expected behavior on a tiny fixed-length
task:

```text
input length = 6
vocab        = O, X, Y, T, F, :, newline
label        = T iff index(X) < index(Y), else F
train/val    = X and Y only in positions 0,1,2
test         = X and Y only in positions 3,4,5
mask         = causal=False first
readout      = mean over the six body-token hidden states
```

This is not length generalization. Every body has length 6; the shift is only over
where `X` and `Y` appear.

## Expected Pattern

The meeting hypothesis:

| PE | train/val | held-out |
|----|-----------|----------|
| `none` | fail | fail |
| `learned` | succeed | fail |
| `sinusoidal` | succeed | fail |
| `rope` | succeed | succeed |
| `t5` | succeed | succeed |

Here `t5` is a simplified T5-style relative bias: one learned bucket for every
relative offset, not logarithmic buckets.

The current version feeds only the six body tokens to the transformer. The `:` and
label remain in the data files for readability, but `:` is not part of the model input.
Prediction uses a real 2-class classifier head (`F/T`) over the mean-pooled hidden states.

## Where To Inspect T5

The T5-style code is in [pos_encoding.py](pos_encoding.py):

- `T5RelativeBias`: lines 132-158
- dispatch from `pos_type='t5'`: lines 161-172

The attention layer uses it in [model.py](model.py):

- attention scores are computed at lines 69-70
- `bias = self.pe.attention_bias(...)` is called at line 71
- the bias is added to attention logits at lines 72-73

The implementation is intentionally simple:

```python
relative_position = key_index - query_index
bucket = relative_position + (block_size - 1)
self.rel_bias = nn.Embedding(2 * block_size - 1, n_head)
```

So every exact relative offset gets one learned scalar per head. This removes standard
T5's logarithmic distance buckets and uses the advisor-requested "one bucket per offset"
version. It does not add absolute position embeddings and does not change Q/K/V, values,
MLP, or the classifier head.

Parameter count:

- `body_only`: `block_size=6` -> `11` buckets x `2` heads = `22` scalar bias params
- `body_plus_colon`: `block_size=7` -> `13` buckets x `2` heads = `26` scalar bias params

This is not a full T5 architecture; it is only the T5-style relative attention bias.

## Current Smoke Results

Two input modes are logged separately:

- [`body_only`](log/2026-07-02-body-only-smoke.md): the transformer sees only the six
  body tokens.
- [`body_plus_colon`](log/2026-07-02-body-plus-colon-smoke.md): the transformer also
  sees the fixed final `:` token, but the classifier still pools only body tokens.

Body-only result, seed `1337`:

| PE | val | held-out |
|----|-----|----------|
| `none` | 50.0% | 50.0% |
| `learned` | 100.0% | 50.0% |
| `sinusoidal` | 100.0% | 33.3% |
| `rope` | 100.0% | 100.0% |
| `t5` | 100.0% | 100.0% |

Interpretation: NoPE cannot learn without positional information; absolute PEs learn
the training positions but fail on held-out positions; relative PEs learn and transfer.

Body-plus-colon result, seed `1337`:

| PE | val | held-out |
|----|-----|----------|
| `none` | 50.0% | 50.0% |
| `learned` | 100.0% | 66.7% |
| `sinusoidal` | 100.0% | 50.0% |
| `rope` | 100.0% | 100.0% |
| `t5` | 100.0% | 100.0% |

Interpretation: adding `:` changes the absolute-PE failure pattern, so body-only is
the cleaner sanity check for the core hypothesis.

## Run

From this directory:

```bash
../venv/bin/python data/order/prepare.py

for pe in none learned sinusoidal rope t5; do
  ../venv/bin/python train.py config/body_only.py --pos_type=$pe
  ../venv/bin/python evaluate.py config/body_only.py \
    --results_csv=results_body_only.csv \
    --predictions_csv=predictions_body_only.csv
done

../venv/bin/python plot_latest_commit_style.py --split=half \
  --results_csv=results_body_only_lateststyle.csv \
  --predictions_csv=predictions_body_only_lateststyle.csv \
  --out_dir=log/figures/body_only
```

For seed repeats:

```bash
for pe in none learned sinusoidal rope t5; do
  for s in 1337 1338 1339 1340; do
    ../venv/bin/python train.py config/basic.py --pos_type=$pe --seed=$s
    ../venv/bin/python evaluate.py config/basic.py
  done
done
```

`evaluate.py` appends:

- `results.csv`: aggregate held-out and in-distribution accuracy
- `predictions.csv`: per-position diagnostic sweep for heatmaps

Generated bins, metadata, test set, checkpoints, and `out*/` are ignored by the
repo-level `.gitignore`.
