# Small Even/Odd Distance Measurement Task

This directory is the 6/30 advisor sanity check, separated from the larger experiments
so the previous results stay untouched.

## Question

Before running larger PE sweeps, verify the expected behavior on a tiny fixed-length
task:

```text
input length = 6
vocab        = O, X, Y, T, F, :, newline
label        = T iff abs(index(X)-index(Y)) is even, else F
train/val    = X and Y only in positions 0,1,2
test         = X and Y only in positions 3,4,5
mask         = causal=False
readout      = mean over the body-token hidden states
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

## Where To Inspect T5

The T5-style code is in [pos_encoding.py](pos_encoding.py):

- `T5RelativeBias`: lines 132-158
- dispatch from `pos_type='t5'`: lines 161-172

The attention layer uses it in [model.py](model.py):

- attention scores are computed at line 72
- `bias = self.pe.attention_bias(...)` is called at line 73
- the bias is added to attention logits at lines 74-75

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

- `no_colon`: `block_size=6` -> `11` buckets x `2` heads = `22` scalar bias params
- `with_colon`: `block_size=7` -> `13` buckets x `2` heads = `26` scalar bias params

This is not a full T5 architecture; it is only the T5-style relative attention bias.

## Current Results

The main input modes are logged separately:

- [no_colon](log/2026-07-02-no-colon.md): the transformer sees only the six
  body tokens.
- [with_colon](log/2026-07-02-with-colon.md): the transformer also
  sees the fixed final `:` token, but the classifier still pools only body tokens.
- [colon-position ablation](log/2026-07-04-colon-position-ablation.md): checks whether
  the weird T5 result is caused by the `:` marker acting as a fixed positional anchor.

No-colon result, four seeds (`1337`, `2024`, `31415`, `27182`):

| PE | val | held-out |
|----|-----|----------|
| `none` | 50.0 +/- 0.0 | 50.0 +/- 0.0 |
| `learned` | 100.0 +/- 0.0 | 50.0 +/- 0.0 |
| `sinusoidal` | 100.0 +/- 0.0 | 50.0 +/- 0.0 |
| `rope` | 100.0 +/- 0.0 | 100.0 +/- 0.0 |
| `t5` | 100.0 +/- 0.0 | 100.0 +/- 0.0 |

With-colon result, four seeds (`1337`, `2024`, `31415`, `27182`):

| PE | val | held-out |
|----|-----|----------|
| `none` | 50.0 +/- 0.0 | 50.0 +/- 0.0 |
| `learned` | 100.0 +/- 0.0 | 50.0 +/- 0.0 |
| `sinusoidal` | 100.0 +/- 0.0 | 50.0 +/- 0.0 |
| `rope` | 100.0 +/- 0.0 | 100.0 +/- 0.0 |
| `t5` | 100.0 +/- 0.0 | 56.2 +/- 10.8 |

Interpretation: no-colon matches the meeting hypothesis across four seeds. Adding `:`
specifically hurts simplified T5 generalization while RoPE stays robust.

Colon-anchor follow-up, four seeds:

| condition | RoPE held-out | T5 held-out |
|----|-----:|-----:|
| `no_colon` | 100.0 +/- 0.0 | 100.0 +/- 0.0 |
| `with_colon` | 100.0 +/- 0.0 | 56.2 +/- 10.8 |
| `front_colon` | 100.0 +/- 0.0 | 68.8 +/- 20.7 |
| `with_colon_masked` | 100.0 +/- 0.0 | 100.0 +/- 0.0 |

The masked-colon condition keeps the final `:` in the input but prevents body-token
queries from attending to that final marker key. This restores T5 to 100%, supporting
the fixed-anchor explanation.

## Run

From this directory:

```bash
../venv/bin/python data/evenodd_distance/prepare.py

for pe in none learned sinusoidal rope t5; do
  ../venv/bin/python train.py config/no_colon.py --pos_type=$pe
  ../venv/bin/python evaluate.py config/no_colon.py \
    --results_csv=results_no_colon_lateststyle.csv \
    --predictions_csv=predictions_no_colon_lateststyle.csv
done

../venv/bin/python plot_latest_commit_style.py --split=half \
  --results_csv=results_no_colon_lateststyle.csv \
  --predictions_csv=predictions_no_colon_lateststyle.csv \
  --out_dir=log/figures/no_colon
```

For seed repeats:

```bash
for pe in none learned sinusoidal rope t5; do
  for s in 1337 2024 31415 27182; do
    ../venv/bin/python train.py config/no_colon.py --pos_type=$pe --seed=$s
    ../venv/bin/python evaluate.py config/no_colon.py \
      --results_csv=results_no_colon_lateststyle.csv \
      --predictions_csv=predictions_no_colon_lateststyle.csv
  done
done
```

`evaluate.py` appends:

- aggregate held-out and in-distribution accuracy
- per-position diagnostic sweep rows for heatmaps

Generated bins, metadata, test set, predictions CSVs, checkpoints, and `out*/` are ignored
by the repo-level `.gitignore` or the local `.gitignore`.
