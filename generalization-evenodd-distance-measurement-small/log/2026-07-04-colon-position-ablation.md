# Colon Anchor Ablations - RoPE vs T5

*2026-07-04 - John Lee*

## Question

Prof. MacCormick noted that the main results matched the hypothesis, except for the
weird T5 behavior when a final `:` was added. These follow-ups check whether the `:`
token is acting like a fixed positional anchor for T5.

## Setup

Same small even/odd distance measurement task:

```text
T iff abs(index(X) - index(Y)) is even
F iff abs(index(X) - index(Y)) is odd
```

The body length is always 6. The train/val examples keep `X` and `Y` in positions
`0,1,2`; held-out examples keep them in positions `3,4,5`.

First, I changed the marker position:

| condition | model input example |
|---|---|
| `no_colon` | `XOYOOO` |
| `with_colon` | `XOYOOO:` |
| `front_colon` | `:XOYOOO` |

Then I added one masking condition:

| condition | model input example | attention behavior |
|---|---|---|
| `with_colon_masked` | `XOYOOO:` | body tokens cannot attend to the final `:` key |

These follow-ups only rerun `rope` and `t5`, because those are the two relative-position
methods relevant to the weird behavior.

## Config

| item | value |
|---|---|
| directory | `generalization-evenodd-distance-measurement-small/` |
| configs | `config/front_colon.py`, `config/with_colon_masked.py` |
| compared PEs | `rope`, `t5` |
| model | 3 layers, 2 heads, 32 dim, about 0.04M params |
| mask | `causal=False` |
| seeds | `1337`, `2024`, `31415`, `27182` |

## Hypothesis

If the final `:` hurts T5 because it acts as a fixed positional anchor, then moving the
marker should change T5's held-out behavior, and masking attention to the marker should
remove the failure. RoPE should stay stable because it already generalized in both
no-colon and with-colon runs.

## Commands

```bash
cd generalization-evenodd-distance-measurement-small

for seed in 1337 2024 31415 27182; do
  for pe in rope t5; do
    ../venv/bin/python train.py config/front_colon.py --seed=$seed --pos_type=$pe
    ../venv/bin/python evaluate.py config/front_colon.py \
      --results_csv=results_front_colon_lateststyle.csv \
      --predictions_csv=predictions_front_colon_lateststyle.csv \
      --show_errors=0
  done
done

for seed in 1337 2024 31415 27182; do
  for pe in rope t5; do
    ../venv/bin/python train.py config/with_colon_masked.py --seed=$seed --pos_type=$pe
    ../venv/bin/python evaluate.py config/with_colon_masked.py \
      --results_csv=results_with_colon_masked_lateststyle.csv \
      --predictions_csv=predictions_with_colon_masked_lateststyle.csv \
      --show_errors=0
  done
done

../venv/bin/python plot_latest_commit_style.py --split=half \
  --results_csv=results_front_colon_lateststyle.csv \
  --predictions_csv=predictions_front_colon_lateststyle.csv \
  --out_dir=log/figures/front_colon

../venv/bin/python plot_latest_commit_style.py --split=half \
  --results_csv=results_with_colon_masked_lateststyle.csv \
  --predictions_csv=predictions_with_colon_masked_lateststyle.csv \
  --out_dir=log/figures/with_colon_masked
```

The masking code is in `model.py`, lines 76-79. For `mask_marker_key='last'`, body-token
queries cannot attend to the final `:` key, but the input still contains the `:` token.

## Results

Held-out accuracy, mean +/- std over 4 seeds:

| condition | RoPE | T5 |
|---|---:|---:|
| `no_colon` | 100.0 +/- 0.0 | 100.0 +/- 0.0 |
| `with_colon` | 100.0 +/- 0.0 | 56.2 +/- 10.8 |
| `front_colon` | 100.0 +/- 0.0 | 68.8 +/- 20.7 |
| `with_colon_masked` | 100.0 +/- 0.0 | 100.0 +/- 0.0 |

For `front_colon`, T5 held-out accuracy by seed was:

```text
50, 100, 50, 75
```

For `with_colon_masked`, T5 held-out accuracy by seed was:

```text
100, 100, 100, 100
```

**Figure 1 - Front-colon held-out vs validation accuracy.**
RoPE stays perfect, while T5 becomes seed-sensitive when the marker
is moved to the front.

<img src="figures/front_colon/heldout_accuracy_half.png" alt="front-colon held-out accuracy by PE" width="680">

**Figure 2 - Front-colon held-out accuracy by class.**
This shows which label T5 collapses on in the seeds where it fails.

<img src="figures/front_colon/heldout_perclass_half.png" alt="front-colon per-class held-out accuracy by PE" width="680">

**Figure 3 - Front-colon per-position heatmap.**
The failure pattern changes when the fixed marker moves from the end
to the front.

<img src="figures/front_colon/per_position_half.png" alt="front-colon per-position accuracy heatmaps" width="900">

**Figure 4 - Masked-colon held-out vs validation accuracy.**
T5 returns to 100% when the final `:` is still present but cannot be
attended to by body tokens.

<img src="figures/with_colon_masked/heldout_accuracy_half.png" alt="with-colon-masked held-out accuracy by PE" width="680">

**Figure 5 - Masked-colon held-out accuracy by class.**
Both T/even and F/odd recover to 100% for T5.

<img src="figures/with_colon_masked/heldout_perclass_half.png" alt="with-colon-masked per-class held-out accuracy by PE" width="680">

**Figure 6 - Masked-colon per-position heatmap.**
Masking attention to the marker removes the held-out failure pattern.

<img src="figures/with_colon_masked/per_position_half.png" alt="with-colon-masked per-position accuracy heatmaps" width="900">

## Takeaway

The result strongly supports the anchor hypothesis.

RoPE is stable no matter where the `:` marker is placed. T5 is not: it is perfect with no
marker, weak with the marker at the end, and seed-sensitive with the marker at the front.

Most importantly, T5 returns to 100% when the final `:` is still present but body tokens
cannot attend to it. So the issue is not simply sequence length 7. The issue appears to be
T5's ability to use the visible `:` marker as an extra position cue.

## Caveat

This does not prove the exact internal mechanism, but it is stronger evidence than the
position ablation alone: masking attention to the marker removes the failure.
