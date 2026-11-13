# Step 7 plan: running-count information in constant-token representations

**Date:** November 8, 2026

## Question

Can a linear readout recover the current running count from hidden vectors at the
dummy `#`, and does a readout fitted at seen positions transfer to unseen positions?
This diagnoses information available in a representation, not whether the model
causally uses it to produce its answer. The design below is fixed before extraction.

## Frozen models and measurement sites

Use the final 2,000-iteration NoPE Step 5 checkpoints for `dummy`, `dummy_given`, and
`no_scratchpad`, with all five model seeds 1337-1341. The same 43-token vocabulary,
40-character bodies, architecture, and raw-data split are retained. No transformer
weights are trained or changed. Save checkpoint and data checksums.

The dummy and dummy-given records are identical. Measure the residual stream after
the embedding (layer 0) and after each of four blocks, at the current raw character
and the following `#`. The baseline is measured at the corresponding raw character.
Each measurement site/layer has its own probe. Block 4 is the pre-set primary layer;
all layers are reported, with no selection on unseen scores. Final LayerNorm is not
an additional measurement site.

## Controlled probe pools

Generate independent synthetic 40-character bodies using existing filler letters.
Each query has a non-X current character, and its target is the number of Xs in its
prefix. Exactly 0/1/2/3 Xs appear before the query, with no X after it. Thus the running
and final counts agree, but causal masking prevents a query representation from
using future characters, the delimiter, or the answer. The answer token is never
included in model input.

For each query position and random filler template, create all four counts with the
same current letter and the same background template. Insert Xs in uniformly drawn
allowed earlier positions. Counts are exactly balanced at every query position, so
position-only and constant-label baselines are 25%. Probe pool seed is 20261108.

- Fit pool: query positions 3-9, 64 templates per position (1,792 examples).
- Seen validation: positions 3-9, 32 independent templates (896 examples).
- Unseen test: positions 13-39, 32 independent templates (3,456 examples).
- Every X in fit/seen-validation is in 0-9; every X in unseen-test is in 10-39.
- Positions 10-12 are omitted from the balanced unseen test: with a non-X query,
  there are too few prior unseen slots to accommodate count 3.
- All full bodies across pools must be unique. These pools are separate from the
  original model train/validation/test examples and are not substituted for the
  original task's final-answer evaluation.

This is deliberately a balanced state-maintenance diagnostic. Its prefix-state
distribution differs from natural samples in the original dataset. It does not test
the increment at X itself, and fixed raw length 40 remains unchanged.

## Linear probes and controls

Standardize each feature using fit-pool means and standard deviations only. Fit a
four-class ridge least-squares readout with intercept and fixed L2 coefficient 1.0
(unnormalized sum-of-squares objective; intercept unpenalized). There is no
hyperparameter or layer selection using unseen data. Probe training is separate
from transformer training. Save the readouts under an ignored output directory.

For every checkpoint/site/layer, also fit one readout with shuffled fit labels, using
a fixed RNG seed and reporting it as a diagnostic rather than a significance test.
Repeat extraction/probing on freshly initialized models of the same architecture
and seeds, to test how much count is already readable without transformer training.

Layer-0 `#` vectors are identical for every sample and must score 25% on the balanced
test; raw-token embeddings must also score 25% because the query letter is matched
across counts. This is a leakage/control check, not a learned result.

## Reporting and interpretation

Report per-model-seed accuracy, mean and sample SD on seen validation and unseen test,
macro recall over the four counts, per-count recall, and query-position curves for
the primary layer. Report training-label-shuffle and random-model controls alongside
trained-model readouts. Also report supplied-format final-answer accuracy and, for
the trained dummy, whether it would predict the query `#` on these new pools.

If count is readable from a trained `#` above controls, its hidden vector contains
linearly accessible task information despite the constant visible token. Compare raw
sites to assess whether this property is specific to the filler. If unseen decoding
fails, do not conclude the information is absent: the representation/readout may
encode a count-to-position ratio or another changing coordinate system.

No activation patching or attention ablation is included in this step. Decodability
does not establish necessity, use by the final answer, or a complete explanation of
the Step 5/6 accuracy improvements.
