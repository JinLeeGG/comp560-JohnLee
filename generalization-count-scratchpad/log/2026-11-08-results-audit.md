# Counting experiments: numerical and interpretation audit

**Date:** November 8, 2026

## Scope and outcome

I checked the counting README, the eight Step 1-7 result logs (including both
Step 2 settings), the Step 6/7 pre-run plans, source code, saved result tables,
per-example diagnostics, checkpoint metadata, and figure calculations/captions.
Core result tables agree with their source data. Wording and one repeated rounding
convention were corrected; no experiment was retrained and no raw result was changed.

The strongest checks re-evaluated the saved models, rather than only recalculating
means from existing summary CSVs. Step 6's 105,000 final predictions were reproduced
exactly. Step 7's complete probe, position, and supplied-format diagnostic scores
were reproduced using independently written extraction/scoring code and the saved
linear readouts, without refitting them.

## Step 6: every final prediction checked

| Item | Verification |
|---|---|
| Original data | 50,000 / 5,000 / 2,000 train/validation/test bodies; raw length 40; balanced counts; allowed X positions; no duplicate or shared bodies across pools |
| Format and loss mask | All 21,000 condition/example inputs across three formats and the two evaluation pools match manual `body:`, `#body:`, or `body#:` encoding; no answer token in input; no state targets; one answer target at the delimiter |
| Checkpoints | All 60 condition/seed/iteration checkpoints match identity, model/data metadata; all 15 final aliases have the same weights as their iteration-2,000 files |
| Model size | 798,848 parameters, consistent with the reported 0.799M and the under-1M constraint |
| Raw evaluations | All 60 result rows, 105,000 final prediction rows, 12 summary rows, and 15 paired-comparison rows checked independently of `analyze_step6.py` |
| Fresh inference | All 15 final models on all 5,000 validation and 2,000 test examples reproduce their 105,000 saved answers exactly; answer cross-entropy differs from rounded CSV values by at most 5.777e-7 |
| Positions | Single-X curves agree with the predictions; suffix seed 1340 is perfect at every raw position; seed 1339 is perfect over unseen positions 10-33 |
| Errors | All final validation examples are correct. Test errors number 3,500 / 2,688 / 1,940 for baseline / prefix / suffix; every error is an undercount, with no invalid or overcounted answer |

Recomputed primary final results (unseen counts 1-3, mean +/- sample SD over five
model seeds):

| Baseline | One supplied prefix | One supplied suffix |
|---:|---:|---:|
| 53.33 +/- 10.15% | 64.16 +/- 10.60% | 74.13 +/- 22.61% |

The mean suffix-baseline change is +20.80 percentage points, positive in 4/5 paired
seeds. Every per-count value, per-seed difference, and 500/1,000/1,500/2,000 checkpoint
mean in the log agrees. The source training/configuration files also match the
Step 5 record used for the reused baseline.

## Step 7: every saved score checked against re-extracted vectors

| Item | Verification |
|---|---|
| Artifact identity | Actual SHA-256 values for four data files and 15 checkpoints match the manifest; numerical-library versions match |
| Probe pools | Regenerated all 6,144 bodies; pool hashes match; pools are mutually disjoint and exclude the original train/validation/test bodies; counts are balanced at every query; the current letter is matched; all Xs precede a non-X query |
| Independent encoding | Manual `body:` or interleaved `#a#b#...#:` input construction; no count target or answer token is supplied |
| Extraction and scoring | `verify_step7.py` uses separate hooks on normal model forwards, checks fit-only feature means/scales, and applies all 30 saved NPZ readout bundles to newly extracted vectors |
| Complete results | All 1,000 probe rows, 3,400 primary-layer position rows, and 90 supplied-format diagnostic rows match recomputation; all model weights remain unchanged |
| Summary statistics | All 200 summary groups and their 2,400 mean/sample-SD fields agree with independent aggregation |
| Numerical cross-check | Initial and verified CSV rows are identical; every saved parameter is finite; maximum initial-versus-verified readout parameter difference is 8.14e-13 |
| Leakage controls | Every layer-0 probe is exactly 25%; causal future-token tests pass; no unseen data is used to fit feature scaling or select a layer/readout hyperparameter |

All 24 format/loss-mask tests and all four probe tests also passed during the audit.

Primary dummy-predicted `#`, after block 4:

| Transformer / pool | Recomputed mean +/- sample SD |
|---|---:|
| Trained / seen | 88.15 +/- 2.77% |
| Random / seen | 79.38 +/- 2.22% |
| Trained / unseen | 47.84 +/- 5.71% |
| Random / unseen | 42.65 +/- 1.55% |

The trained-random unseen difference is +5.190972 percentage points on average,
positive in all five seeds; seed 1338 differs by only +0.028935 points. The trained
dummy exceeds trained dummy given by +8.391204 points on average, in 4/5 seeds.
These remain descriptive comparisons.

The unseen trained dummy's count-0/1/2/3 recalls are 100.00 / 50.90 / 32.18 / 8.29%.
Nonzero-class macro recall is 30.455247%, against 23.603395% for the random control,
using the unchanged four-class predictions. All supplied-format answer means/SDs
and the selected query-`#` prediction scores also match the log. They describe the
synthetic probe pools, not the original final-answer test distribution.

## Earlier results and presentation checked

Independent aggregation of earlier CSVs/per-example predictions confirmed:

- Step 1-5 final means, sample SDs, per-count and per-seed tables.
- Single-X first-error, below-50%, and zero-tail tables, position-bin summaries,
  and Step 5's last-X/count-3 breakdown.
- Among incorrect meaningful-scratchpad final answers on unseen inputs with counts
  1-3, the first trace mismatch occurs at an X in 38 NoPE / 3,512 RoPE cases, and
  at a following non-X in 561 / 336 cases; 72 / 63 cases have a correct trace but
  a wrong final answer. In the second category, the first mismatch occurs
  immediately after an X, matching the logs.
- Referenced distance-task means: NoPE 67.39 / 20.71 / 100%, RoPE
  49.45 / 52.09 / 100%, for no scratchpad / dummy / meaningful.
- Main and supporting Step 6/7 figures use the intended metric, condition, layer,
  seed pairing, and range/mean conventions. Their captions distinguish final-answer
  accuracy from hidden-state decoding and include the count-0 limitation.
- Local document/image/PDF links and the new supporting-figure anchors resolve.

## Corrections made during this audit

1. Distinguished the initial length-20 example from the current length-40 setting B.
   All train/test comparisons still hold raw length fixed.
2. Corrected Step 2A's reversed description of the hard-setting selection gate:
   unseen mean at most about 70% defines a sufficiently hard setting, not a too-easy one.
3. Replaced "dummy tokens carry no information" with "constant visible tokens carry
   no explicit count labels"; their contextual hidden vectors can be informative.
4. Limited meaningful-versus-dummy conclusions to the tested supervision/history
   comparison, without claiming a unique internal mechanism.
5. Clarified that NoPE removes explicit positional encoding; causal context can
   still supply implicit positional information.
6. Clarified that count-0 distributions are unchanged across separate pools, rather
   than suggesting the same input examples are reused.
7. Restricted claims about extra training to the measured 500-2,000-iteration budget,
   and replaced Step 2A's claim that no scratchpad effect was possible with the
   actual selection decision and limited headroom.
8. Standardized the RoPE baseline at 12.35% from the unrounded result, removing the
   repeated double-rounded 12.4% text. Also corrected Step 1's caption: count-2
   accuracy is nearly perfect, rather than exactly perfect.
9. Made Step 5's "about equal" rule explicitly descriptive, not statistical equivalence.
10. Clarified that the final delimiter and intermediate query pass through the same
    four blocks; they differ in context and trained versus separately fitted readout.
11. Clarified that Step 7's 15 random-model evaluations reuse five distinct random
    initializations, and that its X placements do not include raw positions 9 or 39.
12. Marked Step 6's completed-run commands as historical and documented fresh output
    paths needed for repetition. Earlier logs record this later editorial audit.

## Limits of the verification

The numerical checks establish consistency with the saved data, checkpoints, and
readouts. They do not turn the findings into a causal explanation or a significance
test. Five model seeds share one original raw-data seed; Step 7 also shares one
synthetic-pool seed. The probe pools have a deliberately restricted state-maintenance
distribution. Poor linear transfer does not establish absent information, and good
decoding does not establish causal use by the final answer.

Historical checkpoints do not store optimizer settings or original data hashes.
Matching saved metadata, raw examples, unchanged training/configuration source,
run records, and exact inference reproduction support provenance, but cannot
retroactively recover a complete historical execution manifest. Training was not
repeated; evaluation and readout scoring were.

For the read-only Step 7 model/readout check, run from the experiment directory:

```bash
../venv/bin/python verify_step7.py
```

The output files used by the analysis scripts are not modified by this verifier.
