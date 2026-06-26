<!--
EXPERIMENT-REPORT TEMPLATE  (copy to log/YYYY-MM-DD-<slug>.md, then fill in)

This is the *polished, per-result* log — one file per experiment, written so a future reader
(or thesis chapter) can trust and reproduce it. Keep the separate running/activity log for the
day-to-day "what I did and why, including dead ends".

It follows the reproducibility rules we adopted (Sandve 2013 / Wilson 2017):
  - record HOW each result was produced: commit + exact command + config + seeds + env  (R1,R3,R4,R6)
  - the numbers/figures sit on top of committed raw data (results.csv / predictions.csv)     (R5,R7)
  - every claim links to the result that backs it; mark guesses as hypotheses                 (R9)
Delete these comments when you fill the file in.
-->

# <Task> — <one-line what this run tests> (Phase N)

*YYYY-MM-DD · commit `<short-sha>` · John Lee*

> **Main takeaway.** <the 3-second conclusion in one or two sentences.>

### The task

<one sentence + ONE concrete example, e.g. `...X...Y... : T`. Define any acronyms on first use,
e.g. `none` = NoPE (no positional encoding), `learned` = APE (absolute position embedding).>

---

## Setup  <!-- R1/R3/R6: enough to reproduce exactly -->

- **data / split:** <length, classes + balance, train/val/test rule, sizes>; chance = <x%>.
- **model / config:** <n_layer / n_head / n_embd / params>, <iters, optimizer, lr>, device.
- **seeds:** <which seeds; say if they vary init-only vs data-too>.
- **env:** python <ver> · torch <ver> · numpy <ver> (· matplotlib <ver> for figures); data `SEED=<n>`.
- **reproduce:**
  ```bash
  python data/order/prepare.py            # writes the split this run used
  python train.py    config/basic.py --pos_type=<...> --seed=<...>
  python evaluate.py config/basic.py --pos_type=<...> --seed=<...>   # appends results.csv (+ predictions.csv)
  python plot.py --split=<...> --out_dir=log/figures                 # regenerates the figures below
  ```

---

## Results  <!-- R5/R7/R9: tables + figures, with the raw data committed behind them -->

<lead with a summary table; one row per condition. Report per-class where a single-label collapse
would otherwise hide behind an aggregate. Note in-distribution val alongside held-out test.>

| <condition> | <metric> | ... |
|---|---|---|

**Figures** (from `results.csv` / `predictions.csv` via `plot.py`; committed under `figures/`):

<img src="figures/<name>.png" alt="<what it shows>" width="...">

---

## Why / interpretation

<the mechanism or explanation. **Mark anything not directly measured as a hypothesis** and say
what would confirm it.>

## Caveats / limitations  <!-- be honest here; this is what makes the log trustworthy -->

<e.g. single seed, one fixed data split (init-only variance), saturated metric, small n, etc.>

## Relation to prior work  <!-- optional; only papers you've actually checked -->

<cite only what you've read/verified; otherwise note "to read before relying on it".>

## Next

1. <the next concrete step this result motivates>
