# Evaluation

## What to submit

A zip file with two CSV files **at its root** (no wrapping folder):

| File | Rows |
|---|---|
| `predictions_test.csv` | every scored test sample (92,933) |
| `predictions_exemplar.csv` | every scored exemplar sample (886) |

Both files have the columns:

```
sample_id,BRAM,DSP,FF,LUT,cycles_max,interval_max
```

with `sample_id` as defined on the **Data** page and predictions as absolute counts (not
percentages of the device). A prediction is required for every scored sample, and every
value must be a finite number; otherwise the submission fails with a message saying
what's wrong. Extra rows are ignored. The starting kit has the scored id lists and a
ready-to-upload example submission.

## Metrics

Computed per target, against post-logic-synthesis ground truth (paper Section 3.2,
Eq. 1–3), separately for the test set and the exemplar set:

| Metric | Definition | Better |
|---|---|---|
| R² | `1 − Σ(y−ŷ)² / Σ(y−ȳ)²` | higher |
| SMAPE | `200%/n · Σ |y−ŷ| / (|y|+|ŷ|+1)`. ε = 1 keeps y = ŷ = 0 at 0% | lower |
| RMSE | `sqrt(mean((y−ŷ)²))`, native units | lower |

**Ranking:** mean test R² over the six targets, with ties broken by mean test SMAPE.
The leaderboard also shows per-target test R² and SMAPE, and the same for the exemplar
set.

The scoring log of each submission contains the full tables, including RMSE, for the
dense / Conv1D / Conv2D test groups and for each exemplar architecture, as reported in
paper Tables 4–5. RMSE isn't ranked, because it's dominated by the largest designs and
favors models that under-predict them (paper Section 5.2).

## Reproducing the scores offline

The same metrics are implemented in the benchmark package
(`python -m wa_hls4ml_bench.score`, see
[github.com/ben-hawks/axess-benchmark](https://github.com/ben-hawks/axess-benchmark)).
That repository's `reference_results/` contains the reference solutions' full results.
