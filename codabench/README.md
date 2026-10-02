# Codabench competition: AXESS wa-hls4ml Benchmark

A results-submission Codabench bundle for this benchmark. Participants run their model
offline and upload predictions for both the test and exemplar splits. Codabench scores
them against the hidden post-synthesis ground truth, using the same metrics as
`src/wa_hls4ml_bench/score.py`.

| | |
|---|---|
| Submission | zip with `predictions_test.csv` + `predictions_exemplar.csv` at its root |
| Phases | one, "Open Benchmark", opens 10-02-2026, no end date, 5 submissions/day |
| Ranking | mean test R² over the 6 targets, tie-break mean test SMAPE; per-target test and exemplar R²/SMAPE also shown (28 columns) |
| Leaderboard rule | `Add_And_Delete_Multiple` (a participant can list several models) |
| Docker image | stock `codalab/codalab-legacy:py312` (the scoring program needs only numpy + pandas) |
| Reference solution | the retrained Transformer's predictions (`solution/`, also `starting_kit/sample_submission.zip`) |

## Layout

```
codabench/
  bundle_src/              hand-written bundle files (versioned)
    competition.yaml
    pages/                 overview, data, evaluation, terms (PLACEHOLDER)
    scoring_program/       scoring.py, metrics.py (vendored), metadata.yaml
    starting_kit/          README.md, make_submission.py
  build_bundle.py          adds the generated files and zips the bundle
  build/                   generated, not versioned
    bundle/                bundle directory (validate this)
    competition_bundle.zip upload this
    extra/baseline_mean_submission.zip   weak baseline, for checking the metrics
```

`build_bundle.py` generates `reference_data/truth_{test,exemplar}.csv` (the hidden
ground truth), `starting_kit/{test,exemplar}_sample_ids.csv`, `solution/`, and
`starting_kit/sample_submission.zip` from a benchmark run's caches and predictions.

## Build

After a benchmark run (docs/PERLMUTTER.md) has produced `$WA_CACHE/{train,test,exemplar}.npz`
and `$WA_RESULTS/{test,exemplar}/predictions_transformer.csv`:

```bash
PYTHONPATH=src python codabench/build_bundle.py --cache-dir $WA_CACHE \
    --results-dir $WA_RESULTS --out codabench/build
```

## Validate

With the benchmark-builder skill's validator:

```bash
python validate_codabench_bundle.py codabench/build/bundle \
    --submission codabench/build/bundle/starting_kit/sample_submission.zip --docker --no-build
```

Last validated 2026-10-02:

| Tier | Result |
|---|---|
| 1 structure | passes except the missing logo (below) |
| 2 submission contract | passes: both files found at the zip root |
| 3 local dry run | passes. Scores equal `reference_results/` exactly (Transformer test mean R² 0.809, exemplar −0.517; the GNN, packaged with `make_submission.py`, 0.780 / −1.956) |
| 4 Docker run in `codalab-legacy:py312` | **not yet run**: the local Docker backend (Rancher Desktop) refused containers ("timed out dialing Hyper-V socket") |

The metrics separate the reference from the weak baseline: the training-mean
submission scores test mean R² 0.000 and SMAPE 114% (the reference: 0.809 and 10.3%).
Malformed submissions (a missing sample, a NaN, a missing file) fail with a message
naming the problem. A wrapping folder and extra unscored rows are tolerated.

## Before uploading

1. **Logo:** add `bundle_src/logo.png` and rebuild. `competition.yaml` requires `image`,
   and Codabench will reject the bundle without it.
2. **Terms:** `bundle_src/pages/terms_and_conditions.md` is a placeholder. Replace it
   with real terms written or reviewed by the organizers.
3. Run tier 4 once Docker works locally.
4. Upload `build/competition_bundle.zip` on codabench.org (Benchmarks → Management →
   Upload).

Scoring is CPU-only and fast (well under a minute for both splits), so no custom
image or GPU compute worker is needed.
