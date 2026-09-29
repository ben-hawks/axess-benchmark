# Submission Report: {{SOLUTION_NAME}} on wa-hls4ml

Adapted directly from the paper's own three-tier submission guidelines
(arXiv:2511.05615, Section 3.1) — Fill in the sections below for a new resource/latency
surrogate model submitted against this benchmark.

## Required

- [ ] **Predicted values** for each of the 6 FPGA metrics (BRAM, DSP, FF,
      LUT, cycles_max, interval_max) — see `data/SCHEMA.md` — for every sample
      of the test set and of the exemplar set, as `predictions_<name>.csv`
      (`sample_id` + 6 target columns). Coverage must be complete.
- [ ] **Visual comparison** between predicted and actual values, as box plots
      of relative percent error (RPE) per target variable (paper Figures
      7-12 style). `scripts/score_all.sh` (or
      `python -m wa_hls4ml_bench.score`) writes these as `rpe_boxplot.png`.
- [ ] **Metric table**: R², SMAPE, and RMSE (paper Eq. 1-3), computed exactly
      as specified in the benchmark card (`README.md` Section 3) against the
      post-synthesis ground truth,
      reported per target variable, for both the test set and the exemplar
      set.

## Strongly recommended

- [ ] **Architecture/method description**: enough detail to reimplement
      without reading the submitted code — design choices, key
      hyperparameters, and why they were chosen (see `reference_solution/README.md`
      for the level of detail expected — architecture diagram, feature
      preprocessing, training procedure, epoch count, optimizer, loss).
- [ ] **Source code and trained weights**, shared openly, to support
      reproducibility and direct comparison by future submitters.
- [ ] **Inference hardware specification** and measured inference time per
      sample — a resource/latency prediction that takes seconds is the whole
      point of this benchmark (vs. hours for real synthesis, paper Fig. 1),
      so report the actual number, not just "fast."

## Suggested / as applicable

- [ ] **Additional constraints used**: any extra training data (e.g.
      externally-generated hls4ml synthesis samples beyond the published
      training split), additional target boards/hls4ml configurations, or
      specific precision/optimization strategies applied during evaluation —
      document these so results aren't silently non-comparable to other
      submissions (see benchmark card Section 2, "Bounded-ness").

It is worth noting the upstream benchmark's own submission process is open
and ongoing, with no fixed release schedule — the dataset and reference
solutions are expected to be periodically updated (paper Section 3.1,
Section 6). A submission must record the dataset revision it was evaluated against
(`$WA_DATA/REVISION`, written by `scripts/fetch_data.py`) and the commit of
this repository used to score it.

---

## Metric results

| Metric | BRAM | DSP | FF | LUT | Cycles | II |
|---|---|---|---|---|---|---|
| R² | | | | | | |
| SMAPE [%] | | | | | | |
| RMSE | | | | | | |

(Report once for the test set, and once for the exemplar set. Copy from
`<results>/<split>/<name>/METRICS.md`; compare with `reference_results/`.)

## Visual comparisons

_(embed or link `<results>/{test,exemplar}/<name>/rpe_boxplot.png`)_

## Reproduction

```bash
# exact commands to regenerate this report's numbers from a clean environment
# place predictions_<name>.csv under $WA_RESULTS/test/ and $WA_RESULTS/exemplar/, then:
source perlmutter/env.sh && wa_activate_torch
bash scripts/score_all.sh
```
