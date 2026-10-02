# Validation of the reference-model inference path

This records how we know the benchmark runs the published checkpoints correctly. Every
number comes from an actual run (2026-10-02, Windows workstation CPU, torch 2.9.0,
torch_geometric 2.8.0, rule4ml 0.2.0) over the full HuggingFace test (102,484 samples)
and exemplar (887) splits. Outputs are in `reference_results/`.

Reference GNN/Transformer weights: [ben-hawks/wa_hls4ml_models release
`resource-report-retrain`](https://github.com/ben-hawks/wa_hls4ml_models/releases/tag/resource-report-retrain)
(tag at commit `ac394e9`). Both were retrained on post-synthesis `resource_report` labels.

## 1. Feature extraction

`src/wa_hls4ml_bench/features.py` vendors `ModelProcessor` from
`wa_hls4ml_models/dataset/Dataset_to_csvs6_with_ii.py`. The retrain release only
refactored that file (`process_record`, a configurable label key). Feature extraction is
unchanged. `tests/test_features_equivalence.py` asserts bit-identical output against the
submodule at the release tag on every fixture sample. An earlier run also matched the
original on 2,400 random test/exemplar samples.

## 2. Normalization statistics and the prediction cap

Both retrained models were trained on the same arrays with the same log transform
(ε = 1e-6). The Transformer's `train.log` prints statistics identical to the GNN's
shipped `resource_report_results/gnn/normalization_stats_log.npy`, so one set serves both.

`python -m wa_hls4ml_bench.stats` rebuilds them from the HF train split, using labels
exactly as the upstream converter built them (`data.truth_training_labels`):

| Check | Result |
|---|---|
| Train samples kept | **433,676**, matching the checkpoint's `dataset_info.train_size` and the release README |
| Feature means / stds vs shipped `.npy` | max relative difference 5.6e-8 / 4.2e-8 (float32 rounding) |
| Label means / stds vs shipped `.npy` | max relative difference 5.7e-8 / 2.9e-8 |
| log shift | 1e-6 (identical) |

The converter keeps a sample when its `resource_report` is non-empty and treats a missing
latency report as 0. The benchmark's ground truth also requires a latency report, which
drops 2 more train samples (433,674). That's why the stats use the converter's rule
rather than the benchmark's.

The cap is the per-target maximum of those training labels (Cycles 38,349,241; FF
1,570,553; LUT 2,265,699; BRAM 1,029.5; DSP 12,280; II 38,349,236). It matches upstream
`transformer/run.py` and `GNN/load_pretrained.py`.

## 3. Per-sample agreement with the release's own predictions

The release ships `resource_report_results/{gnn,transformer}/test_predictions.npz`,
written by its own evaluation scripts (uncapped). Aligned row by row with the benchmark's
predictions (labels checked equal first), after applying the cap to theirs:

| Model | Samples | max \|ours − theirs\| / (\|theirs\| + 1) | 99.9th percentile | Predictions capped |
|---|---|---|---|---|
| GNN | 92,933 | 2.8e-4 | 5.9e-6 | 9 |
| Transformer | 92,933 | 2.1e-5 | 6.6e-6 | 124 |

The residual is CPU-vs-GPU arithmetic. The cap counts match the release README exactly.
The benchmark's test R² equals the release README's to every printed digit, for all,
dense, conv1d and conv2d groups:

| R² (BRAM, DSP, FF, LUT, Cycles, II) | Benchmark | Release README |
|---|---|---|
| GNN, all | 0.64, 0.56, 0.93, 0.90, 0.81, 0.83 | 0.64, 0.56, 0.93, 0.90, 0.81, 0.83 |
| Transformer, all | 0.35, 0.83, 0.93, 0.90, 0.93, 0.92 | 0.35, 0.83, 0.93, 0.90, 0.93, 0.92 |

SMAPE differs because the benchmark uses ε = 1, as paper Eq. 2 specifies. The release
reports ε = 1e-8 ("raw") and a version with BRAM/DSP rounded. Only BRAM and DSP, with
their many zero counts, are affected.

## 4. Feature-extraction gap fixed for the exemplar set

All 119 **Bipc** exemplar samples have an `hls_config.LayerName` that the original
extractor can't align with `model_config` (17 vs 12 layers). The original then leaves
precision/reuse factor/strategy unset (NaN), and the GNN and Transformer output NaN.
The benchmark fills only those unset entries from the global `hls_config.Model` settings
(`features._fill_from_global_model_config`). No test-set sample and no other exemplar
sample is affected, and the equivalence test asserts that every value the original sets
is unchanged.

**Consequence worth knowing:** the retrained GNN handles these filled-in Bipc samples
badly. Its exemplar Cycles RMSE is 9,497 on Bipc vs ~690 on every other architecture,
which drives its exemplar Cycles/II R² to −6.9/−5.8. The Transformer and both rule4ml
models handle the same samples normally. Either the fallback values are out of
distribution for the GNN, or this is a GNN-specific generalization failure; it is not
resolved here.

## 5. History: the original (HLS-estimate) checkpoints

Paper references below are to the published article (ACM TRETS 19(2), 2026,
doi:10.1145/3787490). Its section, equation, table and figure numbering is the same as
the arXiv preprint's, and every Table 4 value quoted here was checked against the
published table.

The checkpoints behind paper Table 4 (`gnn_final_model.pth`, and
`transformer_best_model.pt` at wa_hls4ml_models@4aff94b) were reference solutions in
this repo until the retrain release. What was established about them:

- **They were trained on `hls_resource_report`.** The original converter read that key.
  The training-set filter on it reproduces the GNN checkpoint's stored split sizes
  (440,650 / 94,430). Every label row of `Full_dataset_processed_split.zip` (the
  arrays they were trained on) equals an HF sample's `hls_resource_report` values: 100% of
  440,650 train and 94,430 test rows. Zero rows match `resource_report`, even on the
  resource columns alone.
- **The GNN reproduced paper Table 4 exactly on those labels**, for R², RMSE and SMAPE in
  every dense/conv1d/conv2d cell. SMAPE only matches with ε = 1e-8, as in the training
  code's `calculate_metrics`, **not ε = 1 as paper Eq. 2 states**. The table itself has
  two anomalies:
  - Dense DSP R² is printed −0.74, but the identical RMSE implies −111.74.
  - The "All" BRAM/DSP entries match the checkpoint's stored training-time metrics and
    can't be consistent with the dense row.
- **The Transformer reproduced Table 4 closely but not exactly** (e.g. dense BRAM R²
  0.43 vs 0.39), so the published file was probably not the exact checkpoint behind the
  table.
- **Table 4 mixes ground truths.** Its MLP row matches the rule4ml MLP scored against
  post-synthesis `resource_report` (5 of 6 targets to the printed precision); its GNN and
  Transformer rows are HLS estimates.
- Against post-synthesis truth, both HLS checkpoints were worse than predicting the mean on
  all four resource targets (test R² from −1.4 to −96). Latency was fine, because both
  label sets share `latency_report`.

## 6. rule4ml models

- **Baseline MLP** (rule4ml 0.2.0 bundled v2 weights, fed each sample's own
  `model_config`/`hls_config` without Keras reconstruction): test R² 0.32, 0.03, 0.20,
  0.49, 0.54, 0.34. The first five match paper Table 4's MLP row; II does not (0.34 vs 0.56).
- **rule4ml GNN** (auxiliary): it is trained on post-synthesis labels, like the MLP. It
  scores BRAM R² 0.70 against post-synthesis truth vs 0.09 against HLS estimates, with
  FF/LUT ~0.5 vs ~0.13. Test mean R² is 0.55, consistent with an earlier independent run
  (0.547).

## 7. Reproducibility on new hardware

`tests/test_pipeline.py` compares GNN and Transformer outputs on 40 fixture samples
(`tests/fixtures/`) against golden predictions from this run (rtol 2e-3). The goldens
agree with the full-split run to <4e-6 relative. Run it on Perlmutter before submitting
jobs (docs/PERLMUTTER.md §1). With `WA_TEST_DEVICE=cuda` it also checks GPU vs CPU
agreement. The rule4ml models aren't in the golden test (they need the separate
TensorFlow environment).

### Perlmutter run (2026-10-02)

The full workflow (`perlmutter/submit.sh`, since reorganized as `slurm/submit.sh` with a
Perlmutter profile: featurize, GPU inference for the GNN and
Transformer, CPU inference for the rule4ml models, scoring) ran on NERSC Perlmutter as
Slurm jobs 59209565–59209568 (account amsc011). All four completed with exit 0, and the
score job logged no errors. It used the retrained `*_resource_report_final_model.*`
checkpoints.

| Check | Result |
|---|---|
| `pytest` on Perlmutter | 14 passed, including bit-equivalence with wa_hls4ml_models@ac394e9 |
| Mean R², test / exemplar | transformer 0.809 / −0.517, gnn 0.780 / −1.956, rule4ml_gnn 0.549 / −1.910, mlp 0.319 / 0.248; identical to `reference_results/` at the printed precision |
| Every `metrics.json` cell (R², SMAPE, RMSE, every group) vs `reference_results/` | coverage and sample counts identical. Max relative difference 2.4e-4 (exemplar GNN DSP R²: 0.037852 vs 0.037861, an absolute difference of about 9e-6); every other cell ≤ 2.5e-5 |
| Codabench submissions (`python -m wa_hls4ml_bench.submission`) | exit 0. All four zips written, each with 92,933 test + 886 exemplar rows |

So the GPU run on Perlmutter reproduces the CPU reference results to within
floating-point differences. These numbers come from the report of the Claude session
that ran the jobs; the run's outputs are under the user's `$SCRATCH` on Perlmutter,
not in this repository.

Timings from the same jobs' logs (whole command: loading, featurization, inference,
writing the CSV):

| Job | Model | Test split (102,484) | Exemplar (887) |
|---|---|---|---|
| 59209565 featurize, 64 CPU cores | — | 20.7 s | 0.6 s |
| 59209566 infer-gpu, NVIDIA A100-SXM4-40GB | GNN | 30.1 s (0.29 ms/sample) | 6.9 s |
| | Transformer | 11.0 s (0.11 ms/sample) | 3.3 s |
| 59209567 infer-rule4ml, 32 CPU cores | MLP | 376.5 s (3.7 ms/sample) | 13.6 s |
| | rule4ml GNN (auxiliary) | 900.7 s (8.8 ms/sample) | 17.0 s |

The exemplar times are dominated by fixed startup cost. Model-only GPU time wasn't
logged by that run; later runs write it to `<results>/TIMINGS.md`.
