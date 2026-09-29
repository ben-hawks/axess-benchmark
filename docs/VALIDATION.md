# Validation of the reference-model inference path

This records how we know the benchmark runs the published checkpoints correctly. Every
number here comes from an actual run (2026-09-29, CPU, torch 2.9.0, torch_geometric
2.8.0, rule4ml 0.2.0) over the full HuggingFace test (102,484 samples) and exemplar
(887) splits. The outputs are in `reference_results/`.

## 1. What had to be reconstructed

The GNN and Transformer checkpoints only produce meaningful numbers when fed exactly
the preprocessing they were trained with. Two pieces were not published with them:

| Missing piece | How it was recovered | Check |
|---|---|---|
| JSON -> per-layer features | Vendored `ModelProcessor` from `wa_hls4ml_models/dataset/Dataset_to_csvs6_with_ii.py` (`src/wa_hls4ml_bench/features.py`) | Bit-identical to the original on 2,400 random test/exemplar samples, and on every fixture sample in `tests/test_features_equivalence.py` |
| Normalization stats (feature z-score, log-space label mean/std, log shift) | Recomputed with the original `_calculate_normalization_stats` logic over the HF train split, filtered on "non-empty `hls_resource_report`" (`src/wa_hls4ml_bench/stats.py` -> `weights/normalization_stats.json`) | Filter keeps **440,650** train samples and **94,430** test samples, exactly the `train_size`/`test_size` stored in the GNN checkpoint. The end-to-end result is in §2 |

## 2. Checkpoint-loading check (HLS-estimate labels)

Both checkpoints were trained to predict `hls_resource_report` (C-synthesis estimates),
not the benchmark's post-synthesis ground truth. So correctness of the *loading* path is
checked against those labels (`truth --gt hls_estimate`). These are not benchmark results.

### GNN: exact reproduction

Our run vs paper Table 4 GNN rows (R^2 / RMSE):

| Group | Target | This run | Paper Table 4 |
|---|---|---|---|
| dense | BRAM | -0.51 / 86.1 | -0.51 / 86.1 |
| dense | DSP | -111.74 / 18950.8 | **-0.74** / 18950.8 |
| dense | FF | 0.73 / 56341.8 | 0.73 / 56341.8 |
| dense | LUT | 0.73 / 107544.5 | 0.73 / 107544.5 |
| dense | Cycles | 0.82 / 1304.9 | 0.82 / 1304.9 |
| dense | II | 0.91 / 415.1 | 0.91 / 415.1 |
| conv1d | all 6 targets | 0.69, 0.02, 0.95, 0.96, 0.97, 0.97 | identical (RMSE identical too) |
| conv2d | all 6 targets | 0.44, 0.51, 0.92, 0.95, 0.84, 0.88 | identical (RMSE identical too) |
| all | FF / LUT / Cycles RMSE | 55087.7 / 104945.7 / 227987.9 | 55087.6 / 104945.7 / 227987.8 (also the checkpoint's stored `test_metrics`) |

Every cell matches except two things in the paper table itself:

- **Dense DSP R^2 reads -0.74 in the paper**, but the RMSE (18,950.8, identical) implies
  -111.74. This looks like dropped digits.
- **The paper's GNN "All" BRAM/DSP (R^2 0.51/0.89, RMSE 48.0/580.0) can't be squared with
  its own dense row.** Dense is ~90% of the test set with DSP RMSE ~19k, so the all-sample
  RMSE must be at least ~18k. Our all-sample numbers are BRAM -0.49/84.1 and DSP
  -111.7/18,488. The paper's "All" BRAM/DSP (and II RMSE 201,369.9 vs our 200,878.1)
  match the `test_metrics` stored inside the checkpoint instead, which were evidently
  computed by a different code path during training.

### Transformer: close, not exact

| Group | This run: BRAM, DSP, FF, LUT, Cycles, II (R^2) | Paper Table 4 |
|---|---|---|
| all | 0.43, -1.34, 0.70, 0.61, 0.93, 0.91 | 0.39, 0.29, 0.72, 0.67, 0.95, 0.95 |
| dense | 0.43, -1.34, 0.70, 0.61, 0.95, 0.91 | 0.39, 0.29, 0.71, 0.67, 0.95, 0.91 |
| conv1d | 0.80, 0.49, 0.95, 0.96, 0.95, 0.91 | 0.77, 0.41, 0.97, 0.96, 0.96, 0.96 |
| conv2d | 0.81, 0.54, 0.87, 0.95, 0.90, 0.87 | 0.79, 0.55, 0.93, 0.96, 0.93, 0.93 |

SMAPE agrees closely too (FF 3.3% vs 2.9%, LUT 3.0% vs 2.9%). The pipeline is shared
with the GNN, which reproduces exactly, and the Transformer's inputs differ only in
dropping the global strategy/io_type features. So the remaining gap most likely means
the published `transformer_best_model.pt` (wa_hls4ml_models@4aff94b) is not byte-for-byte
the checkpoint behind the paper table, or it was trained with slightly different
normalization stats. **Open question for the checkpoint's author.** The DSP difference
comes from a few very large `resource`-subset designs (true DSP 35k-55k, predicted
140k-240k); on `3layer` the DSP R^2 is 0.98.

## 3. Baseline MLP

rule4ml 0.2.0's bundled v2 MLP weights, fed each sample's own `model_config`/`hls_config`
(no Keras reconstruction; `src/wa_hls4ml_bench/models/mlp.py`). On the post-synthesis
test set: BRAM 0.32, DSP 0.03, FF 0.20, LUT 0.49, Cycles 0.54, II 0.34 (R^2). The first
five match paper Table 4's MLP "All" row (and its SMAPE values) to the printed precision.
The paper's II row (0.56 / 25.8%) does not match ours (0.34 / 27.0%). The MLP was already
scored against post-synthesis truth in the paper, unlike the GNN/Transformer rows.

## 4. Feature-extraction gap fixed for the exemplar set

All 119 **Bipc** exemplar samples have an `hls_config.LayerName` that the original
extractor can't align with `model_config` (17 vs 12 layers). The original then leaves
precision/reuse factor/strategy unset, so the GNN and Transformer output NaN. The
benchmark fills only those unset entries from the global `hls_config.Model` settings
(`features._fill_from_global_model_config`). Nothing else changes: no test-set sample and
no other exemplar sample is affected, and the equivalence test asserts that every value
the original sets is unchanged.

## 5. Reproducibility on new hardware

`tests/test_pipeline.py` compares GNN and Transformer outputs on 40 fixture samples
(`tests/fixtures/`) against golden predictions from this run (rtol 2e-3). The goldens
agree with the full-split run to <3e-6 relative, so batch composition doesn't matter.
Run it on Perlmutter before submitting jobs (docs/PERLMUTTER.md §1); with
`WA_TEST_DEVICE=cuda` it also checks GPU vs CPU agreement.

**Not yet verified:** an actual Perlmutter run. These scripts were written for Perlmutter
but executed on a Windows workstation (CPU). The first Perlmutter run should confirm
`pytest` passes on a GPU node and that `LEADERBOARD.md` matches `reference_results/`.
