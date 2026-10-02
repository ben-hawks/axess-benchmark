# Reference solutions

Three pretrained surrogate models, positioned as a ladder of increasingly structured
priors (paper §4), plus one auxiliary comparison model. The benchmark only runs
inference with published weights. Training code and history live upstream and are
linked for provenance.

Everything is run by `python -m wa_hls4ml_bench.predict --model {mlp,gnn,transformer,rule4ml_gnn}`
and scored with the same `score.py` on the same test and exemplar samples. Results:
[../reference_results/LEADERBOARD.md](../reference_results/LEADERBOARD.md), with
per-group tables in `../reference_results/<split>/<model>/METRICS.md`.

## Summary

| | Baseline MLP | GNN | Transformer | rule4ml GNN (auxiliary) |
|---|---|---|---|---|
| Code in this repo | `src/wa_hls4ml_bench/models/mlp.py` (wraps rule4ml) | `models/gnn.py` (vendored) | `models/transformer.py` (vendored) | `models/mlp.py`, `kind="gnn"` (wraps rule4ml) |
| Upstream | [IMPETUS-UdeS/rule4ml](https://github.com/IMPETUS-UdeS/rule4ml) (`trets` branch), paper §4.1 | [ben-hawks/wa_hls4ml_models](https://github.com/ben-hawks/wa_hls4ml_models) @ `resource-report-retrain`, `GNN/`, paper §4.2 | same, `transformer/`, paper §4.3 | rule4ml |
| Weights | bundled in `rule4ml==0.2.0` (`models/weights/v2/mlp/*`) | release asset `gnn_resource_report_final_model.pth`, sha256 `ac8bbfbf…` | release asset `transformer_resource_report_final_model.pt`, sha256 `5e8726c3…` | bundled in `rule4ml==0.2.0` (`models/weights/v2/gnn/*`) |
| Input representation | global + per-layer statistics from `model_config`/`hls_config` (rule4ml parser) | graph: one node per layer, 33-dim node features, sequential edges, 4-dim global one-hot | sequence of 33-dim layer tokens + `[CLS]`, max 51 layers | rule4ml graph features (includes Vivado version) |
| Architecture | 6 independent per-target MLPs with categorical embeddings | 5× GATv2Conv (5 heads × 512, concat), LayerNorm, ELU, residual; learned add/mean/max pooling; MLP 516→512→256→6 | Linear 33→512 + learned positions; 2× TransformerEncoderLayer (8 heads, FF 512); linear head on `[CLS]` | 6 per-target GIN models (GINConv, GraphNorm, JumpingKnowledge, attentional pooling) |
| Training labels | post-synthesis `resource_report` | post-synthesis `resource_report` (FF/LUT/DSP int, BRAM float) + HLS-estimate `latency_report`; log-transformed (ε = 1e-6), z-scored | same as GNN | post-synthesis (rule4ml's own dataset) |
| Training (upstream) | 200 epochs, Adam, MSLE | AdamW lr 3e-3, wd 5e-6, batch 1024, MSE, ReduceLROnPlateau, early stopping: stopped at epoch 105 of 200, best 65 (upstream README) | Adam lr 1e-5, batch 512, MSE, 200 epochs, best 197 | see rule4ml |
| Output post-processing | none | `exp(y·σ + μ) − shift`, clamp at 0, **cap at the largest training label** | same as GNN | none |
| Parameters | small (per-target MLPs) | 54.5 M | 3.2 M | small (per-target GNNs) |
| Inference hardware | CPU (TensorFlow) | CPU or 1 GPU | CPU or 1 GPU | CPU (torch) |
| Measured cost, Perlmutter (test split, 102,484 samples, whole command) | 376.5 s = 3.7 ms/sample (32 CPU cores) | 30.1 s = 0.29 ms/sample (NVIDIA A100-SXM4-40GB) | 11.0 s = 0.11 ms/sample (A100) | 900.7 s = 8.8 ms/sample (32 CPU cores) |
| Measured cost, workstation CPU (test split; model only in brackets) | — | 100.4 s = 0.98 ms/sample (0.87) | 16.2 s = 0.16 ms/sample (0.09) | — |

The Perlmutter numbers come from the 2026-10-02 run's job logs (`wa-infer-gpu-59209566.out`,
`wa-infer-rule4ml-59209567.out`). They cover the whole command: loading weights and the
cache, featurization, inference, writing the CSV. Model-only time on the GPU wasn't
recorded then; since 2026-10-02 every run writes it to `<results>/TIMINGS.md`
(`predict.py`'s `.timing.json` files), split into total and model-only time. The workstation
row is from an Intel CPU with 32 threads, run the same day. The rule4ml models spend most
of their time in per-sample feature parsing, not in the networks.

**Why rule4ml's GNN is auxiliary.** It is a different architecture from the paper's GNN
(GIN rather than GATv2) and isn't part of the paper's reference ladder. It's included
because it's a second, independently trained post-synthesis graph model that ships with
pretrained weights, which makes it a useful sanity comparison. On the test set it has
the best BRAM R² of any model here (0.70), but it is far weaker than the retrained
GATv2 GNN on DSP/FF/LUT.

## Preprocessing the GNN/Transformer depend on

`src/wa_hls4ml_bench/features.py` has two stages:

1. **Raw per-layer features** (18 per layer: input/output dims ×3 each, precision, reuse
   factor, strategy, layer type, activation, filters, kernel size, stride, padding,
   pooling, batch-norm, I/O type). This is vendored verbatim from
   `wa_hls4ml_models/dataset/Dataset_to_csvs6_with_ii.py`, including its quirks. A test
   proves it is bit-identical to the original at the `resource-report-retrain` commit.
2. **Encoding**: 12 numerical features z-scored (−1 treated as 0), plus one-hot layer
   type (12), activation (6), and padding (3), giving 33 dims. For the GNN it adds
   one-hot strategy (2) and I/O type (2) as graph-level features.

Both retrained models were trained on the same arrays with the same transform, so they
share one set of statistics and caps (`weights/normalization_stats.json`). These agree
with the stats file shipped in the GNN release to float32 precision (../docs/VALIDATION.md §2).

**The cap.** Upstream `transformer/run.py` and `GNN/load_pretrained.py` clip predictions
at the per-target training maximum. For a few inputs, the log-space output otherwise
extrapolates to physically impossible values (up to ~2 M DSPs). On the test set the cap
changes 124 Transformer and 9 GNN predictions out of 557,598. It is part of each model's
published inference procedure, so the benchmark applies it.

## The original (HLS-estimate) checkpoints

The checkpoints behind the paper's Table 4 (`gnn_final_model.pth`, and
`transformer_best_model.pt` at wa_hls4ml_models@4aff94b) were trained on
`hls_resource_report`, so they predict HLS C-synthesis estimates, not the post-synthesis
counts the benchmark scores. Scored against post-synthesis truth, they're worse than
predicting the mean on all four resource targets. They are therefore no longer reference
solutions. ../docs/VALIDATION.md §5 records how they were verified and what they score.
