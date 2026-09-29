# Reference solutions

Three pretrained surrogate models, positioned as a ladder of increasingly structured
priors (paper §4). The benchmark only runs inference with the published weights.
Training code and history live upstream and are linked for provenance.

All three are run by `python -m wa_hls4ml_bench.predict --model {mlp,gnn,transformer}`
and scored with the same `score.py` on the same test and exemplar samples. Results:
[../reference_results/LEADERBOARD.md](../reference_results/LEADERBOARD.md), with
per-group tables in `../reference_results/<split>/<model>/METRICS.md`.

## Summary

| | Baseline MLP | GNN | Transformer |
|---|---|---|---|
| Code in this repo | `src/wa_hls4ml_bench/models/mlp.py` (wraps rule4ml) | `models/gnn.py` (vendored) | `models/transformer.py` (vendored) |
| Upstream | [IMPETUS-UdeS/rule4ml](https://github.com/IMPETUS-UdeS/rule4ml) (`trets` branch), paper §4.1 | [jdweitz/wa_hls4ml_models](https://github.com/jdweitz/wa_hls4ml_models) `GNN/`, paper §4.2 | same repo, `transformer/`, paper §4.3 |
| Weights | bundled in `rule4ml==0.2.0` (`models/weights/v2/mlp/*`) | `gnn_final_model.pth`, sha256 `7a8e9c42…` | `transformer_best_model.pt` (wa_hls4ml_models@4aff94b), sha256 `83046ae9…` |
| Input representation | global + per-layer statistics from `model_config`/`hls_config` (rule4ml parser) | graph: one node per layer, 33-dim node features, sequential edges, 4-dim global one-hot | sequence of 33-dim layer tokens + `[CLS]`, max 51 layers |
| Architecture | 6 independent per-target MLPs with categorical embeddings | 5× GATv2Conv (5 heads × 512, concat), LayerNorm, ELU, residual; learned add/mean/max pooling; MLP 516→512→256→6 | Linear 33→512 + learned positions; 2× TransformerEncoderLayer (8 heads, FF 512); linear head on `[CLS]` |
| Training labels | post-synthesis `resource_report` | **HLS estimate `hls_resource_report`**, log-transformed, z-scored | **HLS estimate `hls_resource_report`**, log-transformed, z-scored |
| Training (upstream) | 200 epochs, Adam, MSLE | AdamW lr 3e-3, wd 5e-6, batch 1024, MSE, ReduceLROnPlateau; best checkpoint at epoch ≤180 of the run; NVIDIA A10 | Adam, MSE, batch 1024; NVIDIA A100 |
| Parameters | small (per-target MLPs) | 54.5 M | 3.2 M |
| Inference hardware used here | CPU (TensorFlow) | CPU or 1 GPU | CPU or 1 GPU |
| Measured cost (Windows workstation CPU, incl. featurization) | ~58 ms/sample exemplar* | 2.1 ms/sample (test) | 0.34 ms/sample (test) |

\*The MLP cost is dominated by per-sample rule4ml feature parsing and TensorFlow
startup, and amortizes over larger splits. GPU timings on Perlmutter will be recorded on
the first run.

## Preprocessing the checkpoints depend on

`src/wa_hls4ml_bench/features.py` has two stages:

1. **Raw per-layer features** (18 per layer: input/output dims ×3 each, precision, reuse
   factor, strategy, layer type, activation, filters, kernel size, stride, padding,
   pooling, batch-norm, I/O type). This is vendored verbatim from
   `wa_hls4ml_models/dataset/Dataset_to_csvs6_with_ii.py`, including its quirks.
   A test proves it is bit-identical to the original.
2. **Encoding**: 12 numerical features z-scored (−1 treated as 0), plus one-hot layer
   type (12), activation (6), and padding (3), giving 33 dims. For the GNN it adds
   one-hot strategy (2) and I/O type (2) as graph-level features.

Outputs are de-normalized as `exp(y·σ + μ) − shift`, clamped at 0. The label/feature
statistics (`weights/normalization_stats.json`) were not published with the checkpoints.
They were regenerated from the training split; see ../docs/VALIDATION.md for how that
was verified (the GNN reproduces paper Table 4 exactly on HLS-estimate labels).

## Why the GNN/Transformer score poorly on resources

Both were trained to reproduce HLS C-synthesis *estimates*, and the benchmark's ground
truth is post-logic-synthesis. The two differ systematically: HLS over-reports BRAM
(e.g. 32 vs 2), and LUT/FF shift after logic optimization. Latency comes from the same
report in both cases, which is why both models are the best latency predictors
(R² 0.89–0.93) while being worse than the mean on resources. This isn't a pipeline error.
Scored against the labels they were trained on, the same predictions give LUT R² 0.73
(GNN) and 0.61 (Transformer) (`../reference_results/checkpoint_check/`).
