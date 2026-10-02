# Dataset schema

Source: wa-hls4ml paper Section 2.1.3 ("Dataset Structure"), cross-checked
against `wa-hls4ml-search/gen_models_documentation.md` and
`wa-hls4ml-search/util/json_dataset_merge.py` (github.com/ben-hawks/wa-hls4ml-search)'s output format, and against
direct inspection of the real HuggingFace test-set JSON files (see
`docs/VALIDATION.md`). All splits (train/validation/test/exemplar) share
this same schema — one JSON record per synthesized sample, with one small
variation: the legacy `2_20` subset has 10 top-level fields (using
`backend`/`backend_version` for the toolchain-version constraint), while
the other 6 test subsets have 9 (using `vivado_version` for the same
purpose instead) — see the `vivado_version` / `backend`+`backend_version`
rows below.

## Distribution

| Split | Samples | File location (results dataset) |
|---|---|---|
| Training | 478,220 | HuggingFace `fastmachinelearning/wa-hls4ml`, train split |
| Validation | 102,472 | HuggingFace `fastmachinelearning/wa-hls4ml`, validation split |
| Test | 102,484 | HuggingFace `fastmachinelearning/wa-hls4ml`, test split |
| Exemplar | 887 | HuggingFace `fastmachinelearning/wa-hls4ml`, exemplar split |

Within train/validation/test (683,176 total, excluding the exemplar set),
samples are additionally categorized by generation method — informational
only, **not** a further train/eval split:

| Category | Description |
|---|---|
| `2_20` | Updated `rule4ml` dataset: fully-connected, 2-20 layers, randomly generated, hls4ml resource/latency strategies |
| `2_layer` | 2-layer fully-connected, grid search, resource strategy, `io_parallel` |
| `3_layer` | 3-layer fully-connected, grid search, resource strategy, `io_parallel` |
| `conv1d` | 3-7 layer 1D-conv, randomly generated, resource strategy, `io_stream` |
| `conv2d` | 3-7 layer 2D-conv, randomly generated, resource strategy, `io_stream` |
| `latency` | 3-7 layer fully-connected, randomly generated, latency strategy, `io_parallel` |
| `resource` | 3-7 layer fully-connected, randomly generated, resource strategy, `io_parallel` |

## Top-level fields (every sample)

| Field | Type | Contents |
|---|---|---|
| `meta_data` | object | A unique identifier (`uuid`, or `model_id` for the `2_20` subset -- this is the benchmark's `sample_id`), the model name, and the name of the corresponding gzipped tarball of the full synthesized project (logs/reports/RTL) in the companion Projects dataset (`fastmachinelearning/wa-hls4ml-projects`). |
| `model_config` | list | A **flat, ordered list of per-layer dicts** — not a raw Keras `model.to_json()` dump. Each layer dict has `class_name`, `name`, `input_shape`, `output_shape`, `parameters`, `trainable_parameters`, `dtype`, `reuse_factor` (the **actual**, post-synthesis reuse factor), plus per-class fields (`neurons`/`use_bias` for Dense/QDense; `channels`/`filters`/`kernel_size`/`strides`/`padding` for Conv1D/2D/QConv1D/QConv2D; `activation` for Activation/QActivation). Confirmed by direct inspection to be schema-identical to what `rule4ml.parsers.network_parser.config_from_keras_model()` derives from a live Keras model — see `src/wa_hls4ml_bench/models/mlp.py` for why this means a reference solution can ingest this field directly, with no Keras model reconstruction needed. |
| `hls_config` | object | The hls4ml conversion configuration actually used, nested as `{"Model": {"Precision", "ReuseFactor", "Strategy", "BramFactor", "TraceOutput"}, "clock_period", "io_type"}` — `Model.ReuseFactor`/`Strategy`/`Precision` are the **target** values as requested (compare against `model_config`'s per-layer `reuse_factor` to see whether hls4ml honored the request). `Precision` is usually a plain string (e.g. `"ap_fixed<16, 6>"`) but is sometimes a dict (`{"default": "fixed<16,6>"}`) — normalize before use. |
| `resource_report` | object | Post-**logic**-synthesis resource usage — the ground-truth regression targets: string-valued component counts for `bram`/`dsp`/`ff`/`lut` (lowercase keys; cast to float/int before use). This is the number a full Vivado/Vitis logic-synthesis run reports, not a C-synthesis estimate. **Empty (`{}`) for ~9.3% of the real test set** — those synthesis runs didn't complete or weren't recorded; treat as missing, not zero. |
| `hls_resource_report` | object | Post-**HLS** (C-synthesis) resource *estimate*, same field names/format as `resource_report`. Confirmed on real data to be a measurably different (generally larger) number than `resource_report` for the same sample — do not use as a fallback ground truth when `resource_report` is present for some samples and missing for others in the same evaluation, or you'll silently mix two different ground-truth definitions. |
| `latency_report` | object | Post-**HLS** (C-synthesis) latency estimates — the dataset has no post-synthesis latency, so these are the ground-truth latency targets: `cycles_min`, `cycles_max`, `target_clock`, `estimated_clock`, `interval_min`, `interval_max` (all string-valued). Empty alongside `resource_report` for the same ~9.3% of missing samples. |
| `target_part` | string | The FPGA part targeted for HLS and logic synthesis, e.g. `xcu250-figd2104-2L-e` (Alveo U250) or `xc7z020clg400-1` (Pynq-Z2, legacy `2_20` subset only) — a system constraint (see benchmark card Section 1), not a prediction target. Reverse-mappable to a board name via `rule4ml`'s own `parsers/supported_boards.json`. |
| `vivado_version` | string | The AMD Vivado/Vitis version used to synthesize the sample (e.g. `"2023.2"`, `"2024.2"`) — a top-level key, present and 100%-populated in 6 of the 7 real test-set files, confirmed by direct inspection (uniform per file: `"2024.2"` for `resource`/`conv1d`/`conv2d`/`latency`, `"2023.2"` for `2layer`/`3layer`). **Not present at all in the legacy `2_20` subset** — that subset instead has `backend`/`backend_version` (below) carrying the same information under different key names. |
| `backend` / `backend_version` | string, `2_20` subset only | Present **only** in the legacy `2_20` subset (not merely `null` elsewhere — the keys are absent entirely from the other 6 subsets). `backend` is the hls4ml backend used (e.g. `"VivadoAccelerator"`); `backend_version` is that subset's equivalent of `vivado_version` (e.g. `"2019.1"`). A field extractor should check `vivado_version` first and fall back to `backend_version` only when `vivado_version` is absent. |
| `hls4ml_version` | string | The hls4ml version used to perform the model-to-HLS conversion for the sample (e.g. `"0.8.1"`, `"1.1.0"`) — a system constraint. Populated for every subset. |

## Regression targets (for the Performance Metrics element)

Extracted from `resource_report` (post-logic-synthesis) and `latency_report` (HLS
estimate), these are the 6 scalar values a submission predicts per sample:

| Target | Source field | Units |
|---|---|---|
| `LUT` | `resource_report` | count |
| `FF` | `resource_report` | count |
| `DSP` | `resource_report` | count |
| `BRAM` | `resource_report` | count |
| `cycles_max` | `latency_report` (HLS estimate) | clock cycles |
| `interval_max` | `latency_report` (HLS estimate) | clock cycles (initiation interval) |

`hls_resource_report` is **not** the ground truth. It is the C-synthesis estimate, and it
exists for more samples (94,430 vs 92,933 in the test set). The paper's original GNN and
Transformer checkpoints were trained on it; the reference checkpoints used here are the
versions retrained on `resource_report` (docs/VALIDATION.md §5). The benchmark can still
emit it (`python -m wa_hls4ml_bench.truth --gt hls_estimate`) for checking models
trained on it, but benchmark scores always use `resource_report`.

Samples with no `resource_report` (9,551 test, 1 exemplar) have no ground truth. They
are excluded from scoring, never imputed. A submission should still predict them: the
scorer reports coverage against the samples that do have truth.

All 6 are reported as **absolute counts**, not as a percentage of the target
device's total capacity — a submission must know `target_part` to interpret a
prediction as a utilization fraction.

## Model generation parameter ranges (context for `model_config`/`hls_config`)

From paper Section 2.1.1, for the randomly/grid-generated portion of the
synthetic dataset:

| Parameter | Range |
|---|---|
| Number of layers | 2-7 (fully-connected), 3-7 (convolutional) |
| Activation functions | linear (most 2-3 layer FC); ReLU/tanh/sigmoid (all others) |
| Features/neurons | 8-128, step 8 (2-3 layer FC); 32-128, 8-64 filters (conv) |
| Weight/bias bit precision | 2-16 bits, step 2 (2-3 layer FC); 4-16 bits, powers of 2 (3-7 layer FC/conv) |
| hls4ml target reuse factor | 1-4093 (FC); 8192-32795 (conv) |
| hls4ml implementation strategy | `resource` (most FC + all 3-7 layer conv); `latency` (some 3-7 layer FC) |
| hls4ml I/O type | `io_parallel` (all FC); `io_stream` (all conv) |

Weight/bias precision is implemented in HLS as `ap_fixed<X,1>`, X = specified
total bit width, 1 integer bit.

## Access and versioning

Hosted on HuggingFace (`fastmachinelearning/wa-hls4ml` for this
results/labels dataset; `fastmachinelearning/wa-hls4ml-projects` for the full
synthesis artifacts) and mirrored on the Fermilab American Science Cloud Data
Platform, under CC-BY-NC 4.0. Both are versioned datasets with dataset cards;
the generation pipeline that produced them (`wa-hls4ml-search/`) is itself
public and versioned (github.com/ben-hawks/wa-hls4ml-search), so the pipeline — not just its
output — is reproducible.
