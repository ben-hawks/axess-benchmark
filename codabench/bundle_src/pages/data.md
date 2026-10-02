# Data

The dataset is public on HuggingFace:
[fastmachinelearning/wa-hls4ml](https://huggingface.co/datasets/fastmachinelearning/wa-hls4ml)
(CC-BY-NC 4.0). Full Vivado/Vitis projects for every sample are in
[wa-hls4ml-projects](https://huggingface.co/datasets/fastmachinelearning/wa-hls4ml-projects).

| Split | Samples | Scored (have post-synthesis results) | Use |
|---|---|---|---|
| `train/` | 478,216 | — | training |
| `val/` | 102,472 | — | validation |
| `test/` | 102,484 | 92,933 | **scored** |
| `exemplar/` | 887 | 886 | **scored** (generalization) |

Each split is a set of JSON files, one array of samples per generation subset
(`2_20`, `2layer`, `3layer`, `latency`, `resource`, `conv1d`, `conv2d`). The exemplar
set is a single file covering 7 real architectures.

## Sample fields

| Field | Role |
|---|---|
| `meta_data.uuid` (`meta_data.model_id` for the `2_20` subset) | the sample's `sample_id` |
| `model_config` | **input**: flat list of per-layer dicts (type, shapes, parameters, reuse factor, …) |
| `hls_config` | **input**: hls4ml configuration (precision, reuse factor, strategy, I/O type) |
| `target_part`, `vivado_version`/`backend_version`, `hls4ml_version` | **input**: fixed constraints (FPGA part, toolchain versions) |
| `resource_report` | **ground truth** for BRAM, DSP, FF, LUT (post-logic-synthesis) |
| `latency_report` | **ground truth** for `cycles_max`, `interval_max` |
| `hls_resource_report` | HLS C-synthesis estimate. **Not** ground truth, and not a valid model input |

Predictions must come from `model_config`, `hls_config` and the constraint fields only.
A model must not read any `*_report` field at prediction time. The test and exemplar
ground truth is part of the public dataset, so this rests on participants' honesty:
describe your inputs in your submission's method description.

Samples without a `resource_report` have no ground truth and aren't scored. You can
include predictions for them; they're ignored.

## Download

```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('fastmachinelearning/wa-hls4ml', repo_type='dataset', local_dir='wa-hls4ml')"
```

or use `scripts/fetch_data.py` from
[the benchmark repository](https://github.com/ben-hawks/axess-benchmark), which also
records the dataset revision. The starting kit lists the exact `sample_id`s that are
scored in each split.
