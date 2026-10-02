# AXESS wa-hls4ml Benchmark

hls4ml compiles neural networks into FPGA IP for latency- and power-constrained
scientific edge systems, such as real-time trigger systems at the LHC. Choosing a design
means knowing how many FPGA resources it uses and how fast it runs. Getting those
numbers from HLS and logic synthesis takes minutes to hours per design, which is far too
slow for a codesign loop that explores many candidates.

**The task:** given a Keras/QKeras network (`model_config`) and its hls4ml conversion
configuration (`hls_config`), predict what logic synthesis reports:

| Target | Meaning |
|---|---|
| `BRAM`, `DSP`, `FF`, `LUT` | post-logic-synthesis resource counts (BRAM can be fractional: a BRAM18 counts as 0.5) |
| `cycles_max` | latency, in clock cycles |
| `interval_max` | initiation interval, in clock cycles |

A good surrogate model answers in milliseconds instead of hours, without running any
part of the synthesis flow.

**How it works:** run your model on the test and exemplar splits of the wa-hls4ml
dataset (on your own hardware), and upload the predictions. Each submission is scored on
both splits:

- the **test set**: 92,933 synthetic networks (dense, Conv1D, Conv2D) drawn from the
  same generation process as the training data;
- the **exemplar set**: 886 real scientific architectures (jet tagging, anomaly
  detection, particle tracking, and others) that test generalization beyond the
  training distribution.

See **Data** for what to download and **Evaluation** for the submission format and
metrics.

## Reference solutions

| Model | Test mean R² | Exemplar mean R² |
|---|---|---|
| Transformer (retrained on post-synthesis labels) | 0.81 | −0.52 |
| GNN, GATv2 (retrained on post-synthesis labels) | 0.78 | −1.96 |
| Baseline MLP (rule4ml) | 0.32 | 0.25 |

Pretrained weights, inference code, and NERSC Perlmutter workflows for all three are in
[github.com/ben-hawks/axess-benchmark](https://github.com/ben-hawks/axess-benchmark). No
model generalizes reliably to the exemplar set yet; that's the open problem.

## Citation

B. Hawks *et al.*, "wa-hls4ml: A Benchmark and Surrogate Models for hls4ml Resource
and Latency Estimation," *ACM Transactions on Reconfigurable Technology and Systems*
19(2), 2026. [doi:10.1145/3787490](https://doi.org/10.1145/3787490)

Contact: bhawks@fnal.gov
