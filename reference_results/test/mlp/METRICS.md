# mlp on test (post-synthesis ground truth)

Scored 92933 of 92933 ground-truth samples (0 without a prediction).

## all

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | 0.318 | 33.88 | 12.5 | 92933 |
| DSP | 0.033 | 105.66 | 590.4 | 92933 |
| FF | 0.198 | 24.90 | 31897.9 | 92933 |
| LUT | 0.491 | 15.48 | 40463.1 | 92933 |
| cycles_max | 0.539 | 31.80 | 450993.3 | 92933 |
| interval_max | 0.335 | 26.96 | 541624.8 | 92933 |

Mean R^2 over targets: 0.319

## conv1d

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | 0.386 | 44.73 | 8.9 | 1995 |
| DSP | -0.070 | 80.46 | 2.5 | 1995 |
| FF | 0.315 | 28.70 | 14249.3 | 1995 |
| LUT | 0.238 | 27.18 | 8994.6 | 1995 |
| cycles_max | 0.376 | 48.46 | 99128.5 | 1995 |
| interval_max | -0.039 | 74.52 | 127909.9 | 1995 |

Mean R^2 over targets: 0.201

## conv2d

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -0.501 | 51.45 | 59.6 | 1865 |
| DSP | 0.149 | 87.03 | 6.0 | 1865 |
| FF | 0.513 | 41.76 | 57190.7 | 1865 |
| LUT | 0.414 | 34.41 | 18547.8 | 1865 |
| cycles_max | 0.326 | 56.10 | 3181909.7 | 1865 |
| interval_max | 0.029 | 79.05 | 3821058.6 | 1865 |

Mean R^2 over targets: 0.155

## dense

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | 0.469 | 33.27 | 9.3 | 89073 |
| DSP | 0.032 | 106.61 | 603.0 | 89073 |
| FF | 0.127 | 24.46 | 31441.0 | 89073 |
| LUT | 0.491 | 14.82 | 41221.3 | 89073 |
| cycles_max | 0.739 | 30.92 | 1549.8 | 89073 |
| interval_max | 0.772 | 24.81 | 639.8 | 89073 |

Mean R^2 over targets: 0.438

*N/A: ground truth has zero variance for this target in this group (R^2 undefined).