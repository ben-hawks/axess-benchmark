# gnn on test (post-synthesis ground truth)

Scored 92933 of 92933 ground-truth samples (0 without a prediction).

## all

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -51.244 | 108.05 | 109.2 | 92933 |
| DSP | -95.924 | 24.80 | 5909.6 | 92933 |
| FF | -1.440 | 19.38 | 55637.7 | 92933 |
| LUT | -5.277 | 27.17 | 142159.4 | 92933 |
| cycles_max | 0.893 | 15.77 | 217686.0 | 92933 |
| interval_max | 0.916 | 13.22 | 192032.4 | 92933 |

Mean R^2 over targets: -25.346

## conv1d

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -12.564 | 103.87 | 41.7 | 1995 |
| DSP | 0.165 | 70.91 | 2.2 | 1995 |
| FF | 0.488 | 26.08 | 12319.2 | 1995 |
| LUT | -4.620 | 70.37 | 24419.9 | 1995 |
| cycles_max | 0.973 | 10.90 | 20799.3 | 1995 |
| interval_max | 0.966 | 10.98 | 23288.8 | 1995 |

Mean R^2 over targets: -2.432

## conv2d

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -0.309 | 85.77 | 55.7 | 1865 |
| DSP | 0.182 | 76.79 | 5.9 | 1865 |
| FF | 0.886 | 27.14 | 27683.8 | 1865 |
| LUT | -4.897 | 81.69 | 58823.4 | 1865 |
| cycles_max | 0.843 | 19.51 | 1536476.5 | 1865 |
| interval_max | 0.878 | 18.47 | 1355347.2 | 1865 |

Mean R^2 over targets: -0.403

## dense

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -75.038 | 108.61 | 111.1 | 89073 |
| DSP | -95.957 | 22.68 | 6036.3 | 89073 |
| FF | -1.834 | 19.06 | 56659.1 | 89073 |
| LUT | -5.288 | 25.06 | 144911.2 | 89073 |
| cycles_max | 0.813 | 15.81 | 1310.5 | 89073 |
| interval_max | 0.903 | 13.16 | 417.0 | 89073 |

Mean R^2 over targets: -29.400

*N/A: ground truth has zero variance for this target in this group (R^2 undefined).