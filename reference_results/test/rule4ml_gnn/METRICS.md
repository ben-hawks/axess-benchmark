# rule4ml_gnn on test (ground truth: post-synthesis resources, HLS latency)

Scored 92933 of 92933 ground-truth samples (0 without a prediction).

## all

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | 0.703 | 46.26 | 8.2 | 92933 |
| DSP | 0.184 | 103.28 | 542.1 | 92933 |
| FF | 0.511 | 16.00 | 24919.5 | 92933 |
| LUT | 0.552 | 16.52 | 37974.6 | 92933 |
| cycles_max | 0.789 | 12.93 | 305382.9 | 92933 |
| interval_max | 0.558 | 14.18 | 441539.9 | 92933 |

Mean R^2 over targets: 0.549

## conv1d

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | 0.533 | 40.27 | 7.7 | 1995 |
| DSP | -0.885 | 93.50 | 3.3 | 1995 |
| FF | 0.359 | 27.74 | 13778.5 | 1995 |
| LUT | 0.275 | 29.85 | 8768.3 | 1995 |
| cycles_max | 0.825 | 26.96 | 52489.6 | 1995 |
| interval_max | 0.358 | 60.53 | 100552.6 | 1995 |

Mean R^2 over targets: 0.244

## conv2d

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | 0.480 | 40.34 | 35.1 | 1865 |
| DSP | -0.263 | 100.14 | 7.3 | 1865 |
| FF | 0.412 | 33.43 | 62854.3 | 1865 |
| LUT | 0.126 | 36.71 | 22649.9 | 1865 |
| cycles_max | 0.691 | 36.47 | 2155022.9 | 1865 |
| interval_max | 0.354 | 68.54 | 3115111.3 | 1865 |

Mean R^2 over targets: 0.300

## dense

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | 0.731 | 46.52 | 6.6 | 89073 |
| DSP | 0.184 | 103.57 | 553.7 | 89073 |
| FF | 0.505 | 15.37 | 23683.8 | 89073 |
| LUT | 0.553 | 15.79 | 38627.7 | 89073 |
| cycles_max | 0.971 | 12.12 | 520.6 | 89073 |
| interval_max | 0.949 | 12.00 | 303.0 | 89073 |

Mean R^2 over targets: 0.649

*N/A: ground truth has zero variance for this target in this group (R^2 undefined).