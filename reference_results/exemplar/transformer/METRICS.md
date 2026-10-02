# transformer on exemplar (post-synthesis ground truth)

Scored 886 of 886 ground-truth samples (0 without a prediction).

## all

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -0.680 | 59.32 | 3.2 | 886 |
| DSP | 0.578 | 107.35 | 528.8 | 886 |
| FF | -2.984 | 81.50 | 33811.9 | 886 |
| LUT | -0.573 | 55.44 | 34400.6 | 886 |
| cycles_max | 0.405 | 84.24 | 970.4 | 886 |
| interval_max | 0.154 | 112.26 | 467.9 | 886 |

Mean R^2 over targets: -0.517

## exemplar/Anomaly

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -1.504 | 60.89 | 5.2 | 133 |
| DSP | 0.181 | 81.98 | 528.0 | 133 |
| FF | 0.369 | 61.28 | 15766.0 | 133 |
| LUT | -0.342 | 59.54 | 22229.7 | 133 |
| cycles_max | 0.438 | 52.76 | 747.3 | 133 |
| interval_max | 0.011 | 128.98 | 522.9 | 133 |

Mean R^2 over targets: -0.141

## exemplar/Automlp

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -0.329 | 43.56 | 0.7 | 127 |
| DSP | -62.217 | 154.50 | 1076.4 | 127 |
| FF | 0.668 | 71.50 | 1578.6 | 127 |
| LUT | 0.626 | 48.46 | 1987.2 | 127 |
| cycles_max | 0.103 | 87.19 | 187.1 | 127 |
| interval_max | -0.083 | 103.92 | 103.2 | 127 |

Mean R^2 over targets: -10.206

## exemplar/Bipc

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -2.016 | 69.68 | 4.5 | 119 |
| DSP | 0.908 | 80.02 | 512.3 | 119 |
| FF | -10.323 | 113.02 | 89669.1 | 119 |
| LUT | -2.187 | 73.79 | 88544.0 | 119 |
| cycles_max | 0.333 | 93.76 | 1980.2 | 119 |
| interval_max | 0.418 | 83.37 | 577.9 | 119 |

Mean R^2 over targets: -2.144

## exemplar/Cookie

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -0.360 | 49.53 | 3.3 | 130 |
| DSP | 0.648 | 131.10 | 435.9 | 130 |
| FF | 0.613 | 90.94 | 8951.0 | 130 |
| LUT | 0.682 | 41.65 | 12963.4 | 130 |
| cycles_max | 0.220 | 40.14 | 781.4 | 130 |
| interval_max | -0.145 | 122.77 | 533.3 | 130 |

Mean R^2 over targets: 0.276

## exemplar/Jet

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -0.114 | 54.69 | 1.6 | 124 |
| DSP | 0.941 | 123.92 | 162.4 | 124 |
| FF | 0.548 | 76.20 | 7086.5 | 124 |
| LUT | 0.607 | 49.65 | 10796.0 | 124 |
| cycles_max | 0.221 | 92.54 | 915.2 | 124 |
| interval_max | -0.047 | 114.74 | 566.0 | 124 |

Mean R^2 over targets: 0.359

## exemplar/Particle

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -0.183 | 50.37 | 1.5 | 127 |
| DSP | 0.961 | 128.45 | 125.3 | 127 |
| FF | 0.526 | 77.25 | 6815.7 | 127 |
| LUT | 0.619 | 49.69 | 9908.3 | 127 |
| cycles_max | 0.193 | 98.17 | 902.4 | 127 |
| interval_max | -0.062 | 115.62 | 557.2 | 127 |

Mean R^2 over targets: 0.342

## exemplar/Quarks

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | N/A* | 87.46 | 2.6 | 126 |
| DSP | -0.268 | 50.32 | 135.7 | 126 |
| FF | 0.629 | 82.94 | 1158.0 | 126 |
| LUT | 0.450 | 66.51 | 2059.6 | 126 |
| cycles_max | -2.601 | 128.80 | 136.3 | 126 |
| interval_max | 0.335 | 113.66 | 57.7 | 126 |

Mean R^2 over targets: -0.291

*N/A: ground truth has zero variance for this target in this group (R^2 undefined).