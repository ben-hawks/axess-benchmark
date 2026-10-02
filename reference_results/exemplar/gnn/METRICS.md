# gnn on exemplar (post-synthesis ground truth)

Scored 886 of 886 ground-truth samples (0 without a prediction).

## all

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -0.213 | 57.16 | 2.7 | 886 |
| DSP | 0.038 | 84.06 | 798.1 | 886 |
| FF | 0.522 | 85.61 | 11709.6 | 886 |
| LUT | 0.563 | 56.95 | 18127.5 | 886 |
| cycles_max | -6.863 | 88.00 | 3529.0 | 886 |
| interval_max | -5.785 | 116.26 | 1325.4 | 886 |

Mean R^2 over targets: -1.956

## exemplar/Anomaly

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -0.240 | 63.40 | 3.7 | 133 |
| DSP | 0.048 | 86.21 | 569.4 | 133 |
| FF | 0.648 | 53.21 | 11775.5 | 133 |
| LUT | 0.532 | 48.25 | 13122.2 | 133 |
| cycles_max | 0.515 | 45.34 | 694.0 | 133 |
| interval_max | 0.311 | 120.70 | 436.5 | 133 |

Mean R^2 over targets: 0.302

## exemplar/Automlp

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -1.231 | 47.61 | 0.9 | 127 |
| DSP | 0.191 | 94.86 | 121.7 | 127 |
| FF | 0.542 | 70.71 | 1853.1 | 127 |
| LUT | 0.719 | 46.05 | 1722.7 | 127 |
| cycles_max | 0.185 | 97.13 | 178.3 | 127 |
| interval_max | 0.560 | 102.90 | 65.8 | 127 |

Mean R^2 over targets: 0.161

## exemplar/Bipc

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -2.094 | 86.61 | 4.6 | 119 |
| DSP | -0.074 | 86.03 | 1751.2 | 119 |
| FF | 0.131 | 96.21 | 24844.4 | 119 |
| LUT | 0.278 | 62.76 | 42152.0 | 119 |
| cycles_max | -14.346 | 108.28 | 9496.6 | 119 |
| interval_max | -20.093 | 122.94 | 3478.6 | 119 |

Mean R^2 over targets: -6.033

## exemplar/Cookie

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -0.227 | 40.22 | 3.1 | 130 |
| DSP | 0.050 | 72.61 | 716.3 | 130 |
| FF | 0.575 | 90.59 | 9381.1 | 130 |
| LUT | 0.649 | 47.61 | 13605.9 | 130 |
| cycles_max | 0.419 | 36.57 | 674.5 | 130 |
| interval_max | 0.099 | 128.39 | 473.1 | 130 |

Mean R^2 over targets: 0.261

## exemplar/Jet

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -0.474 | 55.53 | 1.8 | 124 |
| DSP | 0.199 | 83.01 | 598.1 | 124 |
| FF | 0.575 | 81.10 | 6873.6 | 124 |
| LUT | 0.612 | 52.19 | 10726.3 | 124 |
| cycles_max | 0.553 | 95.90 | 693.0 | 124 |
| interval_max | 0.258 | 109.43 | 476.4 | 124 |

Mean R^2 over targets: 0.287

## exemplar/Particle

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -0.357 | 50.36 | 1.6 | 127 |
| DSP | 0.192 | 88.35 | 568.9 | 127 |
| FF | 0.546 | 81.96 | 6674.2 | 127 |
| LUT | 0.621 | 52.35 | 9885.5 | 127 |
| cycles_max | 0.529 | 98.04 | 689.3 | 127 |
| interval_max | 0.258 | 110.37 | 465.7 | 127 |

Mean R^2 over targets: 0.298

## exemplar/Quarks

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | N/A* | 58.30 | 0.7 | 126 |
| DSP | -0.176 | 77.58 | 130.7 | 126 |
| FF | -12.606 | 127.77 | 7012.9 | 126 |
| LUT | -4.845 | 90.61 | 6717.2 | 126 |
| cycles_max | -83.733 | 139.84 | 661.4 | 126 |
| interval_max | -8.217 | 118.90 | 215.0 | 126 |

Mean R^2 over targets: -21.915

*N/A: ground truth has zero variance for this target in this group (R^2 undefined).