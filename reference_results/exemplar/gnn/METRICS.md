# gnn on exemplar (post-synthesis ground truth)

Scored 886 of 886 ground-truth samples (0 without a prediction).

## all

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -106.748 | 92.02 | 25.4 | 886 |
| DSP | -7.301 | 67.52 | 2344.1 | 886 |
| FF | -3.921 | 95.06 | 37574.6 | 886 |
| LUT | -28.768 | 125.56 | 149642.9 | 886 |
| cycles_max | 0.405 | 75.20 | 970.8 | 886 |
| interval_max | 0.346 | 89.92 | 411.4 | 886 |

Mean R^2 over targets: -24.331

## exemplar/Anomaly

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -103.014 | 107.16 | 33.5 | 133 |
| DSP | -10.037 | 88.89 | 1938.3 | 133 |
| FF | -5.619 | 110.92 | 51071.8 | 133 |
| LUT | -237.856 | 154.59 | 296601.8 | 133 |
| cycles_max | 0.210 | 42.18 | 885.7 | 133 |
| interval_max | 0.307 | 95.42 | 437.6 | 133 |

Mean R^2 over targets: -59.335

## exemplar/Automlp

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -0.374 | 38.25 | 0.7 | 127 |
| DSP | 0.985 | 71.25 | 16.6 | 127 |
| FF | -0.175 | 82.29 | 2969.3 | 127 |
| LUT | -36.927 | 135.21 | 19998.7 | 127 |
| cycles_max | -0.258 | 88.58 | 221.5 | 127 |
| interval_max | 0.563 | 78.59 | 65.5 | 127 |

Mean R^2 over targets: -6.031

## exemplar/Bipc

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -2.350 | 71.71 | 4.8 | 119 |
| DSP | -11.291 | 51.15 | 5924.2 | 119 |
| FF | -8.953 | 104.35 | 84069.9 | 119 |
| LUT | -5.879 | 108.44 | 130090.7 | 119 |
| cycles_max | 0.360 | 71.07 | 1939.0 | 119 |
| interval_max | 0.429 | 64.61 | 572.2 | 119 |

Mean R^2 over targets: -4.614

## exemplar/Cookie

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -381.614 | 143.24 | 55.2 | 130 |
| DSP | -0.946 | 66.21 | 1025.4 | 130 |
| FF | -0.643 | 107.23 | 18436.7 | 130 |
| LUT | -84.870 | 145.49 | 212884.2 | 130 |
| cycles_max | 0.434 | 31.58 | 665.4 | 130 |
| interval_max | 0.197 | 108.39 | 446.7 | 130 |

Mean R^2 over targets: -77.907

## exemplar/Jet

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -41.517 | 82.03 | 9.7 | 124 |
| DSP | 0.505 | 53.64 | 470.1 | 124 |
| FF | 0.407 | 76.06 | 8117.1 | 124 |
| LUT | -1.397 | 100.62 | 26672.3 | 124 |
| cycles_max | 0.433 | 73.61 | 780.9 | 124 |
| interval_max | 0.363 | 80.49 | 441.5 | 124 |

Mean R^2 over targets: -6.868

## exemplar/Particle

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -38.429 | 85.93 | 8.7 | 127 |
| DSP | 0.492 | 71.29 | 451.0 | 127 |
| FF | 0.404 | 76.02 | 7646.1 | 127 |
| LUT | -1.472 | 100.33 | 25242.8 | 127 |
| cycles_max | 0.396 | 75.36 | 781.0 | 127 |
| interval_max | 0.358 | 82.32 | 433.1 | 127 |

Mean R^2 over targets: -6.375

## exemplar/Quarks

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | N/A* | 112.50 | 4.7 | 126 |
| DSP | -0.294 | 67.90 | 137.1 | 126 |
| FF | -3.454 | 107.76 | 4012.6 | 126 |
| LUT | -13.852 | 130.79 | 10707.6 | 126 |
| cycles_max | -101.075 | 146.88 | 726.0 | 126 |
| interval_max | -16.502 | 117.33 | 296.3 | 126 |

Mean R^2 over targets: -27.036

*N/A: ground truth has zero variance for this target in this group (R^2 undefined).