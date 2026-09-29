# mlp on exemplar (post-synthesis ground truth)

Scored 886 of 886 ground-truth samples (0 without a prediction).

## all

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -0.415 | 84.64 | 2.9 | 886 |
| DSP | 0.112 | 89.28 | 766.8 | 886 |
| FF | 0.493 | 62.95 | 12057.6 | 886 |
| LUT | 0.324 | 71.92 | 22546.1 | 886 |
| cycles_max | 0.513 | 82.68 | 878.0 | 886 |
| interval_max | 0.463 | 86.10 | 372.9 | 886 |

Mean R^2 over targets: 0.248

## exemplar/Anomaly

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -0.804 | 104.32 | 4.4 | 133 |
| DSP | 0.264 | 59.17 | 500.7 | 133 |
| FF | 0.586 | 36.89 | 12774.2 | 133 |
| LUT | 0.449 | 49.21 | 14251.0 | 133 |
| cycles_max | 0.415 | 51.60 | 762.1 | 133 |
| interval_max | 0.487 | 107.91 | 376.7 | 133 |

Mean R^2 over targets: 0.233

## exemplar/Automlp

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -1.090 | 56.41 | 0.9 | 127 |
| DSP | 0.405 | 72.13 | 104.4 | 127 |
| FF | 0.690 | 61.42 | 1524.8 | 127 |
| LUT | -0.220 | 68.91 | 3586.5 | 127 |
| cycles_max | -0.314 | 86.82 | 226.4 | 127 |
| interval_max | -1.705 | 85.95 | 163.1 | 127 |

Mean R^2 over targets: -0.372

## exemplar/Bipc

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -0.705 | 107.73 | 3.4 | 119 |
| DSP | -0.035 | 127.01 | 1719.2 | 119 |
| FF | 0.157 | 77.19 | 24472.0 | 119 |
| LUT | 0.029 | 82.30 | 48867.4 | 119 |
| cycles_max | 0.435 | 75.99 | 1822.4 | 119 |
| interval_max | 0.426 | 70.58 | 573.6 | 119 |

Mean R^2 over targets: 0.051

## exemplar/Cookie

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -0.542 | 64.44 | 3.5 | 130 |
| DSP | 0.199 | 91.34 | 657.9 | 130 |
| FF | 0.341 | 66.75 | 11675.1 | 130 |
| LUT | 0.131 | 67.90 | 21420.7 | 130 |
| cycles_max | 0.453 | 32.20 | 654.2 | 130 |
| interval_max | 0.532 | 72.27 | 341.0 | 130 |

Mean R^2 over targets: 0.186

## exemplar/Jet

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -1.220 | 64.97 | 2.2 | 124 |
| DSP | 0.273 | 85.96 | 569.8 | 124 |
| FF | 0.326 | 58.79 | 8654.0 | 124 |
| LUT | -0.118 | 74.09 | 18214.8 | 124 |
| cycles_max | 0.529 | 89.21 | 711.9 | 124 |
| interval_max | 0.493 | 66.39 | 393.6 | 124 |

Mean R^2 over targets: 0.047

## exemplar/Particle

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -1.411 | 65.61 | 2.1 | 127 |
| DSP | 0.283 | 75.21 | 536.1 | 127 |
| FF | 0.332 | 58.14 | 8093.3 | 127 |
| LUT | -0.081 | 71.18 | 16695.1 | 127 |
| cycles_max | 0.524 | 87.55 | 693.4 | 127 |
| interval_max | 0.500 | 61.61 | 382.0 | 127 |

Mean R^2 over targets: 0.024

## exemplar/Quarks

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | N/A* | 129.90 | 2.3 | 126 |
| DSP | 0.188 | 118.07 | 108.6 | 126 |
| FF | 0.513 | 83.54 | 1327.1 | 126 |
| LUT | -0.409 | 91.90 | 3298.1 | 126 |
| cycles_max | -36.679 | 158.39 | 441.1 | 126 |
| interval_max | -13.175 | 136.27 | 266.6 | 126 |

Mean R^2 over targets: -9.913

*N/A: ground truth has zero variance for this target in this group (R^2 undefined).