# rule4ml_gnn on exemplar (post-synthesis ground truth)

Scored 886 of 886 ground-truth samples (0 without a prediction).

## all

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -11.822 | 116.72 | 8.8 | 886 |
| DSP | 0.416 | 112.98 | 621.7 | 886 |
| FF | -1.312 | 114.75 | 25756.4 | 886 |
| LUT | 0.430 | 81.88 | 20709.7 | 886 |
| cycles_max | 0.436 | 78.44 | 944.9 | 886 |
| interval_max | 0.393 | 90.01 | 396.6 | 886 |

Mean R^2 over targets: -1.910

## exemplar/Anomaly

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -3.915 | 95.62 | 7.3 | 133 |
| DSP | 0.330 | 122.06 | 477.5 | 133 |
| FF | -3.811 | 85.21 | 43541.8 | 133 |
| LUT | 0.210 | 68.06 | 17057.5 | 133 |
| cycles_max | 0.492 | 49.83 | 710.7 | 133 |
| interval_max | 0.436 | 98.34 | 394.7 | 133 |

Mean R^2 over targets: -1.043

## exemplar/Automlp

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -124.858 | 116.51 | 7.1 | 127 |
| DSP | -7.904 | 100.66 | 404.0 | 127 |
| FF | -26.073 | 135.23 | 14254.7 | 127 |
| LUT | -1.549 | 78.57 | 5185.1 | 127 |
| cycles_max | 0.245 | 83.19 | 171.6 | 127 |
| interval_max | -11.130 | 82.54 | 345.3 | 127 |

Mean R^2 over targets: -28.545

## exemplar/Bipc

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -17.821 | 135.84 | 11.3 | 119 |
| DSP | 0.308 | 91.49 | 1405.6 | 119 |
| FF | -1.345 | 100.07 | 40804.9 | 119 |
| LUT | 0.239 | 80.66 | 43262.4 | 119 |
| cycles_max | 0.547 | 63.95 | 1631.4 | 119 |
| interval_max | 0.480 | 75.18 | 546.2 | 119 |

Mean R^2 over targets: -2.932

## exemplar/Cookie

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -10.809 | 102.54 | 9.7 | 130 |
| DSP | 0.791 | 137.68 | 336.0 | 130 |
| FF | -0.354 | 114.91 | 16734.3 | 130 |
| LUT | 0.296 | 81.37 | 19281.9 | 130 |
| cycles_max | -1.769 | 42.03 | 1472.0 | 130 |
| interval_max | 0.487 | 80.25 | 356.9 | 130 |

Mean R^2 over targets: -1.893

## exemplar/Jet

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -30.270 | 100.68 | 8.4 | 124 |
| DSP | 0.767 | 134.44 | 322.4 | 124 |
| FF | -0.812 | 104.79 | 14192.7 | 124 |
| LUT | 0.112 | 83.91 | 16231.7 | 124 |
| cycles_max | 0.571 | 83.76 | 679.3 | 124 |
| interval_max | 0.411 | 82.53 | 424.4 | 124 |

Mean R^2 over targets: -4.870

## exemplar/Particle

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -44.808 | 109.93 | 9.3 | 127 |
| DSP | 0.813 | 120.79 | 273.6 | 127 |
| FF | -0.981 | 105.96 | 13941.6 | 127 |
| LUT | 0.127 | 82.96 | 14998.5 | 127 |
| cycles_max | 0.566 | 81.20 | 662.2 | 127 |
| interval_max | 0.355 | 85.19 | 434.1 | 127 |

Mean R^2 over targets: -7.321

## exemplar/Quarks

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | N/A* | 158.39 | 7.8 | 126 |
| DSP | -9.505 | 81.67 | 390.6 | 126 |
| FF | -55.203 | 157.62 | 14253.1 | 126 |
| LUT | -4.748 | 98.41 | 6661.3 | 126 |
| cycles_max | -12.438 | 147.12 | 263.4 | 126 |
| interval_max | -7.150 | 125.03 | 202.2 | 126 |

Mean R^2 over targets: -17.809

*N/A: ground truth has zero variance for this target in this group (R^2 undefined).