# transformer on exemplar (post-synthesis ground truth)

Scored 886 of 886 ground-truth samples (0 without a prediction).

## all

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -713.269 | 107.79 | 65.4 | 886 |
| DSP | 0.385 | 85.87 | 638.3 | 886 |
| FF | -5.155 | 93.08 | 42025.4 | 886 |
| LUT | -21.893 | 126.90 | 131229.3 | 886 |
| cycles_max | 0.483 | 90.98 | 904.7 | 886 |
| interval_max | 0.243 | 115.29 | 442.9 | 886 |

Mean R^2 over targets: -123.201

## exemplar/Anomaly

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -247.119 | 163.69 | 51.7 | 133 |
| DSP | 0.548 | 68.90 | 392.4 | 133 |
| FF | -9.156 | 116.32 | 63261.0 | 133 |
| LUT | -167.857 | 163.16 | 249382.3 | 133 |
| cycles_max | 0.397 | 52.46 | 774.1 | 133 |
| interval_max | 0.061 | 119.55 | 509.5 | 133 |

Mean R^2 over targets: -70.521

## exemplar/Automlp

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -259.990 | 84.74 | 10.2 | 127 |
| DSP | 0.131 | 104.01 | 126.2 | 127 |
| FF | 0.070 | 74.48 | 2641.5 | 127 |
| LUT | -33.580 | 136.59 | 19096.0 | 127 |
| cycles_max | 0.316 | 98.43 | 163.4 | 127 |
| interval_max | 0.254 | 107.79 | 85.6 | 127 |

Mean R^2 over targets: -48.800

## exemplar/Bipc

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -3.572 | 73.97 | 5.5 | 119 |
| DSP | 0.363 | 71.34 | 1348.2 | 119 |
| FF | -10.637 | 114.21 | 90902.4 | 119 |
| LUT | -11.575 | 122.23 | 175879.3 | 119 |
| cycles_max | 0.418 | 96.22 | 1849.0 | 119 |
| interval_max | 0.412 | 79.10 | 580.9 | 119 |

Mean R^2 over targets: -4.098

## exemplar/Cookie

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -47.865 | 150.65 | 19.7 | 130 |
| DSP | 0.658 | 97.86 | 429.9 | 130 |
| FF | -0.287 | 106.81 | 16315.2 | 130 |
| LUT | -44.260 | 149.02 | 154554.1 | 130 |
| cycles_max | 0.411 | 38.43 | 678.8 | 130 |
| interval_max | 0.058 | 121.25 | 483.9 | 130 |

Mean R^2 over targets: -15.214

## exemplar/Jet

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -19.775 | 79.05 | 6.8 | 124 |
| DSP | 0.092 | 96.34 | 636.8 | 124 |
| FF | 0.492 | 76.85 | 7517.6 | 124 |
| LUT | -1.058 | 99.31 | 24710.6 | 124 |
| cycles_max | 0.350 | 102.20 | 836.1 | 124 |
| interval_max | 0.109 | 120.37 | 522.0 | 124 |

Mean R^2 over targets: -3.298

## exemplar/Particle

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -22.650 | 88.03 | 6.7 | 127 |
| DSP | 0.103 | 103.86 | 599.3 | 127 |
| FF | 0.478 | 77.33 | 7157.9 | 127 |
| LUT | -1.043 | 98.53 | 22946.1 | 127 |
| cycles_max | 0.343 | 106.28 | 814.3 | 127 |
| interval_max | 0.108 | 120.56 | 510.5 | 127 |

Mean R^2 over targets: -3.777

## exemplar/Quarks

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | N/A* | 107.95 | 163.2 | 126 |
| DSP | -0.196 | 58.43 | 131.8 | 126 |
| FF | 0.549 | 85.02 | 1276.2 | 126 |
| LUT | -10.477 | 116.21 | 9412.5 | 126 |
| cycles_max | -5.848 | 146.97 | 188.0 | 126 |
| interval_max | -0.313 | 136.05 | 81.1 | 126 |

Mean R^2 over targets: -3.257

*N/A: ground truth has zero variance for this target in this group (R^2 undefined).