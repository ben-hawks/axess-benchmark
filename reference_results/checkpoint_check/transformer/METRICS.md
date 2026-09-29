# predictions_transformer.csv vs truth_hls_estimate.csv

Scored 94430 of 94430 ground-truth samples (0 without a prediction).

## all

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | 0.434 | 13.16 | 51.9 | 94430 |
| DSP | -1.337 | 9.10 | 2662.1 | 94430 |
| FF | 0.700 | 3.27 | 58857.8 | 94430 |
| LUT | 0.612 | 2.97 | 125897.3 | 94430 |
| cycles_max | 0.929 | 10.45 | 183396.7 | 94430 |
| interval_max | 0.909 | 14.39 | 207614.7 | 94430 |

Mean R^2 over targets: 0.374

## conv1d

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | 0.796 | 29.08 | 13.7 | 2510 |
| DSP | 0.495 | 17.30 | 1.2 | 2510 |
| FF | 0.954 | 8.39 | 5062.2 | 2510 |
| LUT | 0.956 | 6.68 | 3801.8 | 2510 |
| cycles_max | 0.948 | 11.63 | 29770.3 | 2510 |
| interval_max | 0.912 | 13.89 | 38665.5 | 2510 |

Mean R^2 over targets: 0.843

## conv2d

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | 0.815 | 18.05 | 15.7 | 2043 |
| DSP | 0.543 | 21.45 | 5.4 | 2043 |
| FF | 0.866 | 8.92 | 30288.3 | 2043 |
| LUT | 0.952 | 7.14 | 14805.1 | 2043 |
| cycles_max | 0.897 | 22.61 | 1246399.8 | 2043 |
| interval_max | 0.868 | 24.38 | 1410840.1 | 2043 |

Mean R^2 over targets: 0.823

## dense

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | 0.426 | 12.60 | 53.0 | 89877 |
| DSP | -1.337 | 8.59 | 2728.7 | 89877 |
| FF | 0.696 | 3.00 | 60151.1 | 89877 |
| LUT | 0.611 | 2.77 | 129025.9 | 89877 |
| cycles_max | 0.952 | 10.14 | 667.5 | 89877 |
| interval_max | 0.912 | 14.18 | 400.8 | 89877 |

Mean R^2 over targets: 0.376

*N/A: ground truth has zero variance for this target in this group (R^2 undefined).