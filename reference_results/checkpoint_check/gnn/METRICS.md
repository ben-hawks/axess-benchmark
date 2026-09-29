# predictions_gnn.csv vs truth_hls_estimate.csv

Scored 94430 of 94430 ground-truth samples (0 without a prediction).

## all

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -0.490 | 23.50 | 84.1 | 94430 |
| DSP | -111.715 | 21.79 | 18488.3 | 94430 |
| FF | 0.737 | 11.66 | 55087.7 | 94430 |
| LUT | 0.730 | 11.39 | 104945.7 | 94430 |
| cycles_max | 0.890 | 15.76 | 227987.9 | 94430 |
| interval_max | 0.915 | 13.25 | 200878.1 | 94430 |

Mean R^2 over targets: -18.155

## conv1d

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | 0.689 | 33.01 | 16.9 | 2510 |
| DSP | 0.023 | 31.85 | 1.7 | 2510 |
| FF | 0.953 | 7.66 | 5137.6 | 2510 |
| LUT | 0.958 | 6.23 | 3731.4 | 2510 |
| cycles_max | 0.972 | 11.01 | 21648.4 | 2510 |
| interval_max | 0.966 | 11.10 | 24199.6 | 2510 |

Mean R^2 over targets: 0.760

## conv2d

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | 0.436 | 30.90 | 27.4 | 2043 |
| DSP | 0.514 | 30.97 | 5.5 | 2043 |
| FF | 0.915 | 8.80 | 24145.3 | 2043 |
| LUT | 0.949 | 6.75 | 15283.9 | 2043 |
| cycles_max | 0.840 | 19.54 | 1549793.1 | 2043 |
| interval_max | 0.876 | 18.47 | 1365427.9 | 2043 |

Mean R^2 over targets: 0.755

## dense

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -0.512 | 23.07 | 86.1 | 89877 |
| DSP | -111.745 | 21.30 | 18950.8 | 89877 |
| FF | 0.733 | 11.84 | 56341.8 | 89877 |
| LUT | 0.730 | 11.64 | 107544.5 | 89877 |
| cycles_max | 0.817 | 15.81 | 1304.9 | 89877 |
| interval_max | 0.906 | 13.19 | 415.1 | 89877 |

Mean R^2 over targets: -18.179

*N/A: ground truth has zero variance for this target in this group (R^2 undefined).