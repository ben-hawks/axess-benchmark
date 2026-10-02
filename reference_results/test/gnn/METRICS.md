# gnn on test (post-synthesis ground truth)

Scored 92933 of 92933 ground-truth samples (0 without a prediction).

## all

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | 0.640 | 21.57 | 9.1 | 92933 |
| DSP | 0.565 | 14.82 | 395.9 | 92933 |
| FF | 0.935 | 14.06 | 9102.0 | 92933 |
| LUT | 0.901 | 13.99 | 17847.8 | 92933 |
| cycles_max | 0.815 | 17.93 | 285934.4 | 92933 |
| interval_max | 0.827 | 14.45 | 276666.4 | 92933 |

Mean R^2 over targets: 0.780

## conv1d

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | 0.626 | 36.09 | 6.9 | 1995 |
| DSP | 0.799 | 18.68 | 1.1 | 1995 |
| FF | 0.919 | 9.94 | 4892.6 | 1995 |
| LUT | 0.915 | 9.77 | 3006.3 | 1995 |
| cycles_max | 0.893 | 16.33 | 41108.0 | 1995 |
| interval_max | 0.877 | 18.89 | 43960.7 | 1995 |

Mean R^2 over targets: 0.838

## conv2d

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | 0.530 | 30.53 | 33.4 | 1865 |
| DSP | 0.470 | 25.53 | 4.7 | 1865 |
| FF | 0.882 | 12.50 | 28213.5 | 1865 |
| LUT | 0.871 | 12.03 | 8697.1 | 1865 |
| cycles_max | 0.729 | 31.93 | 2017949.5 | 1865 |
| interval_max | 0.746 | 31.17 | 1952467.2 | 1865 |

Mean R^2 over targets: 0.705

## dense

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | 0.622 | 21.06 | 7.8 | 89073 |
| DSP | 0.565 | 14.50 | 404.4 | 89073 |
| FF | 0.939 | 14.19 | 8320.6 | 89073 |
| LUT | 0.901 | 14.13 | 18181.4 | 89073 |
| cycles_max | 0.775 | 17.67 | 1437.2 | 89073 |
| interval_max | 0.902 | 14.00 | 419.8 | 89073 |

Mean R^2 over targets: 0.784

*N/A: ground truth has zero variance for this target in this group (R^2 undefined).