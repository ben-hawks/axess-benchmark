# transformer on test (ground truth: post-synthesis resources, HLS latency)

Scored 92933 of 92933 ground-truth samples (0 without a prediction).

## all

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | 0.349 | 18.88 | 12.2 | 92933 |
| DSP | 0.827 | 8.81 | 249.4 | 92933 |
| FF | 0.925 | 3.55 | 9734.2 | 92933 |
| LUT | 0.904 | 4.47 | 17578.3 | 92933 |
| cycles_max | 0.930 | 11.27 | 175633.0 | 92933 |
| interval_max | 0.918 | 14.74 | 190543.9 | 92933 |

Mean R^2 over targets: 0.809

## conv1d

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | 0.810 | 27.70 | 4.9 | 1995 |
| DSP | 0.798 | 17.64 | 1.1 | 1995 |
| FF | 0.952 | 7.65 | 3767.6 | 1995 |
| LUT | 0.918 | 10.54 | 2943.4 | 1995 |
| cycles_max | 0.938 | 13.73 | 31329.9 | 1995 |
| interval_max | 0.928 | 14.51 | 33739.2 | 1995 |

Mean R^2 over targets: 0.891

## conv2d

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | 0.693 | 25.54 | 27.0 | 1865 |
| DSP | 0.738 | 20.76 | 3.3 | 1865 |
| FF | 0.852 | 8.27 | 31590.0 | 1865 |
| LUT | 0.901 | 10.10 | 7618.3 | 1865 |
| cycles_max | 0.898 | 23.16 | 1239366.2 | 1865 |
| interval_max | 0.880 | 23.78 | 1344601.1 | 1865 |

Mean R^2 over targets: 0.827

## dense

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | 0.142 | 18.54 | 11.8 | 89073 |
| DSP | 0.827 | 8.37 | 254.8 | 89073 |
| FF | 0.931 | 3.36 | 8811.8 | 89073 |
| LUT | 0.904 | 4.21 | 17915.9 | 89073 |
| cycles_max | 0.942 | 10.97 | 728.8 | 89073 |
| interval_max | 0.906 | 14.55 | 412.1 | 89073 |

Mean R^2 over targets: 0.775

*N/A: ground truth has zero variance for this target in this group (R^2 undefined).