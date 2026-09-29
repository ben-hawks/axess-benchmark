# transformer on test (post-synthesis ground truth)

Scored 92933 of 92933 ground-truth samples (0 without a prediction).

## all

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -7.083 | 116.74 | 43.0 | 92933 |
| DSP | -2.185 | 13.34 | 1071.3 | 92933 |
| FF | -2.883 | 10.61 | 70187.6 | 92933 |
| LUT | -10.568 | 19.53 | 192985.6 | 92933 |
| cycles_max | 0.929 | 10.30 | 177285.0 | 92933 |
| interval_max | 0.908 | 14.21 | 201523.7 | 92933 |

Mean R^2 over targets: -3.481

## conv1d

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -8.579 | 99.18 | 35.1 | 1995 |
| DSP | -0.825 | 85.18 | 3.2 | 1995 |
| FF | 0.090 | 31.88 | 16417.8 | 1995 |
| LUT | -5.804 | 73.51 | 26871.0 | 1995 |
| cycles_max | 0.948 | 11.57 | 28558.7 | 1995 |
| interval_max | 0.915 | 13.80 | 36688.6 | 1995 |

Mean R^2 over targets: -2.209

## conv2d

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | 0.169 | 71.51 | 44.4 | 1865 |
| DSP | -0.224 | 85.27 | 7.2 | 1865 |
| FF | 0.757 | 34.19 | 40441.8 | 1865 |
| LUT | -8.012 | 86.28 | 72721.7 | 1865 |
| cycles_max | 0.896 | 22.59 | 1251104.8 | 1865 |
| interval_max | 0.865 | 24.39 | 1422054.8 | 1865 |

Mean R^2 over targets: -0.925

## dense

| Target | R^2 | SMAPE [%] | RMSE | n |
|---|---|---|---|---|
| BRAM | -10.440 | 118.08 | 43.1 | 89073 |
| DSP | -2.186 | 10.23 | 1094.3 | 89073 |
| FF | -3.502 | 9.64 | 71410.8 | 89073 |
| LUT | -10.597 | 16.92 | 196800.6 | 89073 |
| cycles_max | 0.951 | 10.02 | 670.2 | 89073 |
| interval_max | 0.910 | 14.01 | 402.6 | 89073 |

Mean R^2 over targets: -4.144

*N/A: ground truth has zero variance for this target in this group (R^2 undefined).