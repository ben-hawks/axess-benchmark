# wa-hls4ml leaderboard

Ground truth: post-logic-synthesis `resource_report` + `latency_report`.
R^2 per target (higher is better) and SMAPE [%] (lower is better); the full per-group tables are in each `<split>/<model>/METRICS.md`.

## exemplar

| Model | coverage | BRAM R^2 | DSP R^2 | FF R^2 | LUT R^2 | cycles_max R^2 | interval_max R^2 | mean R^2 | BRAM SMAPE | DSP SMAPE | FF SMAPE | LUT SMAPE | cycles_max SMAPE | interval_max SMAPE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gnn | 886/886 | -106.748 | -7.301 | -3.921 | -28.768 | 0.405 | 0.346 | -24.331 | 92.0 | 67.5 | 95.1 | 125.6 | 75.2 | 89.9 |
| mlp | 886/886 | -0.415 | 0.112 | 0.493 | 0.324 | 0.513 | 0.463 | 0.248 | 84.6 | 89.3 | 62.9 | 71.9 | 82.7 | 86.1 |
| transformer | 886/886 | -713.269 | 0.385 | -5.155 | -21.893 | 0.483 | 0.243 | -123.201 | 107.8 | 85.9 | 93.1 | 126.9 | 91.0 | 115.3 |

## test

| Model | coverage | BRAM R^2 | DSP R^2 | FF R^2 | LUT R^2 | cycles_max R^2 | interval_max R^2 | mean R^2 | BRAM SMAPE | DSP SMAPE | FF SMAPE | LUT SMAPE | cycles_max SMAPE | interval_max SMAPE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gnn | 92933/92933 | -51.244 | -95.924 | -1.440 | -5.277 | 0.893 | 0.916 | -25.346 | 108.0 | 24.8 | 19.4 | 27.2 | 15.8 | 13.2 |
| mlp | 92933/92933 | 0.318 | 0.033 | 0.198 | 0.491 | 0.539 | 0.335 | 0.319 | 33.9 | 105.7 | 24.9 | 15.5 | 31.8 | 27.0 |
| transformer | 92933/92933 | -7.083 | -2.185 | -2.883 | -10.568 | 0.929 | 0.908 | -3.481 | 116.7 | 13.3 | 10.6 | 19.5 | 10.3 | 14.2 |
