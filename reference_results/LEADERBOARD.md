# wa-hls4ml leaderboard

Ground truth: post-logic-synthesis `resource_report` (BRAM/DSP/FF/LUT) + HLS-estimate `latency_report` (cycles, II).
R^2 per target (higher is better) and SMAPE [%] (lower is better); the full per-group tables are in each `<split>/<model>/METRICS.md`.
Rows in *italics* are auxiliary comparison models, not reference solutions.

## exemplar

| Model | coverage | BRAM R^2 | DSP R^2 | FF R^2 | LUT R^2 | cycles_max R^2 | interval_max R^2 | mean R^2 | BRAM SMAPE | DSP SMAPE | FF SMAPE | LUT SMAPE | cycles_max SMAPE | interval_max SMAPE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gnn | 886/886 | -0.213 | 0.038 | 0.522 | 0.563 | -6.863 | -5.785 | -1.956 | 57.2 | 84.1 | 85.6 | 57.0 | 88.0 | 116.3 |
| mlp | 886/886 | -0.415 | 0.112 | 0.493 | 0.324 | 0.513 | 0.463 | 0.248 | 84.6 | 89.3 | 62.9 | 71.9 | 82.7 | 86.1 |
| *rule4ml_gnn (auxiliary)* | 886/886 | -11.822 | 0.416 | -1.312 | 0.430 | 0.436 | 0.393 | -1.910 | 116.7 | 113.0 | 114.7 | 81.9 | 78.4 | 90.0 |
| transformer | 886/886 | -0.680 | 0.578 | -2.984 | -0.573 | 0.405 | 0.154 | -0.517 | 59.3 | 107.3 | 81.5 | 55.4 | 84.2 | 112.3 |

## test

| Model | coverage | BRAM R^2 | DSP R^2 | FF R^2 | LUT R^2 | cycles_max R^2 | interval_max R^2 | mean R^2 | BRAM SMAPE | DSP SMAPE | FF SMAPE | LUT SMAPE | cycles_max SMAPE | interval_max SMAPE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gnn | 92933/92933 | 0.640 | 0.565 | 0.935 | 0.901 | 0.815 | 0.827 | 0.780 | 21.6 | 14.8 | 14.1 | 14.0 | 17.9 | 14.5 |
| mlp | 92933/92933 | 0.318 | 0.033 | 0.198 | 0.491 | 0.539 | 0.335 | 0.319 | 33.9 | 105.7 | 24.9 | 15.5 | 31.8 | 27.0 |
| *rule4ml_gnn (auxiliary)* | 92933/92933 | 0.703 | 0.184 | 0.511 | 0.552 | 0.789 | 0.558 | 0.549 | 46.3 | 103.3 | 16.0 | 16.5 | 12.9 | 14.2 |
| transformer | 92933/92933 | 0.349 | 0.827 | 0.925 | 0.904 | 0.930 | 0.918 | 0.809 | 18.9 | 8.8 | 3.6 | 4.5 | 11.3 | 14.7 |
