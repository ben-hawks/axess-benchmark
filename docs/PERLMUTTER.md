# Running the benchmark on NERSC Perlmutter

Everything below runs from a clone of this repo on a Perlmutter login node. Nothing
here trains a model: the benchmark scores the published pretrained checkpoints.

## 0. Where things go

`perlmutter/env.sh` sets every path; override any of them by exporting it first.

| Variable | Default | Contents |
|---|---|---|
| `WA_ROOT` | `$SCRATCH/wa-hls4ml` | everything below |
| `WA_DATA` | `$WA_ROOT/data/wa-hls4ml` | HF dataset snapshot (test + exemplar, ~0.7 GB) |
| `WA_WEIGHTS` | `$WA_ROOT/weights` | 2 checkpoints (~230 MB, from the wa_hls4ml_models GitHub release) + `normalization_stats.json` |
| `WA_CACHE` | `$WA_ROOT/cache` | featurized splits (`test.npz` ~95 MB) |
| `WA_RESULTS` | `$WA_ROOT/results` | predictions, per-model `METRICS.md`, `LEADERBOARD.md` |
| `WA_VENV`, `WA_VENV_MLP` | `$WA_ROOT/venv-torch`, `$WA_ROOT/venv-mlp` | Python envs |
| `WA_PYTORCH_MODULE` | `pytorch/2.6.0` | NERSC module the torch venv is built on |
| `WA_SPLITS` | `test exemplar` | splits to run |

`$SCRATCH` is purged for files not accessed in 8 weeks. To keep results, copy
`$WA_RESULTS` to CFS (`/global/cfs/cdirs/<project>/...`) or set `WA_ROOT` there.

## 1. One-time setup (login node, ~10 min)

```bash
module avail pytorch            # pick a module; export WA_PYTORCH_MODULE=... if not 2.6.0
bash perlmutter/setup.sh        # venvs + dataset + weights
```

`setup.sh` builds two venvs:

- **`venv-torch`**: `module load pytorch` plus `--system-site-packages`, then
  `requirements.txt` (torch_geometric, pandas, ijson, ...). It reuses NERSC's CUDA build
  of torch and is used for the GNN, Transformer, featurization, and scoring.
- **`venv-mlp`**: plain `module load python`, then `rule4ml==0.2.0`, used for the baseline MLP
  (TensorFlow) and the auxiliary rule4ml GNN (CPU torch). It's separate so TensorFlow's and
  rule4ml's own torch dependency can't shadow the module's torch. `triton` is removed from it
  (it can segfault on import on CPU nodes, and neither rule4ml model uses it).

Check the install before submitting anything (about a minute, CPU is fine):

```bash
source perlmutter/env.sh && wa_activate_torch
pip install pytest
WA_WEIGHTS=$WA_WEIGHTS python -m pytest tests -q
```

The golden-prediction tests check that the checkpoints, normalization stats, and
feature pipeline reproduce the validated reference outputs on 40+ real samples.

## 2. Run

```bash
bash perlmutter/submit.sh -A <nersc_project>          # e.g. -A m1234
squeue --me
```

This submits a dependency chain:

| Job | Resources | Wall time (expected) | What |
|---|---|---|---|
| `featurize.sbatch` | CPU shared, 64 cores | a few minutes | JSON -> `$WA_CACHE/<split>.npz` |
| `infer_gpu.sbatch` | 1 A100 (shared GPU) | a few minutes | GNN + Transformer predictions |
| `infer_mlp.sbatch` | CPU shared, 32 cores | ~30-60 min | rule4ml baseline MLP + auxiliary rule4ml GNN predictions (runs in parallel with the above) |
| `score.sbatch` | CPU shared, 4 cores | < 5 min | truth CSVs, per-model metrics, `LEADERBOARD.md` |

Extra `sbatch` flags pass through, e.g. `bash perlmutter/submit.sh -A m1234 -q debug`.
Add `--no-mlp` to skip the rule4ml job.

## 3. Results

```
$WA_RESULTS/
  LEADERBOARD.md                  # one table per split, all models
  test/truth.csv                  # post-synthesis ground truth (92,933 rows)
  test/predictions_<model>.csv    # one row per sample (102,484)
  test/<model>/METRICS.md         # R^2 / SMAPE / RMSE: all, dense, conv1d, conv2d
  test/<model>/metrics.json
  test/<model>/rpe_boxplot.png
  exemplar/...                    # same, grouped by exemplar architecture
  codabench/<model>_submission.zip  # upload-ready Codabench submission per model
```

## Scoring your own model

Write `$WA_RESULTS/test/predictions_<yourname>.csv` (columns `sample_id`, `BRAM`, `DSP`,
`FF`, `LUT`, `cycles_max`, `interval_max`; ids are `meta_data.uuid`, or
`meta_data.model_id` for the `2_20` subset), do the same for `exemplar`, and then run:

```bash
source perlmutter/env.sh && wa_activate_torch && bash scripts/score_all.sh
```

This scores your model next to the reference models and also writes
`$WA_RESULTS/codabench/<yourname>_submission.zip`, ready to upload to the Codabench
competition (codabench/README.md). It holds exactly the scored samples of both
splits, with the same checks the Codabench scoring program applies. If your predictions
are incomplete or non-finite, no zip is written and the step reports why. Then fill in
`SUBMISSION.md`.

## Running interactively instead

```bash
salloc -A <project> -C gpu -q interactive -t 30 --gpus 1
source perlmutter/env.sh && wa_activate_torch
python -m wa_hls4ml_bench.cache --data-root $WA_DATA --split test --out $WA_CACHE/test.npz
python -m wa_hls4ml_bench.predict --model gnn --cache $WA_CACHE/test.npz --device cuda \
    --weights-dir $WA_WEIGHTS --out $WA_RESULTS/test/predictions_gnn.csv
bash scripts/score_all.sh
```

## Troubleshooting

- **`module load pytorch/2.6.0` fails**: pick an installed version from `module avail pytorch`,
  export `WA_PYTORCH_MODULE`, then rerun `bash perlmutter/setup.sh envs`.
- **torch_geometric import errors**: `requirements.txt` pins `torch_geometric==2.6.1`,
  which is pure Python and needs no compiled extensions (`torch_scatter` etc. are not used).
- **Downloads fail on a compute node**: run `setup.sh` on a login node. Jobs never download anything.
