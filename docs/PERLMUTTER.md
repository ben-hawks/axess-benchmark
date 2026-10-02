# Running the benchmark on NERSC Perlmutter

Everything below runs from a clone of this repo on a Perlmutter login node. Nothing
here trains a model: the benchmark scores the published pretrained checkpoints.

The Slurm scripts in `slurm/` are generic; everything Perlmutter-specific is in
`slurm/profiles/perlmutter.sh`, and this benchmark's software-stack fixes are in
`slurm/stack.sh` (see "Troubleshooting"). On Perlmutter, `env.sh` picks the profile
automatically from `$NERSC_HOST`. For another cluster, copy `slurm/profiles/generic.sh` to
`profiles/<machine>.sh`, fill it in, and run every command with `WA_MACHINE=<machine>`.

## 0. Where things go

`slurm/env.sh` sets every path; override any of them by exporting it first.

| Variable | Default | Contents |
|---|---|---|
| `WA_ROOT` | `$SCRATCH/wa-hls4ml` | everything below |
| `WA_DATA` | `$WA_ROOT/data/wa-hls4ml` | HF dataset snapshot (test + exemplar, ~0.7 GB) + `REVISION` |
| `WA_WEIGHTS` | `$WA_ROOT/weights` | 2 checkpoints (~230 MB, from the wa_hls4ml_models GitHub release) + `normalization_stats.json` |
| `WA_CACHE` | `$WA_ROOT/cache` | featurized splits (`test.npz` ~95 MB) |
| `WA_RESULTS` | `$WA_ROOT/results` | predictions, per-model metrics, `LEADERBOARD.md`, `TIMINGS.md`, Codabench zips |
| `WA_VENV`, `WA_VENV_ALT` | `$WA_ROOT/venv-torch`, `$WA_ROOT/venv-mlp` | main and rule4ml Python envs |
| `WA_MAIN_MODULE`, `WA_ALT_MODULE` | `pytorch/2.6.0`, `python` | NERSC modules the two venvs are built on |
| `WA_SPLITS` | `test exemplar` | splits to run |

`$SCRATCH` is purged for files not accessed in 8 weeks. To keep results, copy
`$WA_RESULTS` to CFS (`/global/cfs/cdirs/<project>/...`) or set `WA_ROOT` there.

The paths and venv locations are the same as in the `perlmutter/` scripts this replaced
(before 2026-10-02), so an existing setup is reused as-is.

## 1. One-time setup (login node, ~10 min)

```bash
module avail pytorch            # pick a module; export WA_MAIN_MODULE=... if not 2.6.0
bash slurm/setup.sh             # venvs + dataset + weights
```

`setup.sh` builds two venvs, then runs an import check on each, so a broken environment
fails here rather than in a queued job:

- **main (`venv-torch`)**: `module load pytorch` plus `--system-site-packages`, then
  `requirements.txt` (torch_geometric, pandas, ijson, ...). It reuses NERSC's CUDA build
  of torch and is used for featurization, the GNN, the Transformer and scoring.
- **alt (`venv-mlp`)**: plain `module load python`, then `rule4ml==0.2.0`, used for the
  baseline MLP (TensorFlow) and the auxiliary rule4ml GNN (CPU torch). It's separate so
  TensorFlow's and rule4ml's own torch can't shadow the module's torch.

Check the install before submitting anything (about a minute, CPU is fine):

```bash
source slurm/env.sh && wa_activate
WA_WEIGHTS=$WA_WEIGHTS python -m pytest tests -q
```

The golden-prediction tests check that the checkpoints, normalization stats, and
feature pipeline reproduce the validated reference outputs on 40 real samples. On a GPU
node, add `WA_TEST_DEVICE=cuda` to also check GPU vs CPU agreement.

## 2. Run

```bash
bash slurm/submit.sh -A <nersc_project>          # e.g. -A m1234
squeue --me
```

This submits a dependency chain. Resources come from `slurm/profiles/perlmutter.sh`;
the times are from the 2026-10-02 run (Slurm jobs 59209565–68):

| Job | Resources | Measured | What |
|---|---|---|---|
| `wa-featurize` | CPU shared, 64 cores, 15 min limit | 21 s | JSON → `$WA_CACHE/<split>.npz` |
| `wa-infer-gpu` | 1 A100 (shared GPU), 15 min limit | ~1 min | GNN + Transformer predictions |
| `wa-infer-cpu` | CPU shared, 32 cores, 1 h limit | ~22 min | rule4ml baseline MLP + auxiliary rule4ml GNN (in parallel with the GPU job) |
| `wa-score` | CPU shared, 4 cores, 15 min limit | < 5 min | truth, metrics, `LEADERBOARD.md`, `TIMINGS.md`, Codabench zips |

Extra `sbatch` flags pass through, e.g. `bash slurm/submit.sh -A m1234 -q debug`.
`--no-gpu` or `--no-cpu` skips that inference job (`--no-mlp` still works as the old name
for `--no-cpu`).

**Logs.** Each job writes `<job-name>-<jobid>.out` (e.g. `wa-infer-gpu-59209566.out`) in
the directory `submit.sh` submitted from, which is the repo root. The jobs from before
2026-10-02 were named `wa-featurize`, `wa-infer-gpu`, `wa-infer-rule4ml` and `wa-score`.

## 3. Results

```
$WA_RESULTS/
  LEADERBOARD.md                  # one table per split, all models
  TIMINGS.md                      # per model and split: total and model-only time, device, hardware, Slurm job
  test/truth.csv                  # ground truth: post-synthesis resources, HLS latency (92,933 rows)
  test/predictions_<model>.csv    # one row per sample (102,484); a sample the model can't handle has empty outputs
  test/predictions_<model>.timing.json
  test/<model>/METRICS.md         # R^2 / SMAPE / RMSE: all, dense, conv1d, conv2d
  test/<model>/metrics.json
  test/<model>/rpe_boxplot.png
  exemplar/...                    # same, grouped by exemplar architecture
  codabench/<model>_submission.zip  # upload-ready Codabench submission per model
```

`TIMINGS.md` separates *total* time (loading weights and data, featurization, writing the
CSV) from *model* time (inside the model only, synchronized on the GPU). Copy it into
`reference_solution/README.md` when timings change.

## Scoring your own model

Write `$WA_RESULTS/test/predictions_<yourname>.csv` (columns `sample_id`, `BRAM`, `DSP`,
`FF`, `LUT`, `cycles_max`, `interval_max`; ids are `meta_data.uuid`, or
`meta_data.model_id` for the `2_20` subset), do the same for `exemplar`, and then run:

```bash
source slurm/env.sh && wa_activate && bash scripts/score_all.sh
```

That's the whole interface: your model plugs in by writing its prediction files, and
nothing else changes. This scores your model next to the reference models and also
writes `$WA_RESULTS/codabench/<yourname>_submission.zip`, ready to upload to the
Codabench competition (codabench/README.md). It holds exactly the scored samples of both
splits, with the same checks the Codabench scoring program applies. If your predictions
are incomplete or non-finite, no zip is written and the step reports why. Then fill in
`SUBMISSION.md`.

## Running interactively instead

```bash
salloc -A <project> -C gpu -q interactive -t 30 --gpus 1
source slurm/env.sh && wa_activate
python -m wa_hls4ml_bench.cache --data-root $WA_DATA --split test --cache-dir $WA_CACHE
python -m wa_hls4ml_bench.predict --model gnn --cache-dir $WA_CACHE --split test --device cuda \
    --weights-dir $WA_WEIGHTS --out $WA_RESULTS/test/predictions_gnn.csv
bash scripts/score_all.sh
```

## Troubleshooting

- **`module load pytorch/2.6.0` fails**: pick an installed version from `module avail pytorch`,
  export `WA_MAIN_MODULE`, then rerun `bash slurm/setup.sh envs`.
- **Segfault on import in a CPU-only job** (fixed in `slurm/stack.sh`): importing
  `torch_geometric` pulls in `torch._dynamo`, which imports `triton`, and on CPU-only
  nodes that import segfaulted before any model code ran. `stack.sh` sets
  `TORCHDYNAMO_DISABLE=1` for every job, and removes `triton` from the rule4ml venv after
  install. Nothing in this benchmark uses `torch.compile`.
- **TensorFlow/oneDNN log spam** in the rule4ml job: `stack.sh` sets `TF_CPP_MIN_LOG_LEVEL=3`.
- **torch_geometric import errors**: `requirements.txt` pins `torch_geometric==2.6.1`,
  which is pure Python and needs no compiled extensions (`torch_scatter` etc. are not used).
- **`$'\r': command not found`**: the scripts were checked out with Windows line endings.
  `.gitattributes` forces LF; re-clone, or run `git add --renormalize .`.
- **Downloads fail on a compute node**: run `setup.sh` on a login node. Jobs never download anything.
