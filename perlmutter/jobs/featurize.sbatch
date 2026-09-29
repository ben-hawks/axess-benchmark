#!/bin/bash
# Featurize the benchmark splits into $WA_CACHE (CPU, embarrassingly parallel).
#SBATCH --job-name=wa-featurize
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --cpus-per-task=64
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
set -euo pipefail
source "${WA_REPO:-$SLURM_SUBMIT_DIR}/perlmutter/env.sh"
wa_activate_torch
mkdir -p "$WA_CACHE"

for split in $WA_SPLITS; do
    python -m wa_hls4ml_bench.cache --data-root "$WA_DATA" --split "$split" \
        --out "$WA_CACHE/$split.npz" --workers "$SLURM_CPUS_PER_TASK"
done
