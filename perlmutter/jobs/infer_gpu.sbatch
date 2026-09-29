#!/bin/bash
# GNN + Transformer inference on one A100 (shared GPU node).
#SBATCH --job-name=wa-infer-gpu
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
set -euo pipefail
source "${WA_REPO:-$SLURM_SUBMIT_DIR}/perlmutter/env.sh"
wa_activate_torch
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

for split in $WA_SPLITS; do
    for model in gnn transformer; do
        python -m wa_hls4ml_bench.predict --model "$model" --cache "$WA_CACHE/$split.npz" \
            --weights-dir "$WA_WEIGHTS" --device cuda \
            --out "$WA_RESULTS/$split/predictions_$model.csv"
    done
done
