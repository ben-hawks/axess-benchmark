#!/bin/bash
# GNN + Transformer inference on one GPU, every split. Resources: WA_SB_INFER_GPU; device
# name: WA_GPU_DEVICE (machine profile). Each run also writes predictions_<model>.timing.json.
#SBATCH --job-name=wa-infer-gpu
#SBATCH --output=%x-%j.out
set -euo pipefail
# Slurm runs a spooled copy of this script, so locate env.sh through the exported repo path.
source "${WA_REPO:-$SLURM_SUBMIT_DIR}/${WA_HPC:-slurm}/env.sh"
wa_activate
wa_gpu_info

for split in $WA_SPLITS; do
    for model in gnn transformer; do
        python -m wa_hls4ml_bench.predict --model "$model" --cache-dir "$WA_CACHE" --split "$split" \
            --weights-dir "$WA_WEIGHTS" --device "$WA_GPU_DEVICE" \
            --out "$WA_RESULTS/$split/predictions_$model.csv"
    done
done
