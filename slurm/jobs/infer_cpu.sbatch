#!/bin/bash
# CPU-only models, in the separate rule4ml venv: the baseline MLP (reference solution) and
# rule4ml's GIN GNN (auxiliary). Resources: WA_SB_INFER_CPU. Each run also writes
# predictions_<model>.timing.json.
#SBATCH --job-name=wa-infer-cpu
#SBATCH --output=%x-%j.out
set -euo pipefail
# Slurm runs a spooled copy of this script, so locate env.sh through the exported repo path.
source "${WA_REPO:-$SLURM_SUBMIT_DIR}/${WA_HPC:-slurm}/env.sh"
wa_activate_alt

for split in $WA_SPLITS; do
    for model in mlp rule4ml_gnn; do
        python -m wa_hls4ml_bench.predict --model "$model" --data-root "$WA_DATA" --split "$split" \
            --out "$WA_RESULTS/$split/predictions_$model.csv"
    done
done
