#!/bin/bash
# rule4ml models (CPU): the baseline MLP reference solution (TensorFlow) and rule4ml's
# bundled GIN-based GNN, which is run only as an auxiliary comparison.
#SBATCH --job-name=wa-infer-rule4ml
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
source "${WA_REPO:-$SLURM_SUBMIT_DIR}/perlmutter/env.sh"
wa_activate_mlp

for split in $WA_SPLITS; do
    for model in mlp rule4ml_gnn; do
        python -m wa_hls4ml_bench.predict --model "$model" --data-root "$WA_DATA" --split "$split" \
            --out "$WA_RESULTS/$split/predictions_$model.csv"
    done
done
