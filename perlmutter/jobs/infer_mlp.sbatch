#!/bin/bash
# Baseline MLP (rule4ml, TensorFlow, CPU) inference.
#SBATCH --job-name=wa-infer-mlp
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
source "${WA_REPO:-$SLURM_SUBMIT_DIR}/perlmutter/env.sh"
wa_activate_mlp

for split in $WA_SPLITS; do
    python -m wa_hls4ml_bench.predict --model mlp --data-root "$WA_DATA" --split "$split" \
        --out "$WA_RESULTS/$split/predictions_mlp.csv"
done
