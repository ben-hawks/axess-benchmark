#!/bin/bash
# Featurize each split once into $WA_CACHE/<split>.npz. Resources: WA_SB_FEATURIZE in the
# machine profile (passed by submit.sh).
#SBATCH --job-name=wa-featurize
#SBATCH --output=%x-%j.out
set -euo pipefail
# Slurm runs a spooled copy of this script, so locate env.sh through the exported repo path.
source "${WA_REPO:-$SLURM_SUBMIT_DIR}/${WA_HPC:-slurm}/env.sh"
wa_activate
mkdir -p "$WA_CACHE"

for split in $WA_SPLITS; do
    python -m wa_hls4ml_bench.cache --data-root "$WA_DATA" --split "$split" \
        --cache-dir "$WA_CACHE" --workers "${SLURM_CPUS_PER_TASK:-1}"
done
