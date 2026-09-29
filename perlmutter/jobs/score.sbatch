#!/bin/bash
# Ground truth + metrics for every predictions_*.csv, then the leaderboard.
#SBATCH --job-name=wa-score
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
set -euo pipefail
source "${WA_REPO:-$SLURM_SUBMIT_DIR}/perlmutter/env.sh"
wa_activate_torch

bash "$WA_REPO/scripts/score_all.sh"
