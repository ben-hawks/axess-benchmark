#!/bin/bash
# Truth, metrics for every predictions_*.csv, LEADERBOARD.md, TIMINGS.md, Codabench zips
# (scripts/score_all.sh). Resources: WA_SB_SCORE.
#SBATCH --job-name=wa-score
#SBATCH --output=%x-%j.out
set -euo pipefail
# Slurm runs a spooled copy of this script, so locate env.sh through the exported repo path.
source "${WA_REPO:-$SLURM_SUBMIT_DIR}/${WA_HPC:-slurm}/env.sh"
wa_activate

bash "$WA_REPO/scripts/score_all.sh"
