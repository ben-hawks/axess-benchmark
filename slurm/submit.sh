#!/bin/bash
# Submit the full benchmark as a Slurm dependency chain (pipeline contract: fetch_data ->
# cache -> predict -> score_all.sh). This benchmark is score-only, so there is no train job.
#
#   featurize --> infer_gpu (GNN, Transformer) --+
#                                                +--> score (score_all.sh: truth, metrics,
#   (after featurize) infer_cpu (rule4ml) ------+     LEADERBOARD.md, TIMINGS.md, Codabench zips)
#
# Usage, from the repo root on a login node, after setup.sh and the golden tests:
#     bash slurm/submit.sh -A <account> [--no-gpu] [--no-cpu] [sbatch args...]
# (WA_MACHINE=<machine> off NERSC.) Other arguments pass through to every sbatch call
# (e.g. -q debug). Per-job resources come from the machine profile (WA_SB_*); #SBATCH
# lines can't expand variables, so the account and resources go on the command line and
# the repo path goes through --export. Slurm logs (<job-name>-<jobid>.out) land in the
# repo root, the directory this script submits from.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
HPC_DIR="$(basename "$(dirname "${BASH_SOURCE[0]}")")"
source "$HPC_DIR/env.sh"

RUN_GPU=1 RUN_CPU=1
SB_ARGS=()
for a in "$@"; do
    case "$a" in
        --no-gpu) RUN_GPU=0 ;;
        --no-cpu|--no-mlp) RUN_CPU=0 ;;   # --no-mlp: name used before 2026-10-02
        *) SB_ARGS+=("$a") ;;
    esac
done
if [ "${WA_REQUIRE_ACCOUNT:-1}" = 1 ] && ! printf '%s\n' "${SB_ARGS[@]:-}" | grep -qE '^(-A|--account)'; then
    echo "error: pass your allocation, e.g. bash $HPC_DIR/submit.sh -A <account>" >&2
    exit 2
fi
for f in "$WA_WEIGHTS/gnn_resource_report_final_model.pth" "$WA_WEIGHTS/transformer_resource_report_final_model.pt" \
         "$WA_WEIGHTS/normalization_stats.json" "$WA_DATA/test"; do
    [ -e "$f" ] || { echo "error: missing $f -- run: bash $HPC_DIR/setup.sh" >&2; exit 1; }
done

# shellcheck disable=SC2086
sb() { local res="$1"; shift; sbatch --parsable --export=ALL,WA_REPO="$PWD",WA_HPC="$HPC_DIR",WA_MACHINE="$WA_MACHINE" $res "${SB_ARGS[@]}" "$@"; }

feat=$(sb "$WA_SB_FEATURIZE" "$HPC_DIR/jobs/featurize.sbatch")
deps=""
if [ "$RUN_GPU" = 1 ]; then
    gpu=$(sb "$WA_SB_INFER_GPU" --dependency="afterok:$feat" "$HPC_DIR/jobs/infer_gpu.sbatch")
    deps="$deps:$gpu"
fi
if [ "$RUN_CPU" = 1 ]; then
    cpu=$(sb "$WA_SB_INFER_CPU" --dependency="afterok:$feat" "$HPC_DIR/jobs/infer_cpu.sbatch")
    deps="$deps:$cpu"
fi
score=$(sb "$WA_SB_SCORE" --dependency="afterok${deps:-:$feat}" "$HPC_DIR/jobs/score.sbatch")

echo "featurize=$feat ${gpu:+infer_gpu=$gpu }${cpu:+infer_cpu=$cpu }score=$score"
echo "logs -> $PWD/<job-name>-<jobid>.out   results -> $WA_RESULTS/LEADERBOARD.md, TIMINGS.md   (watch: squeue --me)"
