#!/bin/bash
# Submit the full benchmark as a Slurm dependency chain:
#
#   featurize (cpu) --> infer_gpu (gpu: GNN + Transformer) --+
#                                                            +--> score (cpu)
#                       infer_mlp (cpu: baseline MLP) -------+
#
# Usage (from the repo root, on a login node, after perlmutter/setup.sh):
#     bash perlmutter/submit.sh -A <nersc_project>        # e.g. -A m1234
#     bash perlmutter/submit.sh -A <nersc_project> --no-mlp
# Extra arguments before --no-mlp are passed to every sbatch call (e.g. -q debug).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source perlmutter/env.sh

RUN_MLP=1
SB_ARGS=()
for a in "$@"; do
    if [ "$a" = "--no-mlp" ]; then RUN_MLP=0; else SB_ARGS+=("$a"); fi
done
if ! printf '%s\n' "${SB_ARGS[@]:-}" | grep -qE '^(-A|--account)'; then
    echo "error: pass your NERSC project, e.g. bash perlmutter/submit.sh -A m1234" >&2
    exit 2
fi
for f in "$WA_WEIGHTS/gnn_final_model.pth" "$WA_WEIGHTS/transformer_best_model.pt" \
         "$WA_WEIGHTS/normalization_stats.json" "$WA_DATA/test"; do
    [ -e "$f" ] || { echo "error: missing $f -- run: bash perlmutter/setup.sh" >&2; exit 1; }
done

sb() { sbatch --parsable --export=ALL,WA_REPO="$PWD" "${SB_ARGS[@]}" "$@"; }

feat=$(sb perlmutter/jobs/featurize.sbatch)
gpu=$(sb --dependency=afterok:"$feat" perlmutter/jobs/infer_gpu.sbatch)
deps="afterok:$gpu"
if [ "$RUN_MLP" = 1 ]; then
    mlp=$(sb perlmutter/jobs/infer_mlp.sbatch)
    deps="$deps:$mlp"
fi
score=$(sb --dependency="$deps" perlmutter/jobs/score.sbatch)

echo "featurize=$feat infer_gpu=$gpu ${mlp:+infer_mlp=$mlp }score=$score"
echo "results -> $WA_RESULTS/LEADERBOARD.md   (watch: squeue --me)"
