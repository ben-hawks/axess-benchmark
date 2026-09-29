# Source this on Perlmutter (login or compute node) before any benchmark step:
#     source perlmutter/env.sh
#
# Override any of these by exporting them before sourcing.

# NERSC PyTorch module the GNN/Transformer venv is layered on. Check what is installed
# with `module avail pytorch`; the reference runs used torch >= 2.6.
: "${WA_PYTORCH_MODULE:=pytorch/2.6.0}"

# Working area on $SCRATCH (purged after 8 weeks of no access -- see README).
: "${WA_ROOT:=$SCRATCH/wa-hls4ml}"
: "${WA_DATA:=$WA_ROOT/data/wa-hls4ml}"        # HF dataset snapshot
: "${WA_CACHE:=$WA_ROOT/cache}"                # featurized splits (.npz)
: "${WA_WEIGHTS:=$WA_ROOT/weights}"            # checkpoints + normalization_stats.json
: "${WA_RESULTS:=$WA_ROOT/results}"            # predictions, METRICS.md, LEADERBOARD.md
: "${WA_VENV:=$WA_ROOT/venv-torch}"            # GNN / Transformer / scoring
: "${WA_VENV_MLP:=$WA_ROOT/venv-mlp}"          # rule4ml baseline MLP (TensorFlow)

# Splits the benchmark reports on.
: "${WA_SPLITS:=test exemplar}"

export WA_PYTORCH_MODULE WA_ROOT WA_DATA WA_CACHE WA_WEIGHTS WA_RESULTS WA_VENV WA_VENV_MLP WA_SPLITS

# Repo root (this file lives in <repo>/perlmutter/).
WA_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export WA_REPO
export PYTHONPATH="$WA_REPO/src${PYTHONPATH:+:$PYTHONPATH}"

# Quiet TF/oneDNN chatter; keep torch.compile/dynamo out of the inference path.
export TF_CPP_MIN_LOG_LEVEL=3 TORCHDYNAMO_DISABLE=1 PYTHONUNBUFFERED=1

wa_activate_torch() {
    module load "$WA_PYTORCH_MODULE"
    # shellcheck disable=SC1091
    source "$WA_VENV/bin/activate"
}

wa_activate_mlp() {
    module load python
    # shellcheck disable=SC1091
    source "$WA_VENV_MLP/bin/activate"
}
