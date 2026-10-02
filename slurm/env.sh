# Source before any benchmark step, on a login or compute node:
#     source slurm/env.sh                        # picks profiles/$NERSC_HOST.sh on NERSC machines
#     WA_MACHINE=<machine> source slurm/env.sh   # anywhere else
#
# Cluster facts come from profiles/$WA_MACHINE.sh and this benchmark's software-stack
# settings from stack.sh; this file and jobs/*.sbatch stay cluster-agnostic
# (references/hpc.md in the benchmark-builder skill). Override any path by exporting it
# before sourcing.

WA_HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Pick a profile: explicit WA_MACHINE, else NERSC's $NERSC_HOST, else generic.
: "${WA_MACHINE:=${NERSC_HOST:-generic}}"
if [ ! -f "$WA_HPC_DIR/profiles/$WA_MACHINE.sh" ]; then
    echo "env.sh: no profile $WA_HPC_DIR/profiles/$WA_MACHINE.sh (copy profiles/generic.sh)" >&2
    return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1090
source "$WA_HPC_DIR/profiles/$WA_MACHINE.sh"
# shellcheck disable=SC1091
source "$WA_HPC_DIR/stack.sh"

: "${WA_DATA:=$WA_ROOT/data/wa-hls4ml}"     # HF dataset snapshot (+ REVISION)
: "${WA_CACHE:=$WA_ROOT/cache}"             # featurized splits, <split>.npz
: "${WA_WEIGHTS:=$WA_ROOT/weights}"         # checkpoints + normalization_stats.json
: "${WA_RESULTS:=$WA_ROOT/results}"         # predictions, metrics, LEADERBOARD.md, TIMINGS.md, codabench/
: "${WA_VENV:=$WA_ROOT/venv-torch}"         # main stack: featurize, GNN, Transformer, scoring
: "${WA_VENV_ALT:=$WA_ROOT/venv-mlp}"       # alt stack: rule4ml (TensorFlow + its own torch)
: "${WA_SPLITS:=test exemplar}"
: "${WA_PYTHON:=python3}"                   # interpreter venvs are built from (after loading modules)
export WA_MACHINE WA_ROOT WA_DATA WA_CACHE WA_WEIGHTS WA_RESULTS \
       WA_VENV WA_VENV_ALT WA_SPLITS WA_PYTHON

# Repo root (this directory lives in <repo>/slurm/). Jobs get it via --export from submit.sh.
: "${WA_REPO:=$(cd "$WA_HPC_DIR/.." && pwd)}"
export WA_REPO
export PYTHONPATH="$WA_REPO/src${PYTHONPATH:+:$PYTHONPATH}"

# Unbuffered output so Slurm .out files show progress as it happens.
export PYTHONUNBUFFERED=1
wa_stack_env

wa_activate() {
    wa_load_main_stack
    # shellcheck disable=SC1091
    source "$WA_VENV/bin/activate"
}

wa_activate_alt() {
    wa_load_alt_stack
    # shellcheck disable=SC1091
    source "$WA_VENV_ALT/bin/activate"
}

# Names used by earlier versions of these scripts (perlmutter/env.sh, before 2026-10-02).
wa_activate_torch() { wa_activate; }
wa_activate_mlp() { wa_activate_alt; }
