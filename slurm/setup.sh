#!/bin/bash
# One-time setup on a LOGIN node: build both venvs, download the dataset and the weights
# into $WA_ROOT. All downloads happen here; jobs never download anything.
#
#     bash slurm/setup.sh            # everything (WA_MACHINE=<machine> off NERSC)
#     bash slurm/setup.sh envs       # just the venvs
#     bash slurm/setup.sh data       # just the dataset
#     bash slurm/setup.sh weights    # just the weights (sha256-checked)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
mkdir -p "$WA_ROOT"

make_venv() {  # make_venv <main|alt> <venv path> <requirements file> [extra pip args]
    local kind="$1" venv="$2" req="$3"; shift 3
    if [ "${WA_VENV_SYSTEM_SITE:-0}" = 1 ] && [ "$kind" = main ]; then
        "$WA_PYTHON" -m venv --system-site-packages "$venv"   # reuse the site module's torch
    else
        "$WA_PYTHON" -m venv "$venv"
    fi
    "$venv/bin/pip" install --upgrade pip
    "$venv/bin/pip" install -r "$req" "$@"
    wa_post_install "$kind" "$venv"
    wa_check_env "$kind" "$venv"
}

step_envs() {
    echo "== main venv ($WA_VENV) on $WA_MACHINE: featurize, GNN, Transformer, scoring"
    wa_load_main_stack
    make_venv main "$WA_VENV" "$WA_REPO/requirements.txt" pytest
    wa_unload_main_stack

    echo "== alt venv ($WA_VENV_ALT): rule4ml (TensorFlow + its own torch, kept apart)"
    wa_load_alt_stack
    make_venv alt "$WA_VENV_ALT" "$WA_REPO/requirements-mlp.txt"
}

step_data() {
    echo "== dataset -> $WA_DATA"
    "$WA_VENV/bin/python" "$WA_REPO/scripts/fetch_data.py" --out "$WA_DATA"
}

step_weights() {
    echo "== weights -> $WA_WEIGHTS (sha256-checked)"
    "$WA_VENV/bin/python" "$WA_REPO/scripts/fetch_weights.py" --out "$WA_WEIGHTS"
}

case "${1:-all}" in
    envs) step_envs ;;
    data) step_data ;;
    weights) step_weights ;;
    all) step_envs; step_data; step_weights ;;
    *) echo "usage: $0 [all|envs|data|weights]"; exit 2 ;;
esac
echo "done. Next: run the golden tests (docs/PERLMUTTER.md section 1), then bash slurm/submit.sh -A <account>"
