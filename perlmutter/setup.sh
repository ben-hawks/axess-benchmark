#!/bin/bash
# One-time setup on a Perlmutter LOGIN node: create both venvs, download the dataset
# and the pretrained weights into $WA_ROOT on $SCRATCH.
#
#     bash perlmutter/setup.sh            # everything
#     bash perlmutter/setup.sh envs       # just the venvs
#     bash perlmutter/setup.sh data       # just the dataset
#     bash perlmutter/setup.sh weights    # just the weights
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
mkdir -p "$WA_ROOT"

step_envs() {
    echo "== torch venv ($WA_VENV) on top of $WA_PYTORCH_MODULE"
    module load "$WA_PYTORCH_MODULE"
    python -m venv --system-site-packages "$WA_VENV"
    "$WA_VENV/bin/pip" install --upgrade pip
    "$WA_VENV/bin/pip" install -r "$WA_REPO/requirements.txt"
    "$WA_VENV/bin/python" -c "import torch, torch_geometric; print('torch', torch.__version__, 'pyg', torch_geometric.__version__, 'cuda', torch.version.cuda)"
    module unload "$WA_PYTORCH_MODULE"

    echo "== MLP venv ($WA_VENV_MLP): rule4ml 0.2.0 + TensorFlow"
    module load python
    python -m venv "$WA_VENV_MLP"
    "$WA_VENV_MLP/bin/pip" install --upgrade pip
    "$WA_VENV_MLP/bin/pip" install -r "$WA_REPO/requirements-mlp.txt"
    # rule4ml imports torch_geometric -> torch._dynamo -> triton. On CPU-only nodes the
    # triton import can segfault; the MLP never uses it (see docs/VALIDATION.md).
    "$WA_VENV_MLP/bin/pip" uninstall -y triton || true
    "$WA_VENV_MLP/bin/python" -c "import rule4ml; print('rule4ml', rule4ml.__version__)"
}

step_data() {
    echo "== dataset -> $WA_DATA"
    "$WA_VENV/bin/python" "$WA_REPO/scripts/fetch_data.py" --out "$WA_DATA"
}

step_weights() {
    echo "== weights -> $WA_WEIGHTS"
    "$WA_VENV/bin/python" "$WA_REPO/scripts/fetch_weights.py" --out "$WA_WEIGHTS"
}

case "${1:-all}" in
    envs) step_envs ;;
    data) step_data ;;
    weights) step_weights ;;
    all) step_envs; step_data; step_weights ;;
    *) echo "usage: $0 [all|envs|data|weights]"; exit 2 ;;
esac
echo "done. Next: bash perlmutter/submit.sh -A <nersc_project>"
