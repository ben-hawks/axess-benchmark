# This benchmark's software-stack hooks, sourced by env.sh after the machine profile.
#
# Profiles describe the cluster; this file describes what THIS benchmark's stack needs on
# it. Each setting below fixes a problem actually hit with this stack; docs/PERLMUTTER.md
# ("Troubleshooting") records the symptom and cause.

# Environment every job needs.
wa_stack_env() {
    # torch_geometric imports torch._dynamo, which imports triton; on CPU-only nodes the
    # triton import segfaulted. Nothing here uses torch.compile, so keep dynamo off.
    # TF_CPP_MIN_LOG_LEVEL quiets TensorFlow/oneDNN log spam in the rule4ml job.
    export TORCHDYNAMO_DISABLE=1 TF_CPP_MIN_LOG_LEVEL=3
}

# Fixes after `pip install -r <requirements>`. $1 is "main" or "alt", $2 the venv path.
wa_post_install() {
    # The rule4ml venv runs on CPU only, and triton (pulled in by its torch) segfaulted on
    # import there; neither rule4ml model uses it.
    if [ "$1" = alt ]; then "$2/bin/pip" uninstall -y triton || true; fi
}

# Import check after installing, so a broken environment fails in setup.sh rather than in a
# queued job. $1 is "main" or "alt", $2 the venv path.
wa_check_env() {
    if [ "$1" = main ]; then
        "$2/bin/python" -c "import torch, torch_geometric, wa_hls4ml_bench; print('torch', torch.__version__, 'pyg', torch_geometric.__version__, 'cuda', torch.version.cuda)"
    else
        "$2/bin/python" -c "import rule4ml, wa_hls4ml_bench; print('rule4ml', rule4ml.__version__)"
    fi
}
