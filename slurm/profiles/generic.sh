# Machine profile TEMPLATE. Copy to profiles/<machine>.sh, fill in every value for that
# cluster (partition/qos names, GPU request syntax, filesystems, modules), and run with
# WA_MACHINE=<machine>. Don't guess: a wrong partition fails at submit time, but a wrong
# filesystem choice (purged scratch) loses results silently weeks later. Document what you
# assumed, and where each value came from, in docs/<MACHINE>.md.

WA_MACHINE="<machine>"   # profile name; file is profiles/<machine>.sh

# --- filesystems ---------------------------------------------------------------
: "${WA_ROOT:=<scratch or project path>/wa-hls4ml}"

# --- scheduler -----------------------------------------------------------------
WA_REQUIRE_ACCOUNT=1     # 0 if the site has no accounts/allocations
# Partition/qos, --constraint, GPU syntax (--gpus=N | --gres=gpu:N | --gpus-per-node=N),
# wall time. Perlmutter's measured needs (profiles/perlmutter.sh) are a guide to sizes:
# featurize ~20 s on 64 cores; GPU inference ~1 min on one A100; CPU inference ~22 min on
# 32 cores; scoring a few minutes.
WA_SB_FEATURIZE="--partition=<cpu> --cpus-per-task=<n> --mem=<mem> --time=<hh:mm:ss>"
WA_SB_INFER_CPU="--partition=<cpu> --cpus-per-task=<n> --mem=<mem> --time=<hh:mm:ss>"
WA_SB_INFER_GPU="--partition=<gpu> --gres=gpu:1 --cpus-per-task=<n> --time=<hh:mm:ss>"
WA_SB_SCORE="--partition=<cpu> --cpus-per-task=<n> --mem=<mem> --time=<hh:mm:ss>"

# --- network -------------------------------------------------------------------
WA_COMPUTE_HAS_INTERNET=0  # setup.sh downloads everything on a login node either way

# --- GPUs ------------------------------------------------------------------------
: "${WA_GPU_DEVICE:=cuda}"   # PyTorch device name for this machine's GPUs
wa_gpu_info() { nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true; }

# --- software ------------------------------------------------------------------
# Main stack: PyTorch + PyTorch Geometric (requirements.txt). Layer the venv on a site
# PyTorch module if there is one (WA_VENV_SYSTEM_SITE=1), otherwise requirements.txt
# needs torch added. Alt stack: rule4ml + TensorFlow (requirements-mlp.txt).
: "${WA_MAIN_MODULE:=}"
: "${WA_ALT_MODULE:=}"
: "${WA_VENV_SYSTEM_SITE:=0}"
wa_load_main_stack() { if [ -n "$WA_MAIN_MODULE" ]; then module load "$WA_MAIN_MODULE"; fi; }
wa_unload_main_stack() { if [ -n "$WA_MAIN_MODULE" ]; then module unload "$WA_MAIN_MODULE"; fi; }
wa_load_alt_stack() { if [ -n "$WA_ALT_MODULE" ]; then module load "$WA_ALT_MODULE"; fi; }
