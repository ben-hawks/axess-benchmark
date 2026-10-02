# Machine profile: NERSC Perlmutter. Sourced by env.sh; every cluster-specific value lives
# here, never in jobs/*.sbatch. Verified with this benchmark on 2026-10-02 (Slurm jobs
# 59209565-68, all exit 0; see docs/VALIDATION.md section 7 and docs/PERLMUTTER.md).

WA_MACHINE=perlmutter

# --- filesystems ---------------------------------------------------------------
# $SCRATCH is purged for files not accessed in 8 weeks. Copy results worth keeping to CFS
# (/global/cfs/cdirs/<project>/...) or export WA_ROOT there before sourcing.
: "${WA_ROOT:=$SCRATCH/wa-hls4ml}"

# --- scheduler -----------------------------------------------------------------
# #SBATCH lines can't expand variables, so submit.sh requires the account on its command
# line (bash slurm/submit.sh -A <nersc_project>) and passes these per job.
WA_REQUIRE_ACCOUNT=1
# Sizes from the 2026-10-02 run (102,484 test + 887 exemplar samples), with headroom:
#   featurize 21 s on 64 cores; GPU inference ~1 min on one A100;
#   CPU inference (rule4ml MLP + GNN, both splits) ~22 min on 32 cores; scoring < 5 min.
WA_SB_FEATURIZE="--constraint=cpu --qos=shared --cpus-per-task=64 --mem=120G --time=00:15:00"
WA_SB_INFER_CPU="--constraint=cpu --qos=shared --cpus-per-task=32 --mem=60G --time=01:00:00"
WA_SB_INFER_GPU="--constraint=gpu --qos=shared --gpus=1 --cpus-per-task=32 --time=00:15:00"
WA_SB_SCORE="--constraint=cpu --qos=shared --cpus-per-task=4 --mem=16G --time=00:15:00"

# --- network -------------------------------------------------------------------
# Compute nodes can reach the internet, but downloads still run on the login node in
# setup.sh so jobs never depend on it.
WA_COMPUTE_HAS_INTERNET=1

# --- GPUs ------------------------------------------------------------------------
# NVIDIA A100-SXM4-40GB; PyTorch device name "cuda".
: "${WA_GPU_DEVICE:=cuda}"
wa_gpu_info() { nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true; }

# --- software ------------------------------------------------------------------
# Main venv layered on NERSC's CUDA build of PyTorch (venv --system-site-packages); the
# rule4ml venv on plain `module load python`. List what's installed with `module avail`.
: "${WA_MAIN_MODULE:=pytorch/2.6.0}"
: "${WA_ALT_MODULE:=python}"
: "${WA_VENV_SYSTEM_SITE:=1}"
wa_load_main_stack() { module load "$WA_MAIN_MODULE"; }
wa_unload_main_stack() { module unload "$WA_MAIN_MODULE"; }
wa_load_alt_stack() { module load "$WA_ALT_MODULE"; }
