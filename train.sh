#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1                           # e.g. 1 for l40s and H100, 2 for A100 
#SBATCH --gpus-per-node=H100.80gb:2         # e.g. H100.80gb:8, A100:4 or L40S:4
#SBATCH --ntasks-per-node=2                 # set = GPUs per node
#SBATCH --cpus-per-task=14                   # 14 CPUs per GPU (because 1 task == 1 GPU)
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --job-name=Audio_AR
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

# # Load user defaults (e.g., HF cache paths)
# if [ -f "$HOME/.bashrc" ]; then
#   . "$HOME/.bashrc"
# fi

echo "Running in shell: $SHELL"

PROJECT_ROOT="/share/users/student/f/friverossego/EqSynth"
DATA_ROOT="/share/users/student/f/friverossego/datasets"
MANIFEST_PATHS="/share/users/student/f/friverossego/datasets/audio_manifest_train.jsonl"
SILENCE_LATENT_PATH="silence_samples/silence_10s_dacvae.pt"

# Derive from allocation
NUM_NODES="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}"
TPN_RAW="${SLURM_NTASKS_PER_NODE:-${SLURM_TASKS_PER_NODE:-1}}"
TASKS_PER_NODE="${TPN_RAW%%(*}"

echo "Resolved: num_nodes=$NUM_NODES tasks_per_node=$TASKS_PER_NODE"

SCRATCH_BASE="/share/users/student/f/friverossego/tmp"
export TMPDIR="${SLURM_TMPDIR:-$SCRATCH_BASE/tmp}"
export TORCHINDUCTOR_CACHE_DIR="$SCRATCH_BASE/torchinductor"
export TRITON_CACHE_DIR="$SCRATCH_BASE/triton"
export CUDA_CACHE_PATH="$SCRATCH_BASE/cuda"
export XDG_CACHE_HOME="$SCRATCH_BASE/xdg"

mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME"
# --- Scratch layout ---
# Use SLURM_TMPDIR when available (node-local, auto-cleaned after job).
# Otherwise, fall back to a per-job folder under /scratch.
# SCRATCH_ROOT="/scratch/$USER"
# if [[ ! -d /scratch || ! -w /scratch ]]; then
#   SCRATCH_ROOT="/tmp/$USER"
# fi
# JOB_SCRATCH="${SLURM_TMPDIR:-$SCRATCH_ROOT/jobs/${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}}"

# # Temp + compiler caches (safe to delete; will be recreated)
# export TMPDIR="$JOB_SCRATCH/tmp"
# export TORCHINDUCTOR_CACHE_DIR="$JOB_SCRATCH/torchinductor"
# export TRITON_CACHE_DIR="$JOB_SCRATCH/triton"
# export CUDA_CACHE_PATH="$JOB_SCRATCH/cuda"
# export XDG_CACHE_HOME="$JOB_SCRATCH/xdg"


# mkdir -p \
#   "$TMPDIR" \
#   "$TORCHINDUCTOR_CACHE_DIR" \
#   "$TRITON_CACHE_DIR" \
#   "$CUDA_CACHE_PATH" \
#   "$XDG_CACHE_HOME"

# # If SLURM_TMPDIR is not set, clean fallback job scratch on exit.
# if [[ -z "${SLURM_TMPDIR:-}" ]]; then
#   trap 'rm -rf "$JOB_SCRATCH"' EXIT
# fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

module purge 2>/dev/null || true

spack load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jamendo

cd "$PROJECT_ROOT"

srun python train.py \
  --num-nodes "$NUM_NODES" \
  --devices "$TASKS_PER_NODE" \
  --manifest-paths "$MANIFEST_PATHS" \
  --data-root "$DATA_ROOT" \
  --silence-latent-path "$SILENCE_LATENT_PATH" \
  --results-dir audio_logs/AUDIO_NTP_Transformer_B_80e \
  --model Transformer-B \
  --batch-size 128 \
  --epochs 80 \
  --lr-warmup-steps 300 \
  --precision bf16-mixed 
