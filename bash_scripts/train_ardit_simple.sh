#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=5
#SBATCH --mem=350G
#SBATCH --time=10:00:00
#SBATCH --job-name=Audio_ardit_simple_B
#SBATCH -x klpsy-1
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

echo "Running in shell: $SHELL"

PROJECT_ROOT="/share/users/student/f/friverossego/FlowSynth"
DATA_ROOT="/share/users/student/f/friverossego/datasets"
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
  --data-root "$DATA_ROOT" \
  --silence-latent-path "$SILENCE_LATENT_PATH" \
  --results-dir audio_logs/AUDIO_AR_DiT_Simple_B \
  --model AR-DiT-Simple-B \
  --batch-size 128 \
  --epochs 125 \
  --resume /share/users/student/f/friverossego/FlowSynth/audio_logs/AUDIO_AR_DiT_Simple_B/checkpoints/last.ckpt
