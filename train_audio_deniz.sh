#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1                           # e.g. 1 for l40s and H100, 2 for A100 
#SBATCH --gpus-per-node=H100.80gb:1         # e.g. H100.80gb:8, A100:4 or L40S:4
#SBATCH --ntasks-per-node=1                 # set = GPUs per node
#SBATCH --cpus-per-task=14                   # 14 CPUs per GPU (because 1 task == 1 GPU)
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --job-name=Audio_AR
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

echo "Running in shell: $SHELL"

PROJECT_ROOT="/share/users/student/f/friverossego/LatentLM"
DATA_ROOT="/share/users/student/f/friverossego/datasets"
MANIFEST_PATHS="/share/users/student/f/friverossego/datasets/audio_manifest_train.jsonl"
SILENCE_LATENT_PATH="silence_samples/silence_10s_dacvae.pt"

# Derive from allocation
NUM_NODES="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}"
TPN_RAW="${SLURM_NTASKS_PER_NODE:-${SLURM_TASKS_PER_NODE:-1}}"
TASKS_PER_NODE="${TPN_RAW%%(*}"

echo "Resolved: num_nodes=$NUM_NODES tasks_per_node=$TASKS_PER_NODE"


SCRATCH_BASE="/share/users/student/d/dguen/tmp"
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

srun python train_audio.py \
  --num-nodes "$NUM_NODES" \
  --devices "$TASKS_PER_NODE" \
  --manifest-paths "$MANIFEST_PATHS" \
  --data-root "$DATA_ROOT" \
  --silence-latent-path "$SILENCE_LATENT_PATH" \
  --results-dir audio_logs/Transformer_B_30e \
  --model Transformer-B \
  --seq-len 251 \
  --latent-size 128 \
  --conditioning-type continuous \
  --prompt-seq-len 69 \
  --clap-dim 512 \
  --t5-dim 1024 \
  --batch-size 128 \
  --epochs 30 \
  --lr 1e-4 \
  --lr-warmup-steps 300 \
  --precision bf16-mixed \
  --gradient-accumulation-steps 2 
