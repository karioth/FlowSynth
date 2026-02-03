#!/usr/bin/env bash
#SBATCH -J cache_dacvae_audiocaps
#SBATCH -p workq
#SBATCH -N 6
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=900G
#SBATCH --time=24:00:00
#SBATCH -o slurm_logs/%x_%j.batch.out
#SBATCH -e slurm_logs/%x_%j.batch.err

echo "Running in shell: $SHELL"

PROJECT_ROOT="/share/users/student/f/friverossego/LatentLM"

module purge 2>/dev/null || true

SCRATCH_BASE="/share/users/student/d/dguen/tmp"
export TMPDIR="${SLURM_TMPDIR:-$SCRATCH_BASE/tmp}"
export TORCHINDUCTOR_CACHE_DIR="$SCRATCH_BASE/torchinductor"
export TRITON_CACHE_DIR="$SCRATCH_BASE/triton"
export CUDA_CACHE_PATH="$SCRATCH_BASE/cuda"
export XDG_CACHE_HOME="$SCRATCH_BASE/xdg"

mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME"

cd "$PROJECT_ROOT"
spack load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
eval "$(conda shell.bash hook)"
conda activate jamendo

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
export PYTHONWARNINGS=ignore
# -----------------------------
# User config (edit if needed)
# -----------------------------
OUT_ROOT="/share/users/student/f/friverossego/datasets/AudioCaps"

SPLITS=(
  "train"
  "validation"
  "test"
)

WORKERS=28
WORKER_THREADS=4
CHUNK_SIZE_LATENTS=16384
OVERLAP_LATENTS=12

export OMP_NUM_THREADS=$WORKER_THREADS
export MKL_NUM_THREADS=$WORKER_THREADS
export OPENBLAS_NUM_THREADS=$WORKER_THREADS
export NUMEXPR_NUM_THREADS=$WORKER_THREADS

# -----------------------------
# Ensure weights are local; then force offline
# -----------------------------
WEIGHTS_PATH="/share/users/student/f/friverossego/.cache/huggingface/hub/models--facebook--dacvae-watermarked/snapshots/8680102d141858a21bd533543966a2eb2e569f92/weights.pth"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "[weights] $WEIGHTS_PATH"

# -----------------------------
# Run splits sequentially
# -----------------------------
mkdir -p slurm_logs
for split in "${SPLITS[@]}"; do
  OUT="${OUT_ROOT}/${split}/audio_latents"

  echo "============================================================"
  echo "[split] ${split}"
  echo "[out] ${OUT}"
  echo "============================================================"

  mkdir -p "$OUT"

  srun --cpu-bind=cores --kill-on-bad-exit=1 \
    --output="slurm_logs/%x_%j_${split}_%t.out" \
    --error="slurm_logs/%x_%j_${split}_%t.err" \
    python -m src.data_utils.cache_audio_hfparquet_cpu_pool \
      --hf_dataset "OpenSound/AudioCaps" \
      --hf_split "$split" \
      --cached_path "$OUT" \
      --device cpu \
      --name_column audiocap_id \
      --num_workers "$WORKERS" \
      --worker_threads "$WORKER_THREADS" \
      --chunk_size_latents "$CHUNK_SIZE_LATENTS" \
      --overlap_latents "$OVERLAP_LATENTS" \
      --weights "$WEIGHTS_PATH"

  echo "[done split] ${split}"
done

echo "[done all]"
