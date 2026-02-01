#!/usr/bin/env bash
#SBATCH -J cache_dacvae_wavcaps_all
#SBATCH -p workq
#SBATCH -N 24
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=900G
#SBATCH --time=48:00:00
#SBATCH -o slurm_logs/%x_%j.batch.out
#SBATCH -e slurm_logs/%x_%j.batch.err

PROJECT_ROOT="/share/users/student/f/friverossego/LatentLM"

module purge 2>/dev/null || true

SCRATCH_BASE="/share/users/student/f/friverossego/tmp"
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
# User config
# -----------------------------
WAVCAPS_AUDIO_ROOT="/share/users/student/f/friverossego/raw/wavcaps_audio"
OUT_ROOT="/share/users/student/f/friverossego/datasets/WavCaps"

SUBSETS=(
  "SoundBible"
  "BBC_Sound_Effects"
  "AudioSet_SL"
  "FreeSound"
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

mkdir -p slurm_logs

# -----------------------------
# Run subsets sequentially
# -----------------------------
for subset in "${SUBSETS[@]}"; do
  IN="${WAVCAPS_AUDIO_ROOT}/${subset}"
  OUT="${OUT_ROOT}/${subset}/audio_latents"

  echo "============================================================"
  echo "[subset] ${subset}"
  echo "[in]  ${IN}"
  echo "[out] ${OUT}"
  echo "============================================================"

  mkdir -p "$OUT"

  srun --cpu-bind=cores --kill-on-bad-exit=1 \
    --output="slurm_logs/%x_%j_${subset}_%t.out" \
    --error="slurm_logs/%x_%j_${subset}_%t.err" \
    python -m src.data_utils.cache_audio_files_cpu_pool \
      --data_dir "$IN" \
      --cached_path "$OUT" \
      --device cpu \
      --num_workers "$WORKERS" \
      --worker_threads "$WORKER_THREADS" \
      --chunk_size_latents "$CHUNK_SIZE_LATENTS" \
      --overlap_latents "$OVERLAP_LATENTS" \
      --weights "$WEIGHTS_PATH"

  echo "[done subset] ${subset}"
done

echo "[done all]"
