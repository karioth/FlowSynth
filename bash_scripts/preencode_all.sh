#!/usr/bin/env bash
#SBATCH -J preprocess_all
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
DATA_ROOT="/share/users/student/f/friverossego/datasets"
WAVCAPS_JSON_ROOT="/share/users/student/f/friverossego/raw/wavcaps_hf/json_files"
WAVCAPS_AUDIO_ROOT="/share/users/student/f/friverossego/raw/wavcaps_audio"

# Optional AudioCaps HF paths
AUDIOCAPS_HF_DATA_DIR=""
AUDIOCAPS_HF_CACHE_DIR=""

WORKERS=28
WORKER_THREADS=4
CHUNK_SIZE_LATENTS=16384
OVERLAP_LATENTS=12

# Models
WEIGHTS_PATH="/share/users/student/f/friverossego/.cache/huggingface/hub/models--facebook--dacvae-watermarked/snapshots/8680102d141858a21bd533543966a2eb2e569f92/weights.pth"
CLAP_MODEL="laion/larger_clap_music"
T5_MODEL="google/flan-t5-large"

export OMP_NUM_THREADS=$WORKER_THREADS
export MKL_NUM_THREADS=$WORKER_THREADS
export OPENBLAS_NUM_THREADS=$WORKER_THREADS
export NUMEXPR_NUM_THREADS=$WORKER_THREADS

# -----------------------------
# Optional: force offline mode if caches are present
# -----------------------------
USE_OFFLINE=0
if [[ "$USE_OFFLINE" -eq 1 ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

mkdir -p slurm_logs

ARGS=(
  --data-root "$DATA_ROOT"
  --wavcaps-json-root "$WAVCAPS_JSON_ROOT"
  --wavcaps-audio-root "$WAVCAPS_AUDIO_ROOT"
  --device cpu
  --processes "$WORKERS"
  --threads-per-process "$WORKER_THREADS"
  --chunk-size-latents "$CHUNK_SIZE_LATENTS"
  --overlap-latents "$OVERLAP_LATENTS"
  --dacvae-weights "$WEIGHTS_PATH"
  --clap-model "$CLAP_MODEL"
  --t5-model "$T5_MODEL"
)

if [[ -n "$AUDIOCAPS_HF_DATA_DIR" ]]; then
  ARGS+=(--audiocaps-hf-data-dir "$AUDIOCAPS_HF_DATA_DIR")
fi
if [[ -n "$AUDIOCAPS_HF_CACHE_DIR" ]]; then
  ARGS+=(--audiocaps-hf-cache-dir "$AUDIOCAPS_HF_CACHE_DIR")
fi

echo "============================================================"
echo "[preprocess] AudioCaps + WavCaps"
echo "[data-root] ${DATA_ROOT}"
echo "============================================================"

srun --cpu-bind=cores --kill-on-bad-exit=1 \
  --output="slurm_logs/%x_%j_%t.out" \
  --error="slurm_logs/%x_%j_%t.err" \
  python preprocess.py "${ARGS[@]}"

echo "[done]"
