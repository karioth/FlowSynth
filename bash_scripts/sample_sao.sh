#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=12:00:00
#SBATCH --job-name=stable_audio_sample
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

set -e

BATCH_SIZE=16
STEPS=100
SEED=0
PRECISION=bf16-mixed
EVAL_WORKERS=1

PROMPT_CSV="/share/users/student/f/friverossego/FlowSynth/audiocaps-test.csv"
OUT_ROOT="/share/users/student/f/friverossego/FlowSynth/samples_baselines/stable_audio_open"
OUT_DIR="$OUT_ROOT/cfg7"

FLOWSYNTH_ROOT="/share/users/student/f/friverossego/FlowSynth"

SCRATCH_BASE="/share/users/student/f/friverossego/tmp"
export TMPDIR="${SLURM_TMPDIR:-$SCRATCH_BASE/tmp}"
export TORCHINDUCTOR_CACHE_DIR="$SCRATCH_BASE/torchinductor"
export TRITON_CACHE_DIR="$SCRATCH_BASE/triton"
export CUDA_CACHE_PATH="$SCRATCH_BASE/cuda"
export XDG_CACHE_HOME="$SCRATCH_BASE/xdg"
export HF_HOME="/share/users/student/f/friverossego/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HUGGINGFACE_HUB_CACHE"
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME"
mkdir -p "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

module purge 2>/dev/null || true
spack load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jamendo

cd "$FLOWSYNTH_ROOT"

mkdir -p "$OUT_DIR"

python sample_stable_audio.py \
  --prompt-csv "$PROMPT_CSV" \
  --output-dir "$OUT_DIR" \
  --steps "$STEPS" \
  --audio-length 10 \
  --guidance-scale 7.0 \
  --negative-prompt "" \
  --batch-size "$BATCH_SIZE" \
  --seed "$SEED" \
  --precision "$PRECISION"

python evaluate.py --gen "$OUT_DIR" --workers "$EVAL_WORKERS"
