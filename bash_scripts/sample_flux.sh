#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --time=12:00:00
#SBATCH --job-name=flux_sample
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

set -e

NUM_GPUS=1
STEPS=100
SEED=0
PRECISION=bf16-mixed

PROMPT_CSV="/share/users/student/f/friverossego/FlowSynth/audiocaps-test.csv"
OUT_ROOT="/share/users/student/f/friverossego/FlowSynth/samples_audio_final_per_model"

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
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME"
mkdir -p "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

module purge 2>/dev/null || true
spack load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jamendo

cd "$FLOWSYNTH_ROOT"

python sample_tangoflux.py \
  --prompt-csv "$PROMPT_CSV" \
  --output-dir "$OUT_ROOT/tangoflux/cfg4_5" \
  --steps "$STEPS" \
  --audio-length 10 \
  --guidance-scale 4.5 \
  --seed "$SEED" \
  --precision "$PRECISION"

