#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --time=36:00:00
#SBATCH --job-name=codec_eval
#SBATCH -x klpsy-1
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

set -e

PROMPT_CSV="/share/users/student/f/friverossego/FlowSynth/audiocaps-test.csv"
FLOWSYNTH_ROOT="/share/users/student/f/friverossego/FlowSynth"
CODEC_EVAL_ROOT="/share/users/student/f/friverossego/FlowSynth/samples_codec_eval"

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

SEED=0
EVAL_WORKERS=1
SAMPLES_PER_AUDIO=5
RUN_TAG="s${SAMPLES_PER_AUDIO}_seed${SEED}"

DACVAE_OUT="$CODEC_EVAL_ROOT/dacvae/$RUN_TAG"
AUDIOLDM2_OUT="$CODEC_EVAL_ROOT/audioldm2/$RUN_TAG"
STABLE_AUDIO_OUT="$CODEC_EVAL_ROOT/stableaudio/$RUN_TAG"

module purge 2>/dev/null || true
spack load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jamendo

PYTHON_BIN="$(command -v python)"
if [[ -z "${PYTHON_BIN}" ]]; then
    echo "python not found after activating jamendo."
    exit 1
fi

echo "Using python: ${PYTHON_BIN}"
"${PYTHON_BIN}" -c "import torch; print('torch', torch.__version__)"

mkdir -p "$CODEC_EVAL_ROOT"
cd "$FLOWSYNTH_ROOT"

echo "=== Running latent ceiling for DACVAE, AudioLDM2, and Stable Audio ==="
"${PYTHON_BIN}" sample_latent_ceiling.py \
    --prompts-csv "$PROMPT_CSV" \
    --output-root "$CODEC_EVAL_ROOT" \
    --vae both \
    --samples-per-audio "$SAMPLES_PER_AUDIO" \
    --seed "$SEED"

echo "=== Evaluating DACVAE latent ceiling ==="
"${PYTHON_BIN}" evaluate.py --gen "$DACVAE_OUT" --workers "$EVAL_WORKERS"

echo "=== Evaluating AudioLDM2 latent ceiling ==="
"${PYTHON_BIN}" evaluate.py --gen "$AUDIOLDM2_OUT" --workers "$EVAL_WORKERS"

echo "=== Evaluating Stable Audio latent ceiling ==="
"${PYTHON_BIN}" evaluate.py --gen "$STABLE_AUDIO_OUT" --workers "$EVAL_WORKERS"

echo "Done."
echo "  DACVAE:       $DACVAE_OUT"
echo "  AudioLDM2:    $AUDIOLDM2_OUT"
echo "  Stable Audio: $STABLE_AUDIO_OUT"
