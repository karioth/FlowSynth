#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --time=36:00:00
#SBATCH --job-name=codec_eval_audiogen
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

AUDIOGEN_NQ1_OUT="$CODEC_EVAL_ROOT/audiogen_nq1/$RUN_TAG"
AUDIOGEN_NQ2_OUT="$CODEC_EVAL_ROOT/audiogen_nq2/$RUN_TAG"
AUDIOGEN_NQ3_OUT="$CODEC_EVAL_ROOT/audiogen_nq3/$RUN_TAG"
AUDIOGEN_FULL_OUT="$CODEC_EVAL_ROOT/audiogen/$RUN_TAG"

module purge 2>/dev/null || true
spack load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

mkdir -p "$CODEC_EVAL_ROOT"
cd "$FLOWSYNTH_ROOT"

echo "=== Switching to audiogen_env for AudioGen latent ceiling ==="
conda activate audiogen_env
AUDIOGEN_PYTHON="$(command -v python)"
if [[ -z "${AUDIOGEN_PYTHON}" ]]; then
    echo "python not found after activating audiogen_env."
    exit 1
fi
echo "Using AudioGen python: ${AUDIOGEN_PYTHON}"

echo "=== Running AudioGen latent ceiling full codec sweep ==="
"${AUDIOGEN_PYTHON}" sample_latent_ceiling_audiogen.py \
    --prompts-csv "$PROMPT_CSV" \
    --output-root "$CODEC_EVAL_ROOT" \
    --samples-per-audio "$SAMPLES_PER_AUDIO" \
    --seed "$SEED" \
    --audiogen-num-codebooks 1 2 3 4

echo "=== Switching to jamendo for evaluation ==="
conda activate jamendo
PYTHON_BIN="$(command -v python)"
if [[ -z "${PYTHON_BIN}" ]]; then
    echo "python not found after activating jamendo."
    exit 1
fi
echo "Using eval python: ${PYTHON_BIN}"

echo "=== Evaluating AudioGen latent ceiling nq1 ==="
"${PYTHON_BIN}" evaluate.py --gen "$AUDIOGEN_NQ1_OUT" --workers "$EVAL_WORKERS"

echo "=== Evaluating AudioGen latent ceiling nq2 ==="
"${PYTHON_BIN}" evaluate.py --gen "$AUDIOGEN_NQ2_OUT" --workers "$EVAL_WORKERS"

echo "=== Evaluating AudioGen latent ceiling nq3 ==="
"${PYTHON_BIN}" evaluate.py --gen "$AUDIOGEN_NQ3_OUT" --workers "$EVAL_WORKERS"

echo "=== Evaluating AudioGen latent ceiling full codec ==="
"${PYTHON_BIN}" evaluate.py --gen "$AUDIOGEN_FULL_OUT" --workers "$EVAL_WORKERS"

echo "Done."
echo "  AudioGen nq1:  $AUDIOGEN_NQ1_OUT"
echo "  AudioGen nq2:  $AUDIOGEN_NQ2_OUT"
echo "  AudioGen nq3:  $AUDIOGEN_NQ3_OUT"
echo "  AudioGen full: $AUDIOGEN_FULL_OUT"
