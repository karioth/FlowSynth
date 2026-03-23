#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --time=36:00:00
#SBATCH --job-name=pretrained_cfg_eval
#SBATCH -x klpsy-1
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err


PROMPT_CSV="/share/users/student/f/friverossego/FlowSynth/audiocaps-test.csv"
FLOWSYNTH_ROOT="/share/users/student/f/friverossego/FlowSynth"
OUT_ROOT="/share/users/student/f/friverossego/FlowSynth/samples_baselines"

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

AUDIO_LENGTH=10
SEED=0
PRECISION="bf16-mixed"
EVAL_WORKERS=1

# Keep the sampler step counts aligned with the current script defaults so CFG is
# the main variable that changes relative to your older cfg1 runs.
AUDIOLDM2_STEPS=100
AUDIOLDM2_BATCH_SIZE=32
STABLE_AUDIO_STEPS=100
STABLE_AUDIO_BATCH_SIZE=32
TANGOFLUX_STEPS=100

AUDIO_LDM2_OUT="$OUT_ROOT/audioldm2/cfg3.5"
STABLE_AUDIO_OUT="$OUT_ROOT/stable_audio_open/cfg7"
TANGOFLUX_OUT="$OUT_ROOT/tangoflux/cfg4.5"

LATENT_CEILING_ROOT="/share/users/student/f/friverossego/FlowSynth/samples_audio_vae_ceilings"
LATENT_CEILING_SAMPLES_PER_AUDIO=5
LATENT_CEILING_RUN_TAG="s${LATENT_CEILING_SAMPLES_PER_AUDIO}_seed${SEED}"
LATENT_CEILING_AUDIOGEN_NQ1_OUT="$LATENT_CEILING_ROOT/audiogen_nq1/$LATENT_CEILING_RUN_TAG"
LATENT_CEILING_AUDIOGEN_NQ2_OUT="$LATENT_CEILING_ROOT/audiogen_nq2/$LATENT_CEILING_RUN_TAG"
LATENT_CEILING_AUDIOGEN_NQ3_OUT="$LATENT_CEILING_ROOT/audiogen_nq3/$LATENT_CEILING_RUN_TAG"
LATENT_CEILING_AUDIOGEN_FULL_OUT="$LATENT_CEILING_ROOT/audiogen/$LATENT_CEILING_RUN_TAG"

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

mkdir -p "$AUDIO_LDM2_OUT" "$STABLE_AUDIO_OUT" "$TANGOFLUX_OUT"
cd "$FLOWSYNTH_ROOT"

echo "=== Sampling AudioLDM2 @ cfg 3.5 ==="
"${PYTHON_BIN}" sample_audioldm2.py \
    --prompt-csv "$PROMPT_CSV" \
    --output-dir "$AUDIO_LDM2_OUT" \
    --steps "$AUDIOLDM2_STEPS" \
    --audio-length "$AUDIO_LENGTH" \
    --guidance-scale 3.5 \
    --batch-size "$AUDIOLDM2_BATCH_SIZE" \
    --seed "$SEED" \
    --precision "$PRECISION" 

echo "=== Sampling Stable Audio Open @ cfg 7.0 ==="
"${PYTHON_BIN}" sample_stable_audio.py \
    --prompt-csv "$PROMPT_CSV" \
    --output-dir "$STABLE_AUDIO_OUT" \
    --steps "$STABLE_AUDIO_STEPS" \
    --audio-length "$AUDIO_LENGTH" \
    --guidance-scale 7.0 \
    --negative-prompt "low quality, average quality" \
    --batch-size "$STABLE_AUDIO_BATCH_SIZE" \
    --seed "$SEED" \
    --precision "$PRECISION" 

echo "=== Sampling TangoFlux @ cfg 4.5 ==="
"${PYTHON_BIN}" sample_tangoflux.py \
    --prompt-csv "$PROMPT_CSV" \
    --output-dir "$TANGOFLUX_OUT" \
    --steps "$TANGOFLUX_STEPS" \
    --audio-length "$AUDIO_LENGTH" \
    --guidance-scale 4.5 \
    --seed "$SEED" \
    --precision "$PRECISION" 

echo "=== Evaluating AudioLDM2 cfg3.5 ==="
"${PYTHON_BIN}" evaluate.py --gen "$AUDIO_LDM2_OUT" --workers "$EVAL_WORKERS"

echo "=== Evaluating Stable Audio Open cfg7 ==="
"${PYTHON_BIN}" evaluate.py --gen "$STABLE_AUDIO_OUT" --workers "$EVAL_WORKERS"

echo "=== Evaluating TangoFlux cfg4.5 ==="
"${PYTHON_BIN}" evaluate.py --gen "$TANGOFLUX_OUT" --workers "$EVAL_WORKERS"

echo "=== Switching to audiogen_env for AudioGen latent-ceiling sweep ==="
conda activate audiogen_env
AUDIOGEN_PYTHON="$(command -v python)"
if [[ -z "${AUDIOGEN_PYTHON}" ]]; then
    echo "python not found after activating audiogen_env."
    exit 1
fi
echo "Using latent-ceiling python: ${AUDIOGEN_PYTHON}"

echo "=== Running AudioGen latent ceiling full codec sweep ==="
"${AUDIOGEN_PYTHON}" sample_latent_ceiling.py \
    --output-root "$LATENT_CEILING_ROOT" \
    --vae audiogen \
    --samples-per-audio "$LATENT_CEILING_SAMPLES_PER_AUDIO" \
    --seed "$SEED" \
    --audiogen-num-codebooks 1 2 3 4

echo "=== Switching back to jamendo for latent-ceiling evaluation ==="
conda activate jamendo
PYTHON_BIN="$(command -v python)"
if [[ -z "${PYTHON_BIN}" ]]; then
    echo "python not found after re-activating jamendo."
    exit 1
fi
echo "Using eval python: ${PYTHON_BIN}"

echo "=== Evaluating AudioGen latent ceiling nq1 ==="
"${PYTHON_BIN}" evaluate.py --gen "$LATENT_CEILING_AUDIOGEN_NQ1_OUT" --workers "$EVAL_WORKERS"

echo "=== Evaluating AudioGen latent ceiling nq2 ==="
"${PYTHON_BIN}" evaluate.py --gen "$LATENT_CEILING_AUDIOGEN_NQ2_OUT" --workers "$EVAL_WORKERS"

echo "=== Evaluating AudioGen latent ceiling nq3 ==="
"${PYTHON_BIN}" evaluate.py --gen "$LATENT_CEILING_AUDIOGEN_NQ3_OUT" --workers "$EVAL_WORKERS"

echo "=== Evaluating AudioGen latent ceiling full codec ==="
"${PYTHON_BIN}" evaluate.py --gen "$LATENT_CEILING_AUDIOGEN_FULL_OUT" --workers "$EVAL_WORKERS"

echo "Done."
echo "  AudioLDM2:        $AUDIO_LDM2_OUT"
echo "  Stable Audio Open: $STABLE_AUDIO_OUT"
echo "  TangoFlux:         $TANGOFLUX_OUT"
echo "  Latent ceiling nq1: $LATENT_CEILING_AUDIOGEN_NQ1_OUT"
echo "  Latent ceiling nq2: $LATENT_CEILING_AUDIOGEN_NQ2_OUT"
echo "  Latent ceiling nq3: $LATENT_CEILING_AUDIOGEN_NQ3_OUT"
echo "  Latent ceiling full: $LATENT_CEILING_AUDIOGEN_FULL_OUT"
