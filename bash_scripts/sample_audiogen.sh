#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --time=18:00:00
#SBATCH --job-name=audiogen_cfg13
#SBATCH -x klpsy-1
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

set -e

NUM_GPUS=2
BATCH_SIZE=32
SEED=0
AUDIO_LENGTH=10
EVAL_WORKERS=1

PROMPT_CSV="/share/users/student/f/friverossego/FlowSynth/audiocaps-test.csv"
OUT_ROOT="/share/users/student/f/friverossego/FlowSynth/samples_audio_final_per_model/audiogen"
FLOWSYNTH_ROOT="/share/users/student/f/friverossego/FlowSynth"
MODEL_ID="facebook/audiogen-medium"

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
conda activate audiogen_env

PYTHON_BIN="$(command -v python)"
if [[ -z "${PYTHON_BIN}" ]]; then
    echo "python not found after conda activation."
    exit 1
fi
echo "Using python: ${PYTHON_BIN}"
"${PYTHON_BIN}" -c "import torch; print('torch', torch.__version__)"

if ! conda env list | awk '{print $1}' | grep -qx "jamendo"; then
    echo "conda env 'jamendo' not found; required for evaluation."
    exit 1
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    MASTER_PORT_BASE=$((10000 + SLURM_JOB_ID % 20000))
else
    MASTER_PORT_BASE=29500
fi

mkdir -p "$OUT_ROOT"
RUN_INDEX=0
CFG=3
OUTDIR_CFG3="${OUT_ROOT}/cfg${CFG}"
PORT=$((MASTER_PORT_BASE + RUN_INDEX))
RUN_INDEX=$((RUN_INDEX + 1))

mkdir -p "$OUTDIR_CFG3"
cd "$FLOWSYNTH_ROOT"

echo "=== Running AudioGen cfg${CFG} (resume enabled) ==="
"${PYTHON_BIN}" -m torch.distributed.run --nproc_per_node="${NUM_GPUS}" --master_port "${PORT}" sample_audiogen.py \
    --prompt-csv "$PROMPT_CSV" \
    --output-dir "$OUTDIR_CFG3" \
    --model-id "$MODEL_ID" \
    --guidance-scale "$CFG" \
    --audio-length "$AUDIO_LENGTH" \
    --batch-size "$BATCH_SIZE" \
    --seed "$SEED" \
    --resume

echo "=== Switching to jamendo for evaluation ==="
conda activate jamendo
EVAL_PYTHON="$(command -v python)"
if [[ -z "${EVAL_PYTHON}" ]]; then
    echo "python not found after activating jamendo."
    exit 1
fi
echo "Using eval python: ${EVAL_PYTHON}"

for cfg in 1 3; do
    outdir="${OUT_ROOT}/cfg${cfg}"
    echo "=== Evaluating AudioGen cfg${cfg} ==="
    "${EVAL_PYTHON}" evaluate.py --gen "$outdir" --workers "$EVAL_WORKERS"
done

echo "Done. Outputs and metrics at: $OUT_ROOT"
