#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=H100.80gb:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=48:00:00
#SBATCH --job-name=sample_ardit_210k
#SBATCH -x klpsy-1
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

set -e

NUM_GPUS=2
STEPS=100
SEED=0
BATCH_SIZE=64
PRECISION=bf16-mixed
CFGS=(3 4 5 6 7 8)
ARDIFF_STEPS=(0 1 5 10 25 50 100)

PROMPT_CSV="/share/users/student/f/friverossego/FlowSynth/audiocaps-test.csv"
OUT_ROOT="/share/users/student/f/friverossego/FlowSynth/samples_ardit_medium_sweep_exp/"

FLOWSYNTH_ROOT="/share/users/student/f/friverossego/FlowSynth"
AR_DIT_CKPT="/share/users/student/f/friverossego/FlowSynth/audio_logs/AUDIO_AR_DiT_Medium_gated/checkpoints/step=0210000.ckpt"

SCRATCH_BASE="/share/users/student/f/friverossego/tmp"
export TMPDIR="${SLURM_TMPDIR:-$SCRATCH_BASE/tmp}"
export TORCHINDUCTOR_CACHE_DIR="$SCRATCH_BASE/torchinductor"
export TRITON_CACHE_DIR="$SCRATCH_BASE/triton"
export CUDA_CACHE_PATH="$SCRATCH_BASE/cuda"
export XDG_CACHE_HOME="$SCRATCH_BASE/xdg"
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

module purge 2>/dev/null || true
spack load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jamendo

PYTHON_BIN="$(command -v python)"
if [[ -z "${PYTHON_BIN}" ]]; then
    echo "python not found after conda activation."
    exit 1
fi
echo "Using python: ${PYTHON_BIN}"
"${PYTHON_BIN}" -c "import torch; print('torch', torch.__version__)"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    MASTER_PORT_BASE=$((10000 + SLURM_JOB_ID % 20000))
else
    MASTER_PORT_BASE=29500
fi

mkdir -p "$OUT_ROOT"
RUN_INDEX=0

cd "$FLOWSYNTH_ROOT"

if [[ ! -f "$AR_DIT_CKPT" ]]; then
    echo "Checkpoint not found: $AR_DIT_CKPT"
    exit 1
fi

ckpt_name="$(basename "$AR_DIT_CKPT")"

for cfg in "${CFGS[@]}"; do
    for ardiff_step in "${ARDIFF_STEPS[@]}"; do
        tag="cfg${cfg}_ardiff${ardiff_step}"
        outdir="${OUT_ROOT}/${tag}"
        mkdir -p "$outdir"

        port=$((MASTER_PORT_BASE + RUN_INDEX))
        RUN_INDEX=$((RUN_INDEX + 1))

        echo "=== Running: ${tag} (${ckpt_name}) ==="
        torchrun --nproc_per_node="${NUM_GPUS}" --master_port "${port}" sample.py \
            --checkpoint "$AR_DIT_CKPT" \
            --prompt-csv "$PROMPT_CSV" \
            --cfg-scale "$cfg" \
            --ardiff-step "$ardiff_step" \
            --num-inference-steps "$STEPS" \
            --batch-size "$BATCH_SIZE" \
            --precision "$PRECISION" \
            --seed "$SEED" \
            --output-dir "$outdir"

        echo "=== Evaluating: ${tag} ==="
        "${PYTHON_BIN}" evaluate.py --gen "$outdir" --workers 1
    done
done

echo "Done. Outputs at: $OUT_ROOT"
