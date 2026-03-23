#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --job-name=cfg_sweep_selected
#SBATCH -x klpsy-1
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err


PROMPT_CSV="/share/users/student/f/friverossego/FlowSynth/audiocaps-test.csv"
FLOWSYNTH_ROOT="/share/users/student/f/friverossego/FlowSynth"
AUDIOMAR_ROOT="/share/users/student/f/friverossego/audio_mar"
OUT_ROOT="/share/users/student/f/friverossego/FlowSynth/samples_cfg_sweep"

TRANSFORMER_CKPT="/share/users/student/f/friverossego/FlowSynth/audio_logs/AUDIO_NTP_Transformer_B_125e/checkpoints/last.ckpt"
DIT_CKPT="/share/users/student/f/friverossego/FlowSynth/audio_logs/AUDIO_NTP_DiT_B_sharedtime_125e/checkpoints/last.ckpt"
AUDIONTP_CKPT="/share/users/student/f/friverossego/audio_mar/logs_audiontp/checkpoint-last.pth"
MAR_CKPT="/share/users/student/f/friverossego/audio_mar/logs/checkpoint-last.pth"

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

STEPS=100
SEED=0
PRECISION="bf16-mixed"
FLOWSYNTH_BATCH_SIZE=32
AUDIOMAR_BATCH_SIZE=32
EVAL_WORKERS=1
CFGS=(4 5 6 7 8)

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

mkdir -p "$OUT_ROOT"

run_flowsynth () {
    local tag="$1"
    local ckpt="$2"
    local cfg="$3"
    local outdir="${OUT_ROOT}/flowsynth/${tag}/cfg${cfg}"

    mkdir -p "$outdir"
    cd "$FLOWSYNTH_ROOT"

    echo "=== Sampling FlowSynth ${tag} cfg${cfg} ==="
    "${PYTHON_BIN}" sample.py \
        --checkpoint "$ckpt" \
        --prompt-csv "$PROMPT_CSV" \
        --cfg-scale "$cfg" \
        --num-inference-steps "$STEPS" \
        --batch-size "$FLOWSYNTH_BATCH_SIZE" \
        --precision "$PRECISION" \
        --seed "$SEED" \
        --output-dir "$outdir"

    echo "=== Evaluating FlowSynth ${tag} cfg${cfg} ==="
    "${PYTHON_BIN}" evaluate.py --gen "$outdir" --workers "$EVAL_WORKERS"
}

run_audio_mar () {
    local tag="$1"
    local ckpt="$2"
    local cfg="$3"
    local outdir="${OUT_ROOT}/audio_mar/${tag}/cfg${cfg}"

    mkdir -p "$outdir"
    cd "$AUDIOMAR_ROOT"

    echo "=== Sampling audio_mar ${tag} cfg${cfg} ==="
    "${PYTHON_BIN}" sample.py \
        --checkpoint "$ckpt" \
        --prompt-csv "$PROMPT_CSV" \
        --cfg-scale "$cfg" \
        --cfg-schedule constant \
        --num-diffusion-steps "$STEPS" \
        --batch-size "$AUDIOMAR_BATCH_SIZE" \
        --precision "$PRECISION" \
        --seed "$SEED" \
        --output-dir "$outdir"

    echo "=== Evaluating audio_mar ${tag} cfg${cfg} ==="
    cd "$FLOWSYNTH_ROOT"
    "${PYTHON_BIN}" evaluate.py --gen "$outdir" --workers "$EVAL_WORKERS"
}

for cfg in "${CFGS[@]}"; do
    run_flowsynth "AUDIO_NTP_Transformer_B_125e" "$TRANSFORMER_CKPT" "$cfg"
done

for cfg in "${CFGS[@]}"; do
    run_flowsynth "AUDIO_NTP_DiT_B_sharedtime_125e" "$DIT_CKPT" "$cfg"
done

for cfg in "${CFGS[@]}"; do
    run_audio_mar "audiontp" "$AUDIONTP_CKPT" "$cfg"
done

for cfg in "${CFGS[@]}"; do
    run_audio_mar "mar_regular" "$MAR_CKPT" "$cfg"
done

echo "Done. Outputs at: $OUT_ROOT"
