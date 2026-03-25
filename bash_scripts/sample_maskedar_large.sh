#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=24:00:00
#SBATCH --job-name=sample_maskedar
#SBATCH -x klpsy-1
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

set -e

NUM_GPUS=2
STEPS=100
SEED=0
BATCH_SIZE=64
PRECISION=bf16-mixed

PROMPT_CSV="/share/users/student/f/friverossego/FlowSynth/audiocaps-test.csv"
OUT_ROOT="/share/users/student/f/friverossego/FlowSynth/samples_maskedar_large_exp"

FLOWSYNTH_ROOT="/share/users/student/f/friverossego/FlowSynth"
MASKEDAR_CKPT="/share/users/student/f/friverossego/FlowSynth/audio_logs/AUDIO_MaskedAR_Medium_gated/checkpoints/last.ckpt"

SCRATCH_BASE="/share/users/student/f/friverossego/tmp"
export TMPDIR="${SLURM_TMPDIR:-$SCRATCH_BASE/tmp}"
export TORCHINDUCTOR_CACHE_DIR="$SCRATCH_BASE/torchinductor"
export TRITON_CACHE_DIR="$SCRATCH_BASE/triton"
export CUDA_CACHE_PATH="$SCRATCH_BASE/cuda"
export XDG_CACHE_HOME="$SCRATCH_BASE/xdg"
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

module purge 2>/dev/null || true
spack load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jamendo

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    MASTER_PORT_BASE=$((10000 + SLURM_JOB_ID % 20000))
else
    MASTER_PORT_BASE=29500
fi

mkdir -p "$OUT_ROOT"
RUN_INDEX=0

cd "$FLOWSYNTH_ROOT"

for cfg in 1 3 5; do
    tag="cfg${cfg}"
    outdir="${OUT_ROOT}/${tag}"
    mkdir -p "$outdir"

    port=$((MASTER_PORT_BASE + RUN_INDEX))
    RUN_INDEX=$((RUN_INDEX + 1))

    echo "=== Running: $tag ==="
    torchrun --nproc_per_node="${NUM_GPUS}" --master_port "${port}" sample.py \
        --checkpoint "$MASKEDAR_CKPT" \
        --prompt-csv "$PROMPT_CSV" \
        --cfg-scale "$cfg" \
        --num-inference-steps "$STEPS" \
        --batch-size "$BATCH_SIZE" \
        --precision "$PRECISION" \
        --seed "$SEED" \
        --output-dir "$outdir"

    echo "=== Evaluating: $tag ==="
    python evaluate.py --gen "$outdir" --workers 1
done

echo "Done. Outputs at: $OUT_ROOT"
