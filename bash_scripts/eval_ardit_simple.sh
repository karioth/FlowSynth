#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --time=10:00:00
#SBATCH --job-name=sample_arditsimple
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
OUT_ROOT="/share/users/student/f/friverossego/FlowSynth/samples_ardit_simple_exp"

FLOWSYNTH_ROOT="/share/users/student/f/friverossego/FlowSynth"
AR_DIT_CKPT="/share/users/student/f/friverossego/FlowSynth/audio_logs/AUDIO_AR_DiT_Simple_B/checkpoints/last.ckpt"

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

cd "$FLOWSYNTH_ROOT"

CFGS=(1 3 8)
ARDIFF_STEP=1
RUN_INDEX=0

for CFG in "${CFGS[@]}"; do
    TAG="cfg${CFG}_ad${ARDIFF_STEP}"
    OUTDIR="${OUT_ROOT}/${TAG}"
    MASTER_PORT=$((MASTER_PORT_BASE + RUN_INDEX))
    RUN_INDEX=$((RUN_INDEX + 1))

    echo "=== Evaluating: $TAG ==="
    python evaluate.py --gen "$OUTDIR" --workers 1
done

echo "Done. Outputs at: $OUT_ROOT"
