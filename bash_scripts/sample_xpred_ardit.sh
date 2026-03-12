#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=H100.80gb:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=350G
#SBATCH --time=48:00:00
#SBATCH --job-name=flowsynth_sample_xpred_ardit
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

set -e

NUM_GPUS=2
STEPS=100
SEED=0
BATCH_SIZE=16
PRECISION=bf16-mixed

PROMPT_CSV="/share/users/student/f/friverossego/FlowSynth/audiocaps-test.csv"
OUT_ROOT="/share/users/student/f/friverossego/FlowSynth/samples_audio_final_per_model/flowsynth"

FLOWSYNTH_ROOT="/share/users/student/f/friverossego/FlowSynth"

AR_DIT_XPRED_CKPT="/share/users/student/f/friverossego/FlowSynth/audio_logs/AUDIO_AR_DiT_B_125e_xpred/checkpoints/last.ckpt"

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

run_flowsynth () {
    local tag="$1"
    local ckpt="$2"
    local cfg="$3"
    local port=$((MASTER_PORT_BASE + RUN_INDEX))
    RUN_INDEX=$((RUN_INDEX + 1))
    local outdir="${OUT_ROOT}/${tag}/cfg${cfg}"
    mkdir -p "$outdir"

    cd "$FLOWSYNTH_ROOT"
    torchrun --nproc_per_node="${NUM_GPUS}" --master_port "${port}" sample.py \
        --checkpoint "$ckpt" \
        --prompt-csv "$PROMPT_CSV" \
        --cfg-scale "$cfg" \
        --num-inference-steps "$STEPS" \
        --batch-size "$BATCH_SIZE" \
        --precision "$PRECISION" \
        --seed "$SEED" \
        --output-dir "$outdir"
}

for cfg in 1 3; do
    run_flowsynth "AUDIO_AR_DiT_B_125e_xpred" "$AR_DIT_XPRED_CKPT" "$cfg"
done

cd "$FLOWSYNTH_ROOT"
for cfg in 1 3; do
    gen_dir="${OUT_ROOT}/AUDIO_AR_DiT_B_125e_xpred/cfg${cfg}"
    echo "Evaluating: $gen_dir"
    python evaluate.py --gen "$gen_dir"
done

echo "Done. Outputs at: $OUT_ROOT"
