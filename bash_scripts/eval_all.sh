#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --time=36:00:00
#SBATCH --job-name=eval_all
#SBATCH -x klpsy-1
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

set -e

FLOWSYNTH_ROOT="/share/users/student/f/friverossego/FlowSynth"
SEARCH_ROOTS=(
    "/share/users/student/f/friverossego/FlowSynth/samples_baselines"
    "/share/users/student/f/friverossego/FlowSynth/samples_final"
)

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

cd "$FLOWSYNTH_ROOT"

mapfile -d '' CFG_DIRS < <(find "${SEARCH_ROOTS[@]}" -type d -name '*cfg*' -print0 | sort -z)

if [[ ${#CFG_DIRS[@]} -eq 0 ]]; then
    echo "No cfg directories found under: ${SEARCH_ROOTS[*]}"
    exit 1
fi

echo "Found ${#CFG_DIRS[@]} cfg directories to evaluate."

RUN_INDEX=0

for OUTDIR in "${CFG_DIRS[@]}"; do
    RUN_INDEX=$((RUN_INDEX + 1))
    REL_OUTDIR="${OUTDIR#$FLOWSYNTH_ROOT/}"

    echo "=== Evaluating (${RUN_INDEX}/${#CFG_DIRS[@]}): ${REL_OUTDIR} ==="
    python evaluate.py --gen "$OUTDIR" --workers 1 "$@"
done

echo "Done. Evaluated ${#CFG_DIRS[@]} directories."
