#!/usr/bin/env bash
#SBATCH -J cache_text_wavcaps_freesound
#SBATCH -p workq
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=900G
#SBATCH --time=6:00:00
#SBATCH -o slurm_logs/%x_%j.batch.out
#SBATCH -e slurm_logs/%x_%j.batch.err

PROJECT_ROOT="/share/users/student/f/friverossego/LatentLM"

module purge 2>/dev/null || true

SCRATCH_BASE="/share/users/student/f/friverossego/tmp"
export TMPDIR="${SLURM_TMPDIR:-$SCRATCH_BASE/tmp}"
export TORCHINDUCTOR_CACHE_DIR="$SCRATCH_BASE/torchinductor"
export TRITON_CACHE_DIR="$SCRATCH_BASE/triton"
export CUDA_CACHE_PATH="$SCRATCH_BASE/cuda"
export XDG_CACHE_HOME="$SCRATCH_BASE/xdg"

mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME"

cd "$PROJECT_ROOT"
spack load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
eval "$(conda shell.bash hook)"
conda activate jamendo

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
export PYTHONWARNINGS=ignore

# -----------------------------
# User config
# -----------------------------
OUT_ROOT="/share/users/student/f/friverossego/datasets/WavCaps"
SUBSET="FreeSound"

WORKERS=28
WORKER_THREADS=4

export OMP_NUM_THREADS=$WORKER_THREADS
export MKL_NUM_THREADS=$WORKER_THREADS
export OPENBLAS_NUM_THREADS=$WORKER_THREADS
export NUMEXPR_NUM_THREADS=$WORKER_THREADS


mkdir -p slurm_logs

MANIFEST="${OUT_ROOT}/${SUBSET}/manifest.jsonl"
OUT="${OUT_ROOT}/${SUBSET}/text_embeddings"

echo "============================================================"
echo "[subset] ${SUBSET}"
echo "[manifest] ${MANIFEST}"
echo "[out] ${OUT}"
echo "============================================================"

mkdir -p "$OUT"

srun --cpu-bind=cores --kill-on-bad-exit=1 \
  --output="slurm_logs/%x_%j_${SUBSET}_%t.out" \
  --error="slurm_logs/%x_%j_${SUBSET}_%t.err" \
  python -m src.data_utils.cache_text_embeddings_cpu_pool \
    --metadata_path "$MANIFEST" \
    --output_dir "$OUT" \
    --device cpu \
    --num_workers "$WORKERS" \
    --worker_threads "$WORKER_THREADS"

echo "[done]"
