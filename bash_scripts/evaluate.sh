#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=48:00:00
#SBATCH --job-name=flowsynth_eval_all
#SBATCH -o %x_%j.out
#SBATCH -x klpsy-1
#SBATCH -e %x_%j.err

set -e

FLOWSYNTH_ROOT="/share/users/student/f/friverossego/FlowSynth"
GEN_ROOT="/share/users/student/f/friverossego/FlowSynth/samples_audio_final_per_model"
MODEL_DIRS=(
    "$GEN_ROOT/audioldm2"
    "$GEN_ROOT/stable_audio_open"
    "$GEN_ROOT/tangoflux"
)

module purge 2>/dev/null || true
spack load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jamendo

cd "$FLOWSYNTH_ROOT"

for cfg_dir in $(find "${MODEL_DIRS[0]}" -mindepth 1 -maxdepth 1 -type d -name 'cfg*' | sort); do
    cfg_name="$(basename "$cfg_dir")"
    for model_dir in "${MODEL_DIRS[@]}"; do
        gen_dir="$model_dir/$cfg_name"
        echo "Evaluating: $gen_dir"
        python evaluate.py --gen "$gen_dir" "$@"
    done
done

echo "Done. Evaluated cfg folders across: ${MODEL_DIRS[*]}"
