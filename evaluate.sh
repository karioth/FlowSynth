#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=H100.80gb:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=350G
#SBATCH --time=48:00:00
#SBATCH --job-name=flowsynth_eval_all
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

set -e

FLOWSYNTH_ROOT="/share/users/student/f/friverossego/FlowSynth"
GEN_ROOT="/share/users/student/f/friverossego/FlowSynth/samples_audio_final_per_model"

module purge 2>/dev/null || true
spack load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jamendo

cd "$FLOWSYNTH_ROOT"

for gen_dir in $(find "$GEN_ROOT" -mindepth 3 -maxdepth 3 -type d -name 'cfg*' | sort); do
    echo "Evaluating: $gen_dir"
    python evaluate.py --gen "$gen_dir" "$@"
done

echo "Done. Evaluated all cfg folders under: $GEN_ROOT"
