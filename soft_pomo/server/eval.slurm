#!/bin/bash
#SBATCH --job-name=sunjinhan_pure_eval
#SBATCH --partition=jobs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the training bundle directory}"
IMAGE=/scratch/byk_postech/shared/singularity/pytorch_24.12.sif
test -f "$IMAGE"
test -f eval_addon/eval_pure.py
export OMP_NUM_THREADS=4
export MPLBACKEND=Agg
unset OPENROUTER_API_KEY OPENAI_API_KEY ANTHROPIC_API_KEY
srun singularity exec --nv --bind /scratch:/scratch "$IMAGE" \
    python -u eval_addon/eval_pure.py "$@"
