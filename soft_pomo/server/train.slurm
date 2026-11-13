#!/bin/bash
#SBATCH --job-name=sunjinhan_rl
#SBATCH --partition=jobs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --signal=USR1@300
#SBATCH --output=slurm-%x-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the extracted bundle directory}"
IMAGE=/scratch/byk_postech/shared/singularity/pytorch_24.12.sif
test -f "$IMAGE"
test -f server/train_cached.py
export OMP_NUM_THREADS=4
export MPLBACKEND=Agg
unset OPENROUTER_API_KEY OPENAI_API_KEY ANTHROPIC_API_KEY
srun singularity exec --nv --bind /scratch:/scratch "$IMAGE" \
    python -u server/train_cached.py "$@"
