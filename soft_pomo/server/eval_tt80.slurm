#!/bin/bash
#SBATCH --job-name=sj_eval_tt80
#SBATCH --partition=jobs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=4 MPLBACKEND=Agg
unset OPENROUTER_API_KEY OPENAI_API_KEY ANTHROPIC_API_KEY
IMAGE=/scratch/byk_postech/shared/singularity/pytorch_24.12.sif
for variant in A C; do
    for epoch in 1000 2000; do
        srun singularity exec --nv --bind /scratch:/scratch "$IMAGE" python -u eval_pure_tt_bundle.py --bundle "$variant" --epoch "$epoch"
    done
done
