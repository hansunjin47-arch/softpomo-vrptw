#!/bin/bash
#SBATCH --job-name=sj_rc_v9_resume
#SBATCH --partition=jobs
#SBATCH --array=0-1%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=slurm-%x-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
IMAGE=/scratch/byk_postech/shared/singularity/pytorch_24.12.sif
export OMP_NUM_THREADS=4 MPLBACKEND=Agg
unset OPENROUTER_API_KEY OPENAI_API_KEY ANTHROPIC_API_KEY
arms=(entropy control)
arm=${arms[$SLURM_ARRAY_TASK_ID]}
cd "arms/$arm"
run() { srun singularity exec --nv --bind /scratch:/scratch "$IMAGE" python -u "$@"; }
run verify_resume.py
if [[ "$arm" == entropy ]]; then
    run -m experiments.concurrent_fleet_exploration.train --out runs/resumed --weights 10 --vehicle-minutes 700 --entropy-beta 0.01 --entropy-zero-epoch 500 --resume --max-hours 10
else
    run -m experiments.concurrent_fleet_stop_init.train --out runs/resumed --weights 10 --vehicle-minutes 700 --arm control --entropy-beta 0 --resume --max-hours 10
fi
stage=$(run -c 'import json; print(json.load(open("runs/resumed/status.json"))["stage"])')
if [[ "$stage" == needs_resume ]]; then
    scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
elif [[ "$stage" != complete && "$stage" != paused ]]; then
    echo "Unexpected stage: $stage" >&2
    exit 1
fi
