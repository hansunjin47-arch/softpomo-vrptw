#!/bin/bash
#SBATCH --job-name=sj_kim_tt
#SBATCH --partition=jobs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
VARIANT="${1:?Pass A or C}"
[[ "$VARIANT" == A || "$VARIANT" == C ]]
IMAGE=/scratch/byk_postech/shared/singularity/pytorch_24.12.sif
export OMP_NUM_THREADS=4 MPLBACKEND=Agg
unset OPENROUTER_API_KEY OPENAI_API_KEY ANTHROPIC_API_KEY
run() {
    srun singularity exec --nv --bind /scratch:/scratch "$IMAGE" python -u "$@"
}
if [[ ! -f "$VARIANT/runs/kim/checkpoint-last.pt" ]]; then
    run "$VARIANT/server/train_cached.py" --method kim --smoke --smoke-batch-size 128 --out "$VARIANT/runs/smoke_${SLURM_JOB_ID}"
    run "$VARIANT/server/train_cached.py" --method kim --epochs 1000 --seed 42 --max-hours 11
else
    run "$VARIANT/server/train_cached.py" --method kim --epochs 1000 --seed 42 --max-hours 11 --resume
fi
STATE=$(python3 -c "import json; print(json.load(open('$VARIANT/runs/kim/status.json'))['status'])")
if [[ "$STATE" == needs_resume ]]; then
    sbatch --dependency=afterok:"$SLURM_JOB_ID" kim_tt_ac.slurm "$VARIANT"
elif [[ "$STATE" == complete && "$VARIANT" == A ]]; then
    sbatch --dependency=afterok:"$SLURM_JOB_ID" kim_tt_ac.slurm C
elif [[ "$STATE" == complete ]]; then
    echo 'A and C training complete at epoch 1000.'
else
    echo "Unexpected state: $STATE" >&2
    exit 1
fi
