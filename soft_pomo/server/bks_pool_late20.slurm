#!/bin/bash
#SBATCH --job-name=sj_bks_pool20
#SBATCH --partition=jobs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
INDEX="${1:-0}"
STAGE="${2:-train}"
[[ "$INDEX" =~ ^[0-1]$ ]]
[[ "$STAGE" == train || "$STAGE" == eval ]]
CASES=(train40_late20 train80_late20)
CASE="${CASES[$INDEX]}"
IMAGE=/scratch/byk_postech/shared/singularity/pytorch_24.12.sif
export OMP_NUM_THREADS=4 MPLBACKEND=Agg
unset OPENROUTER_API_KEY OPENAI_API_KEY ANTHROPIC_API_KEY
run() { srun singularity exec --nv --bind /scratch:/scratch "$IMAGE" python -u "$@"; }
if [[ "$STAGE" == train ]]; then
    ARGS=()
    if [[ -f "$CASE/runs/pure/checkpoint-last.pt" ]]; then ARGS+=(--resume); fi
    run "$CASE/server/train_cached.py" --method pure --epochs 1000 --seed 42 --save-every 500 --max-hours 11 "${ARGS[@]}"
    STATE=$(python3 -c "import json; print(json.load(open('$CASE/runs/pure/status.json'))['status'])")
    if [[ "$STATE" == needs_resume ]]; then
        sbatch --dependency=afterok:"$SLURM_JOB_ID" bks_pool_late20.slurm "$INDEX" train
    elif [[ "$STATE" == complete ]]; then
        sbatch --dependency=afterok:"$SLURM_JOB_ID" bks_pool_late20.slurm "$INDEX" eval
    else
        echo "Unexpected training state: $STATE" >&2
        exit 1
    fi
else
    run eval_pure_tt_bundle.py --bundle "$CASE" --epoch 1000
    if [[ "$INDEX" -lt 1 ]]; then
        sbatch --dependency=afterok:"$SLURM_JOB_ID" bks_pool_late20.slurm 1 train
    else
        echo 'Both lambda20 training and test80 evaluations complete.'
    fi
fi
