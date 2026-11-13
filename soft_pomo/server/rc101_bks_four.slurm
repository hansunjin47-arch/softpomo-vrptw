#!/bin/bash
#SBATCH --job-name=sj_rc101_bks
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
[[ "$INDEX" =~ ^[0-2]$ ]]
CASES=(train20_pure train20_kim train80_pure)
CASE="${CASES[$INDEX]}"
METHOD="${CASE##*_}"
IMAGE=/scratch/byk_postech/shared/singularity/pytorch_24.12.sif
export OMP_NUM_THREADS=4 MPLBACKEND=Agg
unset OPENROUTER_API_KEY OPENAI_API_KEY ANTHROPIC_API_KEY
run() { srun singularity exec --nv --bind /scratch:/scratch "$IMAGE" python -u "$@"; }
ARGS=()
if [[ -f "$CASE/runs/$METHOD/checkpoint-last.pt" ]]; then
    ARGS+=(--resume)
else
    run "$CASE/server/train_cached.py" --method "$METHOD" --smoke --smoke-batch-size 1 --out "$CASE/runs/smoke_${SLURM_JOB_ID}"
fi
run "$CASE/server/train_cached.py" --method "$METHOD" --epochs 1000 --seed 42 --save-every 500 --max-hours 11 "${ARGS[@]}"
STATE=$(python3 -c "import json; print(json.load(open('$CASE/runs/$METHOD/status.json'))['status'])")
if [[ "$STATE" == needs_resume ]]; then
    sbatch --dependency=afterok:"$SLURM_JOB_ID" rc101_bks_four.slurm "$INDEX"
elif [[ "$STATE" == complete && "$INDEX" -lt 2 ]]; then
    sbatch --dependency=afterok:"$SLURM_JOB_ID" rc101_bks_four.slurm "$((INDEX+1))"
elif [[ "$STATE" == complete ]]; then
    echo 'All three RC101 BKS training runs complete.'
else
    echo "Unexpected state: $STATE" >&2
    exit 1
fi
