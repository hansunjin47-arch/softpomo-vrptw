#!/bin/bash
#SBATCH --job-name=sj_bks_max2
#SBATCH --partition=jobs
#SBATCH --array=0-4%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=slurm-%x-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
OLD=rc101_bks_train40_80_late5_10_20261014
NEW=rc101_bks_train40_80_late20_20261015
CASES=("$OLD/train40_late10" "$NEW/train40_late20" "$OLD/train80_late5" "$OLD/train80_late10" "$NEW/train80_late20")
CASE="${CASES[${SLURM_ARRAY_TASK_ID:?}]}"
IMAGE=/scratch/byk_postech/shared/singularity/pytorch_24.12.sif
export OMP_NUM_THREADS=4 MPLBACKEND=Agg
unset OPENROUTER_API_KEY OPENAI_API_KEY ANTHROPIC_API_KEY
run() { srun singularity exec --nv --bind /scratch:/scratch "$IMAGE" python -u "$@"; }
STATE=$(python3 -c "import json,pathlib; p=pathlib.Path('$CASE/runs/pure/status.json'); print(json.loads(p.read_text()).get('status') if p.exists() else 'new')")
if [[ "$STATE" != complete ]]; then
    ARGS=()
    if [[ -f "$CASE/runs/pure/checkpoint-last.pt" ]]; then ARGS+=(--resume); fi
    run "$CASE/server/train_cached.py" --method pure --epochs 1000 --seed 42 --save-every 100 --max-hours 10 "${ARGS[@]}"
    STATE=$(python3 -c "import json; print(json.load(open('$CASE/runs/pure/status.json'))['status'])")
    if [[ "$STATE" == needs_resume ]]; then
        # Requeue this array element, preserving the array's concurrency limit.
        scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
        exit 0
    fi
    [[ "$STATE" == complete ]]
fi
run "$(dirname "$CASE")/eval_pure_tt_bundle.py" --bundle "$CASE" --epoch 1000
echo "COMPLETE: $CASE training and test80 evaluation"
