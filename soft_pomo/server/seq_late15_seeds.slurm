#!/bin/bash
#SBATCH --job-name=sj_rc_seq_l15
#SBATCH --partition=jobs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-4%1
#SBATCH --requeue
#SBATCH --output=slurm-%x-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
seed=$((42+SLURM_ARRAY_TASK_ID))
base=$PWD
bundle="$base/seed$seed"
IMAGE=/scratch/byk_postech/shared/singularity/pytorch_24.12.sif
export OMP_NUM_THREADS=4 MPLBACKEND=Agg
unset OPENROUTER_API_KEY OPENAI_API_KEY ANTHROPIC_API_KEY
run() { srun singularity exec --nv --bind /scratch:/scratch "$IMAGE" python -u "$@"; }
args=()
if [[ -f "$bundle/runs/pure/checkpoint-last.pt" ]]; then args+=(--resume); fi
run "$bundle/server/train_cached.py" --method pure --seed "$seed" --epochs 1000 --save-every 10 --max-hours 9 "${args[@]}"
stage=$(run -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$bundle/runs/pure/status.json")
if [[ "$stage" == needs_resume ]]; then
    scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
elif [[ "$stage" == complete ]]; then
    run "$base/eval_pure_tt_bundle.py" --bundle "$bundle" --epoch 1000
else
    echo "Unexpected stage: $stage" >&2
    exit 1
fi
