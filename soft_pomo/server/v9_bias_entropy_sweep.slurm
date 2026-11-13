#!/bin/bash
#SBATCH --partition=jobs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=slurm-%x-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
vehicle=${1:?Expected vehicle cost}
case "$vehicle" in 100|200|400|500) ;; *) exit 2 ;; esac
IMAGE=/scratch/byk_postech/shared/singularity/pytorch_24.12.sif
export OMP_NUM_THREADS=4 MPLBACKEND=Agg
unset OPENROUTER_API_KEY OPENAI_API_KEY ANTHROPIC_API_KEY
run() { srun singularity exec --nv --bind /scratch:/scratch "$IMAGE" python -u "$@"; }
run verify_stop_init.py
out="runs/vehicle$vehicle"
args=()
if [[ -f "$out/protocol.json" ]]; then args+=(--resume); fi
run -m experiments.concurrent_fleet_stop_init.train --out "$out" --weights 10 --vehicle-minutes "$vehicle" --arm stop_bias --stop-target 0.15 --entropy-beta 1 --entropy-zero-epoch 500 --max-hours 10 "${args[@]}"
stage=$(run -c 'import json,sys; print(json.load(open(sys.argv[1]))["stage"])' "$out/status.json")
if [[ "$stage" == needs_resume ]]; then
    scontrol requeue "$SLURM_JOB_ID"
elif [[ "$stage" != complete ]]; then
    echo "Not complete: $stage; do not release dependent jobs" >&2
    exit 1
fi
