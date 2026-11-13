#!/bin/bash
#SBATCH --job-name=sj_rc_v9_sizes
#SBATCH --partition=jobs
#SBATCH --array=0-3%2
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
sizes=(40 80 40 80)
minutes=(500 500 700 700)
size=${sizes[$SLURM_ARRAY_TASK_ID]}
cost=${minutes[$SLURM_ARRAY_TASK_ID]}
run() { srun singularity exec --nv --bind /scratch:/scratch "$IMAGE" python -u "$@"; }
run -c 'import json,hashlib,pathlib; root=pathlib.Path("."); m=json.load(open("bundle_hashes.json")); assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in m.items()); print("Bundle verified")'
cd "train${size}"
out="runs/vehicle${cost}"
args=()
if [[ -f "$out/protocol.json" ]]; then args+=(--resume); fi
run -m experiments.concurrent_fleet_v9.train_vehicle_sweep --out "$out" --weights 10 --vehicle-minutes "$cost" --max-hours 10 "${args[@]}"
stage=$(run -c 'import json,sys; print(json.load(open(sys.argv[1]))["stage"])' "$out/status.json")
if [[ "$stage" == needs_resume ]]; then
    scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
elif [[ "$stage" != complete && "$stage" != paused ]]; then
    echo "Unexpected stage: $stage" >&2
    exit 1
fi
