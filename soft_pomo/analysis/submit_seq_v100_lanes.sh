#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -f submission_released.ok ]]; then
    echo 'Already submitted; no duplicate jobs created.'
    cat submitted_jobs.tsv
    exit 0
fi
mkdir .submission_lock || { echo 'Submission already attempted; inspect held jobs before retrying.' >&2; exit 1; }
[[ $(wc -l < tasks.txt) -eq 10 ]]
# sj_* identifies these experiments; exclude other users sharing the account.
deps=$(squeue -h -r -u "$USER" -o '%i %j' | awk '$2 ~ /^sj_/ {print $1}' | sort -u | paste -sd: -)
printf 'Wait for existing experiment jobs: %s\n' "${deps:-none}"
declare -a jobs
: > submission_pending.tsv
for index in $(seq 0 9); do
    prior="$deps"
    if (( index >= 2 )); then prior="${jobs[index-2]}"; fi
    args=(--hold)
    if [[ -n "$prior" ]]; then args+=(--dependency="afterany:$prior"); fi
    job=$(sbatch --parsable "${args[@]}" task.slurm "$index")
    job=${job%%;*}
    [[ "$job" =~ ^[0-9]+$ ]]
    jobs[index]="$job"
    relative=$(sed -n "$((index+1))p" tasks.txt)
    printf '%s\t%s\t%s\t%s\n' "$index" "$job" "$relative" "${prior:-none}" >> submission_pending.tsv
done
mv submission_pending.tsv submitted_jobs.tsv
for job in "${jobs[@]}"; do scontrol release "$job"; done
touch submission_released.ok
cat submitted_jobs.tsv
echo '10 jobs in two serial chains; 1 GPU each, at most 2 GPUs, after existing experiment jobs.'
squeue -u "$USER" -o '%.18i %.24j %.10T %.40E'
