#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -f submitted_jobs.tsv ]]; then
    echo 'Already submitted. No duplicate jobs created.'
    cat submitted_jobs.tsv
    exit 0
fi
mkdir .submission_lock || { echo 'Previous submission needs inspection; do not release held jobs blindly.' >&2; exit 1; }
[[ $(wc -l < tasks.txt) -eq 15 ]]
# Only this experiment family's jobs, not other users sharing this account.
deps=$(squeue -h -r -u "$USER" -o '%F %j' | awk '$2 ~ /^sj_kim_/ {print $1}' | sort -u | paste -sd: -)
printf 'Existing Kim dependencies: %s\n' "${deps:-none}"
declare -a jobs
: > submission_pending.tsv
for index in $(seq 0 14); do
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
# Record the complete dependency graph before any task can start.
mv submission_pending.tsv submitted_jobs.tsv
for job in "${jobs[@]}"; do scontrol release "$job"; done
touch submission_released.ok
cat submitted_jobs.tsv
echo '15 non-array jobs submitted in 2 serial chains, 1 GPU per job; maximum 2 GPUs for this experiment.'
squeue -u "$USER" -o '%.18i %.24j %.10T %.40E'
