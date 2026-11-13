#!/bin/bash
# Repair only this Kim array; leave unrelated jobs untouched.
set -euo pipefail
array=${1:-231538}
[[ "$array" =~ ^[0-9]+$ ]] || exit 2
snapshot() { squeue -r -h -j "$array" -o '%i|%T' | sort -t_ -k2,2n; }
echo 'BEFORE'
scontrol show job "$array" || true
snapshot
scontrol update JobId="$array" ArrayTaskThrottle=2
while IFS='|' read -r job state; do
    [[ "$job" =~ ^${array}_[0-9]+$ ]] || { echo "Unexpected task: $job"; exit 1; }
    if [[ "$state" == PENDING ]]; then scontrol hold "$job"; fi
done < <(snapshot)
mapfile -t running < <(snapshot | awk -F'|' '$2=="RUNNING" {print $1}')
for ((i=2; i<${#running[@]}; i++)); do
    echo "Requeue with checkpoint resume: ${running[i]}"
    scontrol requeuehold "${running[i]}"
done
# Wait until excess allocations have actually stopped before releasing anything.
ready=0
for ((attempt=0; attempt<60; attempt++)); do
    rows=$(snapshot)
    active=$(printf '%s\n' "$rows" | awk -F'|' '$2!="PENDING" && NF==2 {n++} END {print n+0}')
    if ((active<=2)); then ready=1; break; fi
    sleep 2
done
((ready==1)) || { echo 'Tasks still stopping; pending tasks remain held. Inspect before retrying.'; exit 1; }
mapfile -t running < <(snapshot | awk -F'|' '$2=="RUNNING" {print $1}')
mapfile -t pending < <(snapshot | awk -F'|' '$2=="PENDING" {print $1}')
anchors=("${running[0]:-}" "${running[1]:-}")
# Each lane depends on its predecessor, independently of ArrayTaskThrottle.
lane=0
for job in "${pending[@]}"; do
    predecessor=${anchors[lane]}
    if [[ -n "$predecessor" ]]; then
        scontrol update JobId="$job" Dependency="afterany:$predecessor"
    else
        scontrol update JobId="$job" Dependency=
    fi
    echo "LANE $lane: $job waits for ${predecessor:-nothing}"
    anchors[lane]=$job
    lane=$((1-lane))
done
# No task is released unless every dependency update above succeeded.
for job in "${pending[@]}"; do scontrol release "$job"; done
echo 'AFTER: two dependency lanes, one GPU per task'
squeue -r -j "$array" -o '%.18i %.24j %.12T %.36E %R'
