#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
previous=${1:?Pass the vehicle300 job ID}
[[ "$previous" =~ ^[0-9]+$ ]] || { echo 'Invalid job ID' >&2; exit 2; }
if [[ -e submitted_chain.tsv ]]; then
    echo 'Chain already submitted or partially submitted. Inspect submitted_chain.tsv before retrying.' >&2
    exit 1
fi
for vehicle in 100 200 400 500; do
    job=$(sbatch --parsable --job-name="sj_rc_v9_be$vehicle" --dependency="afterok:$previous" v9_bias_entropy_sweep.slurm "$vehicle")
    job=${job%%;*}
    printf '%s\t%s\tafterok:%s\n' "$vehicle" "$job" "$previous" | tee -a submitted_chain.tsv
    previous=$job
done
