#!/bin/bash
set -euo pipefail
cd /scratch/byk_postech/sunjinhan
name=kim_rl_example_l5_rc_r_c_s42_46_20261027
sha256sum -c "$name.tar.gz.sha256"
if [[ -d "$name" ]]; then
    if [[ -f "$name/submission_released.ok" ]]; then
        echo 'Already submitted; leaving existing jobs and files unchanged.'
        cat "$name/submitted_jobs.tsv"
        exit 0
    fi
    echo 'Destination already exists without a completed submission. Inspect before retrying.' >&2
    exit 1
fi
tar -xzf "$name.tar.gz"
cd "$name"
bash submit.sh
