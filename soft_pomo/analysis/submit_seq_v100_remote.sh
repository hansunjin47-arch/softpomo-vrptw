#!/bin/bash
set -euo pipefail
cd /scratch/byk_postech/sunjinhan
name=seq_pure_v100_l5_rc_r_s42_46_20261028
sha256sum -c "$name.tar.gz.sha256"
if [[ -d "$name" ]]; then
    if [[ -f "$name/submission_released.ok" ]]; then
        echo 'Already submitted; leaving existing jobs unchanged.'
        cat "$name/submitted_jobs.tsv"
        exit 0
    fi
    echo 'Existing incomplete destination; inspect before retrying.' >&2
    exit 1
fi
tar -xzf "$name.tar.gz"
cd "$name"
bash submit.sh
