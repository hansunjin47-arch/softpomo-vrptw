#!/bin/bash
set -euo pipefail
cd /scratch/byk_postech/sunjinhan
name=kim_l5_rc_r_c_s42_46_20261026
sha256sum -c "$name.tar.gz.sha256"
if [[ -e "$name/submitted_job.txt" ]]; then
    echo "Already submitted: $(cat "$name/submitted_job.txt")"
    exit 0
fi
tar -xzf "$name.tar.gz"
cd "$name"
bash submit.sh
