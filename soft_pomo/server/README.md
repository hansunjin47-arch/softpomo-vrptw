# RL-only server bundle

This bundle trains Pure RL, Kim1000, and Pure500 + cached Phase2 on train80.
It never generates confidence, calls an API, evaluates test80, or runs LKH.
Existing model/environment/gradient code is reused. The server entry point
blocks network connections and fails on missing or invalid caches.
Some modules with LLM in their name remain import dependencies; do not launch
their original CLI entry points on the server. No API keys are needed.

## Contents and assumptions

- Only train80 instance files and observation sidecars are included.
- The split lists test membership for provenance, but no test data are uploaded.
- Kim cache includes base and accident-stage files; membership is not regenerated.
- Phase2 uses the bundled caches and their matching trusted Pure500 checkpoint.
- Default bundle: original prompt, beta=5, first-customer-match gating OFF.
- A different Pure500 seed needs freshly matched reference caches generated locally.
- Hardware/software changes do not guarantee identical historical results.
- Dependencies: Python >=3.10, PyTorch and NumPy, supplied by the shared container.
- No system installation, sudo, shared-image modification, or local .venv transfer.

## Upload (Windows PowerShell, NOT inside the SSH session)

Use the actual archive path reported by build_bundle.py:

```powershell
scp -P 51014 server_export/soft_pomo_rl_train80.zip byk_postech@210.125.69.5:/scratch/byk_postech/sunjinhan/
```

## Extract (server SSH terminal)

Do not overwrite an existing bundle or another user's directory.

```bash
cd /scratch/byk_postech/sunjinhan
unzip soft_pomo_rl_train80.zip
cd soft_pomo_rl_train80
```

The partition and image come from the supplied administrator guide; if Slurm
requires an account/QOS or different partition, ask the administrator for it.

## GPU smoke tests

Submit ONE job at a time; one person may use only one GPU concurrently.
Wait until each job finishes before submitting the next. Smoke runs use batch1
and both rain/accident cases, including backward and checkpoint save/reload.
Smoke results are not scientific evaluation results or resumable training runs.

```bash
sbatch --job-name=sunjinhan_pure_smoke --time=00:15:00 server/train.slurm --method pure --smoke --out runs/pure_smoke
sbatch --job-name=sunjinhan_kim_smoke --time=00:15:00 server/train.slurm --method kim --smoke --out runs/kim_smoke
sbatch --job-name=sunjinhan_phase2_smoke --time=00:15:00 server/train.slurm --method phase2 --smoke --out runs/phase2_smoke
```

Batch1 smoke does not establish batch64 memory capacity or throughput. Inspect
the first full training epoch, GPU memory and epoch duration before scaling up.
For a repeat smoke test choose a NEW output directory.
After batch1 passes, repeat with --smoke-batch-size 64 and a fresh output path
to check the actual training batch memory requirement without a full epoch.

## Training (choose ONE)

```bash
sbatch --job-name=sunjinhan_pure1000 server/train.slurm --method pure --out runs/pure1000
sbatch --job-name=sunjinhan_kim1000 server/train.slurm --method kim --out runs/kim1000
sbatch --job-name=sunjinhan_phase2 server/train.slurm --method phase2 --out runs/phase2
```

Pure/Kim start at epoch1; Phase2 starts at epoch501. Final epoch defaults to1000.
Use --epochs 500 for a Pure500-only run. --seed controls training randomness,
not the provenance of the bundled reference checkpoint/caches.

The runner saves checkpoint-last.pt atomically every10 epochs, preserves
checkpoint-500.pt and checkpoint-1000.pt, and stops at an epoch boundary near
11.5 hours or after a Slurm stop signal. Signals cannot guarantee saving before
a hard kill; worst-case progress since the last periodic save can be lost.
Check status.json: needs_resume means submit the same command with --resume.
No automatic requeue or second job submission occurs.

```bash
sbatch --job-name=sunjinhan_phase2 server/train.slurm --method phase2 --out runs/phase2 --resume
squeue -u byk_postech
```

Resume requires unchanged config and bundle. It restores model, Adam, RNG states
and AMP scaler where applicable. Legacy Pure500 has no full RNG/scaler state;
initial Phase2 branching follows the historical model+optimizer transfer.
All runs are training-only; no evaluation-induced RNG consumption occurs.
The shared account shows other people's jobs: never cancel them.
Keep logs/checkpoints in your directory and download final results for backup.

## Local checks

Run on the EXTRACTED bundle with the existing local PyTorch interpreter:

```text
python server/train_cached.py --method phase2 --check-only --cpu --out runs/check_phase2
python server/train_cached.py --method phase2 --smoke --cpu --out runs/cpu_smoke_phase2
```

These checks do not validate the server's driver, Singularity binding, A100
allocation, PyTorch build, or batch64 GPU memory. Those need actual server runs.
