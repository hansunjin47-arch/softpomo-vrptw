# SoftPOMO Code-Only Snapshot

This branch transfers source code from the research PC. It does not contain
datasets, generated LLM caches, checkpoints, experiment outputs, API keys,
or the old repository's artifact history. Existing master is unchanged.

## Setup on another PC

Use Python 3.12. From the cloned repository directory on Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r soft_pomo/requirements-code.txt
cd soft_pomo
```

Install PyTorch separately using the build appropriate for that PC's GPU and
driver. Record Python, torch, CUDA and GPU versions with every experiment.
The dependency list is a starting point for core training and analysis, not
a fully pinned environment for every historical script. LKH-3 is a separate
executable and is not included here.

Set OPENROUTER_API_KEY in the new PC's environment before explicitly starting
paid cache generation or online evaluation. Never commit credentials.

## Current code paths

- `experiments/concurrent_fleet_pair/`: concurrent routing baseline and tests.
- `experiments/concurrent_fleet_pair_kim/`: confidence-guided policy/evaluation.
- `analysis/pair_kim_rl_example.py`: fixed-example prompt and cache adapter.
- `analysis/prepare_rc_mask_kim_20261111.py`: RC cache preparation wrapper.
- `generate_vrptw.py`: instance/event generation.
- `analysis/`: historical experiments, audits and evaluation entry points.
- `server/`: cluster execution helpers.

Read the relevant module's CLI and configuration before launching work.
Many dated research scripts intentionally refer to old `result/` or
`server_export/` artifacts. These are NOT plug-and-play launchers on a clean
clone: generate the required instances, baseline/reference runs and caches,
or parameterize those paths first. New Kim's RL reference requires a trained
Pure RL model; code alone cannot recreate that checkpoint without training.
Input datasets are now included in the transfer branch. External solvers must
still be installed separately. The portable workflow below does not require LKH.

## Reproducibility

Use a separate branch or worktree for each code change. Do not pull changes
into a directory while its experiment is running. Record the Git commit,
seed, configuration, input hashes, software versions and cache request hashes.
Independently regenerated LLM responses can differ, so they are a separate
experimental variable even when the source commit and RL seed match.

`source_manifest.json` records the exported source hashes. This branch is a
code transfer snapshot, not a claim that every historical experiment has
been rerun or that all historical path dependencies have been removed.

## NEW first-stop bias: portable C101 / R101 / RC101 workflow

Run from the checkout's `soft_pomo` directory with the destination Python
environment activated. Update from branch `code-sync-20261113` first.
Install a CUDA-compatible PyTorch separately and `pip install -r requirements-code.txt`.
No checkpoint, LLM confidence cache or API credential is included.
`experiment_inputs/new_head` contains hash-checked Train20/Test80 inputs, configs,
ordered groups and, for R/RC vehicle100, an independent fixed Pure RL example.
C needs a matching concurrent Pure RL checkpoint to generate its example.

The treatment adds +5 only to NEW(group[0]) while dispatch is open. It does not
boost other NEW customers or existing pairs during dispatch. After dispatch,
ordinary Kim anchor-group confidence (beta5) applies. It does not force one
vehicle per group. Default training is fresh 1000 epochs, vehicle cost100,
late lambda5, return mask ON, seed42, advantage standardization ON.

```powershell
python -m experiments.concurrent_fleet_pair_kim.portable validate --case r101
python -m experiments.concurrent_fleet_pair_kim.portable validate --case c101
```

R101: set `OPENROUTER_API_KEY` in this terminal, then explicitly allow cache API
calls. The usual cache budget is 21 logical requests (base1 + Train20; failed
requests may retry). No Test API calls are made by these commands.

```powershell
python -m experiments.concurrent_fleet_pair_kim.portable prepare --case r101 --allow-api
python -m experiments.concurrent_fleet_pair_kim.portable train --case r101 --method newhead --seed 42
```

C101: if you already have the matching Pure RL run directory containing
`protocol.json`, `split.json`, and `late5/checkpoint-1000.pt`, skip the first
command and pass that directory to `--pure-run`. Sequential-routing checkpoints
are not compatible substitutes. The reference command uses CPU, no API, and
creates a distinct event case checked against all Train/Test cases.

```powershell
python -m experiments.concurrent_fleet_pair_kim.portable train --case c101 --method pure --seed 42
python -m experiments.concurrent_fleet_pair_kim.portable reference --case c101 --pure-run result/new_head/c101_pure_v100_s42/run
python -m experiments.concurrent_fleet_pair_kim.portable prepare --case c101 --allow-api
python -m experiments.concurrent_fleet_pair_kim.portable train --case c101 --method newhead --seed 42
```

Use `--method kim` for the unchanged Kim baseline and share the SAME generated
cache with `--method newhead` to isolate the treatment. Seeds42-46 share caches;
each training output is separate. To run sequentially on one GPU:

```powershell
foreach ($experimentSeed in 42..46) {
    python -m experiments.concurrent_fleet_pair_kim.portable train --case r101 --method newhead --seed $experimentSeed
    if ($LASTEXITCODE -ne 0) { throw 'Training failed; inspect logs before continuing' }
}
```

Do not include a seed already completed in that loop. Use `--resume` only to
resume the same output/configuration. Additional options: `--artifact PATH`,
`--out PATH`, `--vehicle-cost 200`, `--head-beta 5`, `--smoke-updates 1` (use a
separate output directory for smoke tests). Vehicle200 requires its own matching
Pure RL reference/cache; a vehicle100 reference is rejected. Late lambda is
fixed to5 in this workflow. Independently regenerated caches are not byte-identical
to the old PC; compare baseline and treatment with the same new cache.

Progress: `result/new_head/CASE_METHOD_v100_s42/run/status.json` and
`run/late5/training.jsonl`. Training stops after saving
`run/late5/checkpoint-1000.pt` and `training_complete.json`; Test evaluation is
pending and must be run causally with the appropriate evaluator. No test results
are claimed by a training completion marker.
