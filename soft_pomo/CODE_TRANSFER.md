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
Solomon input instances and external solvers must also be supplied separately.

## Reproducibility

Use a separate branch or worktree for each code change. Do not pull changes
into a directory while its experiment is running. Record the Git commit,
seed, configuration, input hashes, software versions and cache request hashes.
Independently regenerated LLM responses can differ, so they are a separate
experimental variable even when the source commit and RL seed match.

`source_manifest.json` records the exported source hashes. This branch is a
code transfer snapshot, not a claim that every historical experiment has
been rerun or that all historical path dependencies have been removed.
