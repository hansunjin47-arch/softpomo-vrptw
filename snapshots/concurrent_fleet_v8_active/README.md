# Active Local V8 Source Snapshot

This directory is the authoritative review target for the active local v8 run,
not the partial adapter upload at the repository root. Source files are copied
byte-for-byte, without portability edits to the running trainer or its imports.

Start with `experiments/concurrent_fleet_v8/train.py`. It includes full-graph
training, AMP/GradScaler, gradient clipping, checkpoint cadence, diagnostic
probes, per-update histories, plots, lambda sequencing and one-shot evaluation.
The v8 environment/policy, v7 SELECT head, v5 tensor environment, shared model,
event loader, transitive research modules and regression tests are included.
The guided package is also included but is NOT used by this pure-RL trainer.

## Identity And Inputs

- The original run protocol is retained under
  `result/concurrent_fleet_v8_fresh_train20_20261018/protocol.json`.
- All 11 protocol-recorded source files match their recorded SHA256 hashes.
- Configurations for lambda 5, 10, 1, 20 and the original train20/test80 split
  and 100 instance/observation pairs are retained at their original relative paths.
- No checkpoints, training results, private keys or environment files are included.
- Files in `result/` here are frozen inputs/provenance, not completed experiments.

## Review And Verification

Run from THIS snapshot directory using an explicit Python environment with
PyTorch, NumPy and Matplotlib. Do not run from the repository root: that resolves
the older release modules instead. `python verify_snapshot.py` checks the active
source hashes, all dataset hashes, input loading and the four configurations.

CPU checks (no additional GPU allocation):

```text
python verify_snapshot.py
python -m unittest experiments.concurrent_fleet_v8.test_equivalence.EnvironmentAudit experiments.concurrent_fleet_v8.test_equivalence.Equivalence.test_cpu_forward_state_gradient experiments.concurrent_fleet_guided.test_prompts experiments.concurrent_fleet_guided.test_guidance
```

Only when explicitly starting a NEW training run on an available GPU:

```text
python -m experiments.concurrent_fleet_v8.train --out runs/v8_new --weights 5 10 1 20
```

Do not use the provenance directory as a new output directory; it already exists.
The original GPU budget is batch128 x 100 rollouts, 2500 episodes per epoch,
1000 epochs per lambda. This upload does not restart or modify local training.
The copied historical experiment modules support imports and comparison; v8 is
the current entry point. Benchmark commands in historical READMEs can require
local checkpoints that are intentionally not uploaded.
