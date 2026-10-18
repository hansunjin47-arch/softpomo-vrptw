# Concurrent Fleet v5 (Experimental)

This is the train20 concurrent-fleet pure-RL experiment described in
`etc/concurrent_fleet_mdp_v5.md`, separate from the four released workflows.
Do not substitute it into `run.py` or reuse released Phase2 caches with it.

## Included

- `env.py`: GPU tensor fleet state, common clock, reservation and action masks.
- `policy.py`: existing encoder/attention decoder with fleet action mapping.
- `train.py`: full-trajectory single-backward training, AMP and diagnostics.
- `reference_env.py`: CPU oracle used only by parity tests.
- `test_tensor.py`, `test_fixture.py`: regression tests and a tiny fixture.
- `inputs/`: frozen train20/test80 split and lambda 5/10 configurations.

The environment and policy source are copied from the active local v5 version.
Portability changes are limited to import resolution, data/config paths, source
snapshot paths and the test fixture/oracle imports. The shared encoder/decoder
implementations were compared by Python AST, and all 100 scenario text hashes
matched the local split. No dataset, checkpoint, result or API secret is added.

## Validation

From the repository root, using the configured virtual environment:

```powershell
.venv\Scripts\python.exe -m unittest experiments.concurrent_fleet_v5.test_tensor -v
```

CUDA-specific tests are skipped when CUDA is unavailable. CPU tests alone are
not evidence of GPU training success. Do not run an additional GPU job alongside
an active full-memory training run just to validate this upload.

## Run

After hardware, occupancy and tests have been checked, a GPU validation run:

```powershell
.venv\Scripts\python.exe -m experiments.concurrent_fleet_v5.train --out runs/concurrent_v5_check --weights 5 --smoke-updates 8
```

This uses the real batch128 x rollout100 size, not a reduced batch. Accident
cases can approach/exceed physical 32 GB VRAM; do not silently lower the batch
or replace full-graph backward with replay. Inspect logs and memory first.

Full sequential lambda5 then lambda10 training and final test80 evaluation:

```powershell
.venv\Scripts\python.exe -m experiments.concurrent_fleet_v5.train --out runs/concurrent_v5_train20 --weights 5 10
```

Each lambda uses 1000 epochs, 2500 episodes per epoch, batch128 (last batch68),
100 rollouts, seed42 and its included reward config. CUDA is required; there is
no CPU fallback. This experimental runner does not inherit the released
runner's 11.5-hour wall-time budget.

To stop safely, create a file named `STOP` inside the run directory. The runner
saves at the next batch boundary. Wait for `stage: paused` and process exit.
Remove only that STOP file before resuming with the same arguments plus
`--resume`. Source/config hashes must still match. A smoke run has a distinct
protocol and is not silently converted into a full run.

Outputs include `status.json`, `protocol.json`, per-lambda `training.jsonl`,
`epochs.jsonl`, `early_history/`, fixed training `checks/`, checkpoints and final
`results.json`. `reward` is the sampled-rollout mean; `best_rollout_reward` is
the legacy per-episode best-rollout training metric. Test uses one greedy
trajectory per instance. Check `failure.json` and stderr if the process exits;
an older status file alone does not establish that a run is still alive.

The local long-running experiment remains untouched by this publication.
