# RC101: four offline experiment workflows

## Fresh PC Setup

This release does not require the original PC's absolute paths, API credentials,
or experiment output directories. The commands below use Windows PowerShell.
On Linux/WSL, use `.venv/bin/python` as the virtual environment's Python path.

```powershell
git clone https://github.com/hansunjin47-arch/softpomo-vrptw.git
cd softpomo-vrptw/experiments/rc101
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
```

Choose the installation command for **your GPU** from the
[PyTorch installation guide](https://pytorch.org/get-started/locally/), replacing
`pip` with `.venv\Scripts\python.exe -m pip`. Then run the commands below.
Full training requires a CUDA GPU.

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m unittest test_release
.venv\Scripts\python.exe server/train_cached.py --method phase2 --check-only --cpu --out runs/install_check
.venv\Scripts\python.exe server/train_cached.py --method pure --batch-size 128 --epochs 1000 --out runs/pure128
.venv\Scripts\python.exe evaluate.py --run runs/pure128
```

Run evaluation after `status.json` reports `complete` and the epoch-1000
checkpoint exists. If the time budget interrupts training, repeat the training
command with `--resume`. For Kim/Phase2, use `--method kim` or `--method phase2`
and a separate output directory. Pure/Kim start from random initialization;
Phase2 trains epochs 501-1000 from the supplied Pure500 checkpoint.
**Running on a fresh PC does not mean retraining Phase1.** A newly trained
Phase1 model requires matching reference routes and confidence caches.

### File Roles and Dependencies

Not every file is used by every run. The release verifies all four workflows
and their reference inputs as a single bundle; deleting files arbitrarily will
fail the manifest check.

| File/directory | Role and scope |
| --- | --- |
| `server/train_cached.py`, `configs/` | Shared entry point and settings for the three RL methods |
| `train_vrptw.py`, `vrptw_env.py`, `event_observation.py`, `advantage.py` | Policy, environment, observations, and training calculations |
| `train_vrptw_llm.py`, `train_soft_cluster.py` | Kim/Phase2 models and trainers, including parent classes |
| `*LLMModule.py`, `*Ontology.py`, `generate_vrptw.py` | Import dependencies; supported commands do not generate API caches or random customer instances |
| `data/` | Train80/test80 scenarios and public observations; membership is specified in `split.json` |
| `cache/kim`, `cache/phase2` | Existing LLM outputs for each guided method; not used by the Pure RL algorithm |
| `checkpoints/pure500.pt` | Initial Phase2 model/optimizer; not used to initialize Pure/Kim |
| `evaluate.py` | Test evaluation of trained models and shared metric aggregation |
| `history.py`, `lkh_io.py` | LKH planning matrix, external solver invocation, and saved-route replay |
| `reference/history/` | Reference LKH plan and matrix checks; not neural training inputs |
| `bundle_manifest.json`, `test_release.py`, `refresh_manifest.py` | Integrity checks, tests, and versioning of intentional changes |
| `README.md`, `requirements.txt`, Git configuration | Installation and usage support |

Suggested reading order: `server/train_cached.py`, `configs/`, the selected trainer,
`vrptw_env.py`, then `evaluate.py`. English comments explain key control paths.
LLM/ontology filenames do not imply API calls or enable the latest ontology experiments.
Shared modules still contain legacy branches. **File-level cleanup is complete,
but the code has not been rewritten into a minimal set of functions for these workflows.**

Run all commands **from this directory**. Python 3.12 is recommended. These are
the original train80/test80 four-method settings, not the later ontology ablations.

## Installation

Create and activate a virtual environment. Install a CUDA-enabled PyTorch build
appropriate for your GPU using https://pytorch.org/get-started/locally/, then:

```sh
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Full training/evaluation requires a CUDA GPU. `--cpu` supports small checks only.
For an RTX 5090, use a PyTorch/CUDA build supporting that GPU; do not reuse an old
CUDA wheel. No API key, OpenRouter, Ollama, MLflow server, or external parent
repository imports are required by the supported commands. Network/API calls are
blocked in the RL runner, and missing confidence caches are errors.

## Quick verification (no full training)

```sh
python server/train_cached.py --method pure --check-only --cpu --out runs/check_pure
python server/train_cached.py --method kim --check-only --cpu --out runs/check_kim
python server/train_cached.py --method phase2 --check-only --cpu --out runs/check_phase2
python history.py --replay-reference --out runs/history_reference
python -m unittest test_release
```

For a forward/backward/checkpoint smoke test use `--smoke` instead of
`--check-only`. It runs one rain and one accident batch, not an epoch.
Always use a fresh output directory for checks. The included SHA-256 manifest
checks source, configuration, input data, caches, and the supplied checkpoint.

## Train and evaluate

```sh
python server/train_cached.py --method pure --out runs/pure --epochs 1000
python evaluate.py --run runs/pure

python server/train_cached.py --method kim --out runs/kim --epochs 1000
python evaluate.py --run runs/kim

python server/train_cached.py --method phase2 --out runs/phase2 --epochs 1000
python evaluate.py --run runs/phase2
```

Run these sequentially on a shared GPU. Defaults are seed 42 and batch 64,
preserving the historical experiment configuration. To test batch 128, add
`--batch-size 128` to the training command. This is a new setting, not an exact
reproduction of the historical batch-64 result. `--seed 43` changes training seed.
For Phase2 this changes continuation randomness, **not** the supplied Phase1 seed.

Pure and Kim train epochs 1-1000. Phase2 loads `checkpoints/pure500.pt` and its
optimizer, then trains epochs 501-1000. Its supplied confidence caches were
generated from that exact reference model. Do not substitute another Pure500
checkpoint without regenerating reference routes and matching caches. Cache/API
generation is deliberately outside this offline release.

The default runner stops gracefully after approximately 11.5 hours at an epoch
boundary; this is **not a scheduler-enforced 12-hour limit**. Resume with the same
command and `--resume` (repeat batch/seed overrides). Check `status.json` for
`complete` versus `needs_resume`. It does not auto-submit another job.
`--max-hours` controls this budget. `checkpoint-last.pt` is saved periodically;
epoch 500 and multiples of 1000 also receive numbered checkpoints.

Evaluation writes `results.json` (per-scenario routes/metrics), `summary.json`
(mean and sample standard deviation), and `protocol.json`. It uses the checkpoint
and settings from `--run`; it never trains on test data. Defaults: 100 starts,
8 augmentations, greedy decoding, best candidate by simulated reward. For an
interrupted or non-1000 run use `--checkpoint checkpoint-last.pt`. A fresh
`--out` avoids overwriting prior evaluations. `--smoke` evaluates two scenarios.

## LKH-3 Full History

The included reference plan can be replayed without installing LKH-3. To solve
again, install LKH-3 separately and pass its executable:

```sh
python history.py --lkh /path/to/LKH --out runs/history_new --time-limit 600
```

On Windows, use a native executable or run the entire command in WSL with its
own Python environment and Linux LKH executable. No fixed WSL distribution or
root account is assumed. LKH binaries/source are not redistributed here.

The matrix averages full event-weighted distances over train80 only. Event onset
and duration do not dilute planning weights. Rounding (`numpy.rint`) occurs once
after averaging. One static route plan is replayed across all test80 cases using
floating-point, departure-time-dependent travel. Solver settings: seed 42,
one run, 10000 maximum trials, 600-second limit. No infeasibility relaxation is
silently applied. Time-limited solving may not reproduce the exact reference plan.

## Protocol and interpretation

- Fixed RC101: 100 customers, capacity 200, maximum 25 vehicles, horizon 240.
- Train80 and disjoint test80 each contain 40 rain and 40 accident scenarios.
  These are synthetic event variations, not 80 independent customer networks.
- Full-service cost is `J = 0.04*K + T/5760 + 0.0075*L`; reward is `-J`.
  `L` and late stops include depot returns. Travel excludes waiting/service.
- Kim uses fixed heuristic groups; Phase2 uses scenario-specific Pure500 route
  groups and offline simulated feedback. Confidence is cached, not refreshed
  during training. This is not a public-information-only online evaluation.
- Guidance uses the actual first customer's reference group until depot return;
  all 100 starts remain available. Groups do not constrain feasible actions.
- Bias strength is 5. In this implementation, a customer score below the group's
  depot score becomes an effective score of -1. This is a soft negative bias,
  not a feasibility mask. Otherwise the customer confidence supplies the bias;
  out-of-group customers receive zero. Depot scores are retained.
- Accident observations/bias are gated at onset on each vehicle clock, including
  reset to zero for a new route. Rain is known at dispatch. Offline reference
  feedback and candidate selection still use the simulator.
- LKH solves its static constrained formulation, not the RL weighted objective.
  Replay compares outcomes; it does not make the optimization problems identical.

## Files, provenance, and modifications

`configs/` defines the three RL methods. `data/split.json` records membership and
scenario hashes. `cache/{kim,phase2}` includes both train and test caches.
`checkpoints/pure500.pt` is the approximately 15 MB seed-42 train80 reference
checkpoint, included to keep Phase2 runnable with the matching caches.
Only load trusted checkpoints: PyTorch checkpoint loading uses pickle.
`reference/history` contains the original static plan, matrix, and summary.

Source workflows: `rc101_pure_train80_fresh1000_20260929`,
`rc101_kim_train80_fresh1000_20260929`, and
`rc101_pure500_train80_phase2_20260930`. Test caches come from each guided run's
`eval_1000_test80/cache`. Runtime logs, API requests/credentials, and later
experimental outputs are not included. Solomon instance data is retained in
the scenario files; cite Solomon (1987) and respect upstream dataset terms.

Use the documented entry points, not the legacy modules' standalone main blocks.
The legacy core is preserved to avoid changing the numerical algorithm while
packaging. CLI batch/seed/epoch overrides are recorded in run settings. For
intentional source/config changes, regenerate the manifest with
`python refresh_manifest.py`; this creates a **different experiment version** and
will intentionally prevent resuming an old run under the new code. The manifest
checks consistency, not authenticity; use Git history for provenance.

## Release validation

Packaging was checked with Python 3.12.6, PyTorch 2.9.1+cu128, and CPU execution:
four contract tests; rain/accident backward steps and checkpoint roundtrips for
Pure, Kim, and Phase2; two-scenario evaluation for all three using historical
trained weights; and all 80 scenarios replayed for the LKH reference plan.
Reference replay reproduces mean K=19, T=1785.70642065, L=11.15164798.
An untrained smoke checkpoint can fail full-service evaluation; use trained
weights for performance evaluation. A fresh 1000-epoch run and a fresh native
LKH solve were not part of this packaging check.
