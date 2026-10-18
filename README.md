# SoftPOMO VRPTW

## Current Concurrent-Fleet Review

The complete active local v8 code, original configurations and event inputs are
in [snapshots/concurrent_fleet_v8_active](snapshots/concurrent_fleet_v8_active/README.md).
Start with its `experiments/concurrent_fleet_v8/train.py`. The root-level release
described below is separate; do not confuse it with the current training snapshot.

Four offline workflows on fixed RC101: LKH-3 Full History, Pure RL,
Kim + Confidence, and Pure RL500 + Phase2. Run commands from the repository root.
The release includes train80/test80, confidence caches, and the matched Pure500
checkpoint. It does not include the later ontology/prompt/scale ablations.

## For Coding Assistants

Read [AGENTS.md](AGENTS.md) before executing experiments. It specifies the
preflight checks, supported entry points, run naming, resume rules, and what
must not be changed automatically. [CLAUDE.md](CLAUDE.md) points to the same
instructions so the two guides do not drift.

This is a runnable source release, not a bundled Python/CUDA installation.
A fresh PC still needs Python, dependencies, and a compatible NVIDIA driver/GPU
for full RL runs. A new LKH solve additionally needs a separately installed LKH-3
executable; replaying the supplied plan does not.

## Layout

```text
data/                     # Scenarios, public observations, split, generator module
pure_rl/                  # Pure policy, shared environment, policy event inputs, config
proposed_model/
  kim/                    # Kim grouping algorithm, config.json, and cache/
    clustering.py         # Actual capacity/time-window clustering implementation
  phase2/                 # Phase2-specific config, caches, and initialization
    reference_cache.py    # Reference-route cache loading and disclosure gating
    checkpoints/pure500.pt
  *.py                    # Shared guided policy, trainer, and LLM/domain modules
lkh-3/                    # Historical matrix, solver interface, replay entry point
  reference/              # Original historical plan/matrix/summary
run.py                    # Shared training entry point
evaluate.py               # Shared learned-model evaluation and metric aggregation
bootstrap.py              # Resolve shared imports after moving files
tools/                    # Tests, manifest, and explicit manifest refresh script
requirements.txt
```

The proposed models reuse the environment and common policy utilities in
`pure_rl/`; those files are not duplicated. Some LLM/domain/generator modules
remain import dependencies even though supported commands do not call their API
or random-instance-generation functions. File organization is not a claim that
every legacy function is exercised by the four workflows.

The Pure RL folder now has four files:
- `config.json`: experiment settings.
- `train_vrptw.py`: network and training, including standard POMO advantage.
- `vrptw_env.py`: physical simulation, constraints, reward, and saved-route replay.
- `policy_event_inputs.py`: information available to the policy at each vehicle's
  time, shared with proposed models. It does not generate events or apply traffic
  effects; those belong to the data pipeline and simulator respectively.

Unused lexicographic advantage modes and their checkpoint state have been removed.
All supported methods keep the original reward-minus-mean, standard-deviation-scaled
POMO advantage. Non-reward modes are rejected by the supported runner.

## Fresh PC Setup

Python 3.12 is recommended. Windows PowerShell example:

```powershell
git clone https://github.com/hansunjin47-arch/softpomo-vrptw.git
cd softpomo-vrptw
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
```

Install a PyTorch build supporting your GPU using the
[official installation selector](https://pytorch.org/get-started/locally/).
Use `.venv\Scripts\python.exe -m pip` instead of `pip` in its command, then:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m unittest discover -s tools
.venv\Scripts\python.exe run.py --method phase2 --check-only --cpu --out runs/install_check
```

On Linux/WSL, use `.venv/bin/python`. The commands below use `python` to mean this
virtual environment's interpreter. Full RL training/evaluation requires CUDA;
`--cpu` supports small checks only. No API key, original PC paths, or external
parent repository is needed. API calls are blocked; missing caches are errors.

## Train and Evaluate

Choose **one** method below, not all methods unless requested. Training and
evaluation are separate steps. A training process can exit successfully after its
time budget with `status: needs_resume`; check `status.json` before evaluation.
Do not run evaluation automatically just because the training exit code is zero.

```sh
python run.py --method pure --batch-size 128 --epochs 1000 --out runs/pure128
python evaluate.py --run runs/pure128

python run.py --method kim --batch-size 128 --epochs 1000 --out runs/kim128
python evaluate.py --run runs/kim128

python run.py --method phase2 --batch-size 128 --epochs 1000 --out runs/phase2_128
python evaluate.py --run runs/phase2_128
```

Run sequentially on a shared GPU. Config defaults preserve historical batch 64
and seed 42; the explicit batch-128 commands above are new settings. `--seed 43`
changes the training seed. Pure/Kim start from random initialization. Phase2
loads the supplied Pure500 model and optimizer and trains epochs 501-1000;
changing its seed changes continuation randomness, not the Phase1 model.
Retraining Phase1 requires new matching reference routes and confidence caches.
Cache generation is outside this offline release.

Training stops gracefully at an epoch boundary after approximately 11.5 hours.
This is not a scheduler-enforced 12-hour limit and does not auto-submit a job.
Repeat the same command with `--resume` if `status.json` says `needs_resume`;
repeat batch/seed overrides. `--max-hours` changes this budget. Evaluation should
start after epoch 1000 completes. To evaluate another saved checkpoint, pass
`--checkpoint checkpoint-last.pt`. Use a new `--out` to preserve past evaluations.

For a small training check, use `--smoke --cpu` and a fresh `--out`; this tests
one rain and one accident backward pass plus checkpoint restoration, not an epoch.
Evaluation `--smoke --cpu` uses two test scenarios. Untrained smoke weights may
fail full-service evaluation; use trained weights for performance measurement.

Results include per-scenario routes/metrics in `results.json`, means and sample
standard deviations in `summary.json`, and provenance in `protocol.json`.
The sample unit is a test scenario, not a training seed or rollout candidate.

## LKH-3 Full History

Replay the included static plan without installing LKH:

```sh
python lkh-3/history.py --replay-reference --out runs/history_reference
```

To optimize a new plan, install LKH-3 separately and provide its executable:

```sh
python lkh-3/history.py --lkh /path/to/LKH --time-limit 600 --out runs/history_new
```

Use a native executable, or run the entire command in WSL with Linux Python and
LKH. No binary/source is redistributed. Settings: seed 42, one run, 10000 trials,
600 seconds. No silent constraint relaxation is applied if no feasible tour is
found. Time-limited solving need not reproduce the exact reference plan.

## Protocol

- RC101: 100 fixed customers, capacity 200, at most 25 vehicles, horizon 240.
- Train80/test80 each contain 40 rain and 40 accident scenarios, with disjoint
  membership. These are event variations, not independent customer networks.
- `data/generate_vrptw.py` is a shared customer-instance generator dependency,
  not a reproduction entry point for this frozen event pool. Use the supplied files.
- Full-service cost: `J = 0.04*K + T/5760 + 0.0075*L`; reward is `-J`.
  Lateness and late stops include depot returns; travel excludes waiting/service.
- Evaluation uses 100 starts and 8 geometric transforms, greedy decoding, and
  best candidate by realized simulated reward; no test gradient updates.
- Kim uses heuristic groups; Phase2 uses scenario-specific Pure500 route groups
  and offline simulated feedback. Existing caches are not refreshed during learning.
- The actual first customer's group anchors bias until depot return. Groups do
  not constrain feasible actions. Bias strength is 5; a customer below its group's
  depot score receives effective score -1, a soft penalty rather than a hard mask.
- Rain is known at dispatch. Accident observations/bias are gated at onset using
  each vehicle clock, reset to zero for new routes. Offline feedback and candidate
  selection still use the simulator; this is not an online-only information protocol.
- Full History averages training event-weighted distances without duration dilution,
  then rounds once with `numpy.rint`. One plan is replayed under test event timing
  with floating-point travel. LKH optimizes its static constrained formulation,
  not the RL weighted objective.

## Reproducibility and Changes

Source runs: `rc101_pure_train80_fresh1000_20260929`,
`rc101_kim_train80_fresh1000_20260929`, and
`rc101_pure500_train80_phase2_20260930`. Guided test caches came from each run's
`eval_1000_test80/cache`. Runtime logs, credentials, and later results are excluded.
The approximately 15 MB Pure500 checkpoint matches the Phase2 reference caches.
Load only trusted checkpoints: checkpoint loading uses pickle.
The scenario files retain Solomon data; cite Solomon (1987) and respect its terms.

All bundle inputs are checked, including files not needed by a particular method.
For intentional source changes run `python tools/refresh_manifest.py`; this creates a
new experiment version. Do not regenerate hashes to bypass failed integrity checks.
Moving files also changes the manifest: pre-reorganization runs cannot resume
under this version's strict contract. Keep an old checkout to resume those runs.
The model parameters, supplied caches, and initial checkpoint bytes are unchanged.

Previous validation covered contract tests, all three training smoke tests,
two-scenario evaluation with historical weights, and full test80 reference replay.
The folder migration is checked with the same input tests and training smoke paths.
Reference replay has mean K=19, T=1785.70642065, L=11.15164798. Full retraining and
a fresh LKH solve are not packaging checks. Legacy versions remain in Git history.

## Root Files

`run.py` and `evaluate.py` are the shared execution commands. `bootstrap.py`
resolves imports across method folders and is currently required by those commands.
`requirements.txt` and `README.md` support installation and usage, while
`.gitignore` and `.gitattributes` control versioning and byte-preserving checkout.
They are not all numerical model inputs.

Tests and maintenance utilities live in `tools/`, not at the root. The manifest
there is still checked at runtime. Tests and manifest regeneration are development
tools; regenerating the manifest is not a normal training step.

Kim/Phase2 training still consumes existing caches. Moving the actual Kim grouping
implementation into `proposed_model/kim/clustering.py` does not enable paid cache
generation. Shared attention, decoding, and training code lives directly in `proposed_model/`
to prevent the two methods from drifting through duplicated implementations.

## Preflight, Resume, and Troubleshooting

After installation, use an explicit interpreter instead of relying on shell activation.
Windows PowerShell, from the repository root:

```powershell
$py = (Resolve-Path .venv\Scripts\python.exe).Path
& $py -c "import sys,torch; print(sys.executable); print(torch.__version__); print('CUDA:',torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU')"
& $py -m unittest discover -s tools
& $py run.py --method kim --smoke --cpu --out runs/kim_preflight_001
```

Check each command's exit code (`$LASTEXITCODE`) before proceeding. Use a new
preflight directory on each attempt. CPU smoke tests check code/data/cache and
checkpoint handling, not GPU driver compatibility. Before a long GPU run, perform
the selected method's smoke test **without** `--cpu`, using another fresh output
directory. A batch-1 smoke pass does not establish that batch 128 fits GPU memory;
use `--smoke-batch-size 128` to check that requested batch when GPU resources permit.

Example training, status inspection, and resume commands (execute as needed):

```powershell
& $py run.py --method kim --batch-size 128 --seed 42 --epochs 1000 --out runs/kim_b128_s42_001
Get-Content runs/kim_b128_s42_001/status.json
# Run this only when status is needs_resume and no process still owns the run:
& $py run.py --method kim --batch-size 128 --seed 42 --epochs 1000 --out runs/kim_b128_s42_001 --resume
# Run evaluation only after completion and checkpoint-1000.pt exists:
& $py evaluate.py --run runs/kim_b128_s42_001
```

Linux/WSL equivalents use `.venv/bin/python` and `cat` for status inspection.
Status is the last recorded state, so also inspect the process/log before deciding
that a run is active, interrupted, or complete.

| Symptom | Action |
| --- | --- |
| CUDA unavailable or unsupported GPU | Check the selected interpreter, driver, and PyTorch build; CPU checks are not full training |
| CUDA out of memory | Check competing processes; do not silently lower the requested batch size |
| Nonempty output directory | Use a new directory, or `--resume` for the same interrupted training run |
| Missing cache or manifest mismatch | Verify the checkout/files; do not invoke an API or regenerate hashes to suppress the error |
| Resume contract mismatch | Use the matching code/settings version; do not patch the checkpoint contract |
| Missing epoch-1000 checkpoint | Inspect training status; finish/resume training or explicitly evaluate an available checkpoint |
| Incomplete service from smoke weights | Untrained smoke weights are not a trained model; do not report them as results |
| LKH not found | Provide the installed executable path, or explicitly choose reference replay |

Evaluation has no `--resume` option. If interrupted, retain partial outputs and
rerun with a new `--out` directory. Never label partial results as test80 evaluation.
