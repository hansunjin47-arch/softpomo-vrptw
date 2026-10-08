# SoftPOMO VRPTW

Four offline workflows on fixed RC101: LKH-3 Full History, Pure RL,
Kim + Confidence, and Pure RL500 + Phase2. Run commands from the repository root.
The release includes train80/test80, confidence caches, and the matched Pure500
checkpoint. It does not include the later ontology/prompt/scale ablations.

## Layout

```text
data/                     # Scenarios, public observations, split, generator module
pure_rl/                  # Pure policy, shared environment/advantage/observations, config
proposed_model/           # Kim/Phase2 trainers and shared LLM/domain modules
  configs/                # kim.json and phase2.json
  cache/kim/              # Training and test confidence caches
  cache/phase2/
  checkpoints/pure500.pt   # Matched Phase2 initialization
lkh-3/                    # Historical matrix, solver interface, replay entry point
  reference/              # Original historical plan/matrix/summary
run.py                    # Shared training entry point
evaluate.py               # Shared learned-model evaluation and metric aggregation
bootstrap.py              # Resolve shared imports after moving files
test_release.py           # Input/integrity/resume checks
refresh_manifest.py       # Explicitly version intentional source changes
bundle_manifest.json      # SHA-256 of source and experiment inputs
requirements.txt
```

The proposed models reuse the environment and common policy utilities in
`pure_rl/`; those files are not duplicated. Some LLM/domain/generator modules
remain import dependencies even though supported commands do not call their API
or random-instance-generation functions. File organization is not a claim that
every legacy function is exercised by the four workflows.

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
.venv\Scripts\python.exe -m unittest test_release
.venv\Scripts\python.exe run.py --method phase2 --check-only --cpu --out runs/install_check
```

On Linux/WSL, use `.venv/bin/python`. The commands below use `python` to mean this
virtual environment's interpreter. Full RL training/evaluation requires CUDA;
`--cpu` supports small checks only. No API key, original PC paths, or external
parent repository is needed. API calls are blocked; missing caches are errors.

## Train and Evaluate

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
For intentional source changes run `python refresh_manifest.py`; this creates a
new experiment version. Do not regenerate hashes to bypass failed integrity checks.
Moving files also changes the manifest: pre-reorganization runs cannot resume
under this version's strict contract. Keep an old checkout to resume those runs.
The model parameters, supplied caches, and initial checkpoint bytes are unchanged.

Previous validation covered contract tests, all three training smoke tests,
two-scenario evaluation with historical weights, and full test80 reference replay.
The folder migration is checked with the same input tests and training smoke paths.
Reference replay has mean K=19, T=1785.70642065, L=11.15164798. Full retraining and
a fresh LKH solve are not packaging checks. Legacy versions remain in Git history.
