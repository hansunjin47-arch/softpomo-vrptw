# Experiment Agent Instructions

## Scope

- Read README.md and work from this repository's root with an explicit virtual-environment Python.
- Support only Pure RL, Kim + Confidence, Pure500 + Phase2, and LKH-3 Full History.
- Use run.py, evaluate.py, and lkh-3/history.py. Do not execute legacy trainer main blocks.
- Do not start long training merely to check installation. Start it when requested by the user.
- Comments and repository documentation must be in English.

## Before Running

1. Inspect git status and preserve user changes. Do not pull over a dirty checkout or update code under an active run.
2. Identify the requested method, batch, seed, final epoch, and output directory. Follow explicit user settings.
   README examples use batch 128, seed 42, final epoch 1000; config defaults remain batch 64.
   If the method is ambiguous, ask. Report any adopted defaults before launching.
3. Verify Python and torch versions, CUDA availability, device name, and GPU occupancy.
   Install dependencies only in the selected environment; do not assume a CPU wheel supports CUDA.
4. Run `python -m unittest discover -s tools` using that interpreter.
5. Run `python run.py --method METHOD --smoke --cpu --out runs/UNIQUE_CHECK`.
   For a long GPU run, also check without --cpu. Use --smoke-batch-size to test the requested batch if needed.
6. Require successful checks. Missing data, cache, checkpoint, or integrity failures are blockers, not reasons to invent replacements.

## Execution

- Train: `python run.py --method METHOD --batch-size B --seed S --epochs E --out runs/UNIQUE_NAME`.
- METHOD is `pure`, `kim`, or `phase2`. Keep separate output directories and run sequentially on a shared GPU.
- Pure/Kim train from scratch. Phase2 loads proposed_model/phase2/checkpoints/pure500.pt and trains from epoch 501.
  `--epochs 1000` means final epoch 1000, not 1000 additional epochs. A Phase2 seed is a continuation seed.
- Phase2 caches correspond to the supplied Pure500 reference. Do not substitute a freshly trained checkpoint with these caches.
- The runner performs no paid API calls. Do not enable them, bypass network blocking, or request API keys for supported runs.
- Default stopping budget is 11.5 hours at epoch boundaries, not a hard scheduler wall-time limit.
  An exit code of zero does not prove final-epoch completion. Inspect status.json and checkpoint files.
- Resume with the same method/output/batch/seed plus --resume only when the prior process is no longer running.
  Do not launch two writers to one output directory. Do not delete outputs to make a command succeed.
- If asked to launch in the background, redirect stdout/stderr to durable files and record the process ID.
  On Windows, keep helper processes hidden. Do not claim a process started without checking its log/process.

## Evaluation and Reporting

- After completion: `python evaluate.py --run runs/NAME` (defaults to checkpoint-1000.pt).
- For another trained checkpoint use `--checkpoint FILENAME`; filenames are relative to the run directory.
- Evaluation output must be fresh; use --out for reruns. There is no evaluation --resume option.
- Only a full 80-row result set is test80 evaluation. --smoke uses two scenarios and is not a performance result.
- Report method, settings, checkpoint epoch, completion status, output paths, and whether evaluation finished.
- Report mean and sample standard deviation across scenarios, not training-seed uncertainty.

## LKH

- Reference replay: `python lkh-3/history.py --replay-reference --out runs/UNIQUE_NAME`.
- New solve: `python lkh-3/history.py --lkh /path/to/LKH --time-limit 600 --out runs/UNIQUE_NAME`.
- Do not describe reference replay as a new optimization. Do not silently relax constraints on solver failure.

## Changes and Integrity

- Do not change reward weights, batch size, data split, confidence caches, or bias semantics merely to make a run succeed.
- Do not regenerate tools/bundle_manifest.json to suppress a failed check.
- After an intentional, requested source/config change, run `python tools/refresh_manifest.py`, tests, and affected smoke paths.
  This changes the experiment contract. Prior runs require their original checkout for strict resume.
- Preserve frozen data and matching caches/checkpoints. Loading a checkpoint uses pickle; use trusted files only.
- Unused experimental branches may remain in shared source; the documented entry points define supported behavior.
