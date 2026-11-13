"""Preserve frozen training contracts; replace only evaluation serialization."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import traceback

import run_pure80_vehicle_distance as corrected


def worker(run, mode):
    spec = importlib.util.spec_from_file_location('frozen_runner', run / 'runner.py')
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    frozen.evaluate = corrected.evaluate
    if mode == 'eval':
        state = corrected.read(run / 'status.json')
        if state['epoch'] != 1000 or not (run / 'training/checkpoint-1000.pt').exists():
            raise ValueError('Evaluation recovery requires completed epoch 1000')
        # Resume restores epoch 1000, so the training loop performs zero updates.
        frozen.execute(run, resume=True)
    else:
        frozen.execute(run, smoke=mode == 'smoke')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--a', type=Path)
    p.add_argument('--c', type=Path)
    p.add_argument('--run', type=Path)
    p.add_argument('--mode', choices=('eval', 'smoke', 'train'))
    args = p.parse_args()
    if args.run:
        worker(args.run.resolve(), args.mode)
    else:
        state = Path(__file__).resolve().parent / 'recovery_status.json'
        try:
            for mode, run in [('eval', args.a), ('smoke', args.c), ('train', args.c)]:
                corrected.write(state, dict(status='running', mode=mode, run=str(run)))
                subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()),
                    '--run', str(run.resolve()), '--mode', mode], check=True)
            corrected.write(state, dict(status='complete'))
        except Exception:
            corrected.write(state, dict(status='failed', traceback=traceback.format_exc()))
            raise
