"""Validate and run fresh train20 A/C travel-time experiments without LLM calls."""
from pathlib import Path
import subprocess
import sys

from run_pure80_vehicle_distance import prepare, write, read

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / 'result/corrected_tt_ac_train20_20261010'


def main():
    runs = []
    for variant in ('A', 'C'):
        run = ROOT / f'result/rc101_pure_train20_b128_s42_config{variant}_travel_time_1000_20261010_v2'
        prepare(run, variant, 20, 'travel_time')
        runs.append(run)
    for run in runs:
        write(QUEUE/'status.json', dict(stage='validating', run=run.name))
        subprocess.run([sys.executable, '-u', str(run/'runner.py'), 'smoke', '--run', str(run)], check=True, cwd=ROOT)
        assert (run/'smoke/travel_time_reward_validation.json').is_file()
    for run in runs:
        write(QUEUE/'status.json', dict(stage='training_and_evaluation', run=run.name))
        subprocess.run([sys.executable, '-u', str(run/'runner.py'), 'train', '--run', str(run)], check=True, cwd=ROOT)
        assert read(run/'status.json')['status'] == 'complete'
    write(QUEUE/'status.json', dict(stage='complete', runs=[r.name for r in runs]))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        write(QUEUE/'status.json', dict(stage='failed', error=repr(exc)))
        raise
