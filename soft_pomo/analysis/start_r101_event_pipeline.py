"""Generate R101 pilot, evaluate history, enrich inputs, then train Pure RL."""
import datetime
from pathlib import Path
import subprocess
import sys
from eval_fixed_v2_lkh import ROOT, write
from generate_rc101_event_pilot import main as generate
from eval_rc101_pilot_history_split import main as history
from enrich_rc101_observations import enrich


def main():
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    raw = ROOT / 'data' / ('r101_event_pilot_' + stamp)
    hist = ROOT / 'analysis' / ('r101_pilot_history_20_80_' + stamp)
    data = ROOT / 'data' / ('r101_event_pilot_discrete_' + stamp)
    run = ROOT / 'result' / ('r101_pilot_discrete_pure1000_' + stamp)
    run.mkdir(parents=True, exist_ok=False)
    status = dict(raw=str(raw), history=str(hist), data=str(data), run=str(run))
    def update(stage):
        status['stage'] = stage
        write(run / 'pipeline.json', status)
        print(stage, flush=True)
    update('generating')
    generate('r101', raw)
    update('history')
    history(raw, hist, 'r101')
    update('observations')
    enrich(raw, hist / 'split.json', data)
    update('training')
    with (run / 'stdout.log').open('w', encoding='utf-8') as stdout, (run / 'stderr.log').open('w', encoding='utf-8') as stderr:
        result = subprocess.run([sys.executable, '-u', str(ROOT / 'analysis/train_rc101_event_pilot.py'),
            '--run-dir', str(run), '--data-dir', str(data), '--benchmark', 'r1'],
            cwd=ROOT, stdout=stdout, stderr=stderr)
    update('complete' if result.returncode == 0 else 'training_failed')
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
