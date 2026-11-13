"""GPT Phase2 from the matched lambda5 Pure500 checkpoint, with test80 evaluation."""
import argparse
from pathlib import Path
import subprocess
import sys

import run_gpt_bks_late10 as shared

shared.BASE = shared.ROOT / 'server_export/rc101_bks_late5_received_20261014_163304/late5'
shared.OUT = shared.ROOT / 'result/rc101_phase2_train20_bks_late5_gpt54_v1'
shared.LATE_WEIGHT = 5
shared.OBJECTIVE = '''Minimize J=K/14+(T+5*Lt)/1696.95 for fully served solutions.
K is the number of nonempty vehicle routes. T is actual event-dependent driving time in minutes,
including depot departure and return legs; it excludes waiting and service durations.
Lt is total customer AND depot lateness in minutes. Late-stop count is not separately penalized.
The references are RC101 best-known 14 vehicles and 1696.95 distance units; at base speed 1
the latter is the reference no-event driving time in minutes.
One lateness minute costs five driving minutes. One vehicle costs 121.210714 driving minutes,
or 24.242143 lateness minutes with other terms fixed. These are weighted trade-offs,
NOT strict lexicographic priorities. Geometric distance is NOT a separate objective term.
The complete cost also includes 500*(unserved customers/100); full service is required.
Do not claim a measured improvement without an evaluated alternative; use only supplied evidence.
'''


def pipeline():
    for stage in ('prepare', 'call', 'train', 'prepare-test', 'call-test', 'evaluate'):
        shared.io.write(shared.OUT / 'pipeline.json', dict(stage=stage, lateness_weight=5))
        command = [sys.executable, '-u', str(Path(__file__).resolve()), stage]
        if stage in ('call', 'call-test'):
            command.append('--allow-api')
        subprocess.run(command, cwd=shared.ROOT, check=True)
    shared.io.write(shared.OUT / 'pipeline.json', dict(stage='complete', epoch=1000, evaluated=80))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'call', 'train', 'prepare-test', 'call-test', 'evaluate', 'pipeline'])
    parser.add_argument('--allow-api', action='store_true')
    args = parser.parse_args()
    if args.stage in ('pipeline', 'call', 'call-test') and not args.allow_api:
        parser.error('Paid calls require --allow-api')
    try:
        if args.stage == 'pipeline':
            pipeline()
        elif args.stage.startswith('prepare'):
            shared.prepare(test=args.stage == 'prepare-test')
        elif args.stage.startswith('call'):
            shared.call(shared.section(args.stage == 'call-test'))
        elif args.stage == 'train':
            shared.train()
        else:
            shared.evaluate()
    except Exception as error:
        shared.io.write(shared.OUT / 'pipeline.json', dict(stage='failed', failed_stage=args.stage, error=repr(error)))
        raise
