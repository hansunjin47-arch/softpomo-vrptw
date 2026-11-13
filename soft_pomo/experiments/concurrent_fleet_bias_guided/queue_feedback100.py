"""Vehicle100 follow-up; wait for all vehicle300 arms and the matching Pure500."""
import argparse
import os
import subprocess
import sys
import time
import traceback

from experiments.concurrent_fleet_bias_guided.support import ROOT, read
from experiments.concurrent_fleet_pair.io import write

OUT = ROOT/'result/v9_gate_feedback100_queue_20261024'
PREVIOUS = ROOT/'result/v9_gate_feedback_queue_20261024'
PURE = ROOT/'server_export/be100_received_20261022_100545/runs/vehicle100'
CHECKPOINT = PURE/'late10/checkpoint-500.pt'


def run_arm(mode):
    from experiments.concurrent_fleet_bias_guided import queue_feedback as queue
    from experiments.concurrent_fleet_v9_phase2 import prepare as reference
    from experiments.concurrent_fleet_stop_init.policy import FleetPolicy
    source = OUT/'references/phase2/artifacts'
    if not (source/'status.json').exists() or read(source/'status.json')['stage'] != 'requests_ready':
        reference.OUT = source
        reference.PURE = PURE
        reference.FleetPolicy = FleetPolicy
        reference.main(paid=False)
    queue.OUT = OUT
    queue.PURE = PURE
    queue.OLD = OUT/'references'
    queue.SHARED = OUT/'shared_feedback'
    queue.arm(mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['gate', 'warm', 'fresh'])
    args = parser.parse_args()
    if args.arm:
        run_arm(args.arm)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT/'queue.lock'
    with lock.open('x') as handle:
        handle.write(str(os.getpid()))
    try:
        while True:
            previous = read(PREVIOUS/'status.json')['stage']
            if previous in ('failed', 'paused'):
                raise RuntimeError('Vehicle300 queue requires attention')
            if previous == 'complete' and CHECKPOINT.exists():
                break
            write(OUT/'status.json', dict(stage='waiting', predecessor=previous,
                checkpoint_available=CHECKPOINT.exists(), required_checkpoint=str(CHECKPOINT),
                order=['gate', 'warm', 'fresh']))
            time.sleep(60)
        for mode in ('gate', 'warm', 'fresh'):
            completed = read(PREVIOUS/mode/'phase2/training/late10/complete.json')
            if completed['epoch'] != 500 or completed['test_count'] != 80:
                raise RuntimeError('Vehicle300 completion is incomplete')
        for mode in ('gate', 'warm', 'fresh'):
            write(OUT/'status.json', dict(stage='running', arm=mode))
            with (OUT/(mode+'.stdout.log')).open('a') as stdout, (OUT/(mode+'.stderr.log')).open('a') as stderr:
                subprocess.run([sys.executable, '-u', '-m',
                    'experiments.concurrent_fleet_bias_guided.queue_feedback100', '--arm', mode],
                    cwd=ROOT, stdout=stdout, stderr=stderr, check=True)
        write(OUT/'status.json', dict(stage='complete'))
    except Exception:
        write(OUT/'status.json', dict(stage='failed', error=traceback.format_exc()))
        raise
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
