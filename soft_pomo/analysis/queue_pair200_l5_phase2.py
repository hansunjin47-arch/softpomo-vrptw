"""Run matched phase two only after new Kim completes training and test80."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.concurrent_fleet_pair.io import write
from experiments.concurrent_fleet_bias_guided.support import ROOT, read

PURE = ROOT/'result/pair200_late5_epsilon005_floor001_std_20261029'
PREVIOUS = ROOT/'result/pair200_late5_std_kim_example_20261030/training'
ART = ROOT/'result/pair200_l5_phase2_cache_20261030'
OUT = ROOT/'result/pair200_l5_phase2_20261030'
QUEUE = ROOT/'result/queue_pair200_l5_phase2_20261030'


def stage(name):
    from analysis import prepare_pair_phase2_feedback as prep
    from analysis import pair_phase2_feedback_run as runner
    prep.PURE, prep.ART, prep.LATE = PURE, ART, 5
    runner.PURE, runner.ART, runner.OUT = PURE, ART, OUT
    runner.LATE, runner.VEHICLE_COST = 5, 200
    if name == 'prepare':
        prep.main(allow_api=True)
    elif name == 'prepare-test':
        runner.prepare_test()
    else:
        runner.run()


def read_status(path):
    for attempt in range(40):
        try:
            return read(path)
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(.1)


def main(retry=False):
    if retry:
        previous = read_status(QUEUE/'status.json')
        assert previous['stage'] == 'failed' and not OUT.exists(), 'Unsafe retry'
        write(QUEUE/'previous_failure.json', previous)
    QUEUE.mkdir(exist_ok=retry)
    try:
        assert not OUT.exists(), 'Duplicate output'
        assert os.environ.get('OPENROUTER_API_KEY'), 'Missing API key'
        checkpoint = PURE/'late5/checkpoint-500.pt'
        files = list((ROOT/'experiments').rglob('*.py')) + [checkpoint, Path(__file__)]
        files += [ROOT/'analysis'/name for name in ('prepare_pair_phase2_feedback.py', 'pair_phase2_feedback_run.py')]
        hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        write(QUEUE/'protocol.json', dict(previous=str(PREVIOUS), pure=str(PURE), out=str(OUT),
            artifacts=str(ART), vehicle_cost=200, late=5, seed=42, hashes=hashes,
            schedule='wait for Kim train1000 + test80; generate train/test caches; restore Pure500; train501-1000; test80',
            test='single greedy, no RL reference, live accident refresh at disclosure'))
        while True:
            if (QUEUE/'STOP').exists():
                write(QUEUE/'status.json', dict(stage='cancelled_before_launch'))
                return
            status = read_status(PREVIOUS/'status.json')
            if (PREVIOUS/'failure.json').exists() or status['stage'] in ('failed', 'paused'):
                raise RuntimeError('Predecessor failed; refusing automatic launch')
            predecessor_queue = read_status(ROOT/'result/queue_pair200_late5_kim_20261030/status.json')
            if predecessor_queue['stage'] == 'failed':
                raise RuntimeError('Kim queue failed')
            if status['stage'] == 'complete':
                done = read(PREVIOUS/'late5/complete.json')
                assert done['epoch'] == 1000 and done['test_count'] == 80
                assert len(read(PREVIOUS/'late5/results.json')) == 80
                break
            write(QUEUE/'status.json', dict(stage='waiting', pid=os.getpid(), previous_epoch=status.get('epoch')))
            time.sleep(30)
        for name in ('prepare', 'prepare-test', 'run'):
            assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p,h in hashes.items()), 'Source/checkpoint changed'
            if (QUEUE/'STOP').exists():
                write(QUEUE/'status.json', dict(stage='cancelled_before_launch'))
                return
            with (QUEUE/f'{name}.stdout.log').open('w') as stdout, (QUEUE/f'{name}.stderr.log').open('w') as stderr:
                child = subprocess.Popen([sys.executable, '-u', '-m', 'analysis.queue_pair200_l5_phase2', '--stage', name],
                                         cwd=ROOT, stdout=stdout, stderr=stderr)
                write(QUEUE/'status.json', dict(stage=name, pid=child.pid))
                if child.wait() != 0:
                    raise RuntimeError(f'{name} failed; inspect logs')
        done = read(OUT/'training/late5/complete.json')
        assert done['epoch'] == 1000 and done['test_count'] == 80
        write(QUEUE/'status.json', dict(stage='complete'))
    except Exception as error:
        write(QUEUE/'status.json', dict(stage='failed', error=repr(error)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['prepare', 'prepare-test', 'run'])
    parser.add_argument('--retry', action='store_true')
    args = parser.parse_args()
    stage(args.stage) if args.stage else main(args.retry)
