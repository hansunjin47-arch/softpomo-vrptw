"""Run the matched vehicle200 experiment after vehicle100 finishes evaluation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.concurrent_fleet_pair.io import write

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / 'result/pair100_epsilon005_floor001_std_20261028'
OUT = ROOT / 'result/pair200_epsilon005_floor001_std_20261028'
QUEUE = ROOT / 'result/queue_pair200_after100_20261028'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    QUEUE.mkdir(exist_ok=False)
    try:
        assert not OUT.exists(), 'Output exists; refusing duplicate run'
        hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (ROOT / 'experiments').rglob('*.py')}
        command = [sys.executable, '-u', '-m', 'experiments.concurrent_fleet_pair.train',
                   '--out', str(OUT), '--weights', '10', '--vehicle-cost', '200',
                   '--new-customer-epsilon', '0.05', '--epsilon-final', '0.01',
                   '--epsilon-end-epoch', '1000']
        write(QUEUE / 'protocol.json', dict(previous=str(PREVIOUS), out=str(OUT),
              command=command, source_hashes=hashes, initialization='fresh; no resume'))
        while True:
            if (QUEUE / 'STOP').exists():
                write(QUEUE / 'status.json', dict(stage='cancelled_before_launch'))
                return
            status = read(PREVIOUS / 'status.json')
            if (PREVIOUS / 'failure.json').exists() or status['stage'] in ('failed', 'paused'):
                raise RuntimeError('Predecessor failed or paused; refusing launch')
            if status['stage'] == 'complete':
                done = read(PREVIOUS / 'late10/complete.json')
                assert done['epoch'] == 1000 and done['test_count'] == 80
                assert len(read(PREVIOUS / 'late10/results.json')) == 80
                break
            write(QUEUE / 'status.json', dict(stage='waiting', pid=os.getpid(),
                  previous_epoch=status.get('epoch'), previous_stage=status['stage']))
            time.sleep(30)
        assert not OUT.exists(), 'Output created while waiting'
        assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h
                   for p, h in hashes.items()), 'Source changed after scheduling'
        with (QUEUE / 'training.stdout.log').open('w') as stdout, (QUEUE / 'training.stderr.log').open('w') as stderr:
            child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
            write(QUEUE / 'status.json', dict(stage='training_and_test', pid=child.pid, out=str(OUT)))
            assert child.wait() == 0, 'Training failed; inspect logs'
        done = read(OUT / 'late10/complete.json')
        assert done['epoch'] == 1000 and done['test_count'] == 80
        write(QUEUE / 'status.json', dict(stage='complete', out=str(OUT)))
    except Exception as error:
        write(QUEUE / 'status.json', dict(stage='failed', error=repr(error)))
        raise


if __name__ == '__main__':
    main()
