"""Serialize phase two after the vehicle200 run, including its test evaluation."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

from analysis.pair_phase2_feedback_run import ROOT, ART, OUT, PURE, read, validate_artifacts
from experiments.concurrent_fleet_pair.io import write

PREVIOUS = ROOT/'result/pair200_epsilon005_floor001_std_20261028'
QUEUE = ROOT/'result/queue_pair_phase2_after200_20261028'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    QUEUE.mkdir(exist_ok=False)
    try:
        assert not OUT.exists(), 'Refusing duplicate training output'
        assert os.environ.get('OPENROUTER_API_KEY'), 'Missing live test API key'
        validate_artifacts()
        assert read(ART/'test_ready.json')['instances'] == 80
        files = list((ROOT/'experiments').rglob('*.py'))
        files += [ROOT/'analysis'/n for n in ('prepare_pair_phase2_feedback.py',
                    'pair_phase2_feedback_run.py', 'queue_pair_phase2_after200.py')]
        files += list(ART.rglob('*.json'))
        files += [PURE/'late10/checkpoint-500.pt']
        hashes = {str(p.relative_to(ROOT)):sha(p) for p in files}
        command = [sys.executable, '-u', '-m', 'analysis.pair_phase2_feedback_run', 'run']
        write(QUEUE/'protocol.json', dict(previous=str(PREVIOUS), out=str(OUT), command=command,
            hashes=hashes, schedule='wait for epoch1000 and test80 complete; then restore Pure epoch500 and train through1000'))
        while True:
            if (QUEUE/'STOP').exists():
                write(QUEUE/'status.json', dict(stage='cancelled_before_launch'))
                return
            status = read(PREVIOUS/'status.json')
            if (PREVIOUS/'failure.json').exists() or status['stage'] in ('failed', 'paused'):
                raise RuntimeError('Predecessor failed/paused; refusing launch')
            if status['stage'] == 'complete':
                done = read(PREVIOUS/'late10/complete.json')
                assert done['epoch'] == 1000 and done['test_count'] == 80
                assert len(read(PREVIOUS/'late10/results.json')) == 80
                break
            write(QUEUE/'status.json', dict(stage='waiting', pid=os.getpid(),
                  previous_epoch=status.get('epoch'), previous_stage=status['stage']))
            time.sleep(30)
        assert all(sha(ROOT/p) == h for p,h in hashes.items()), 'Source/cache/checkpoint changed'
        with (QUEUE/'training.stdout.log').open('w') as stdout, (QUEUE/'training.stderr.log').open('w') as stderr:
            child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
            write(QUEUE/'status.json', dict(stage='training_and_test', pid=child.pid, out=str(OUT)))
            assert child.wait() == 0, 'Phase two failed; inspect logs'
        done = read(OUT/'training/late10/complete.json')
        assert done['epoch'] == 1000 and done['test_count'] == 80
        write(QUEUE/'status.json', dict(stage='complete', out=str(OUT)))
    except Exception as error:
        write(QUEUE/'status.json', dict(stage='failed', error=repr(error)))
        raise


if __name__ == '__main__':
    main()
