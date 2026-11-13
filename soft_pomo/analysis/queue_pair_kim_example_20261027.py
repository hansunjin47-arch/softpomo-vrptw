"""Wait for existing local Kim training AND test80 before the new experiment."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
from experiments.concurrent_fleet_pair.io import write
from experiments.concurrent_fleet_bias_guided.support import ROOT, read
from analysis.pair_kim_rl_example import OUT, ART, PURE

PREVIOUS = ROOT/'result/pair300_epsilon005_floor001_std_kim_20261026'
QUEUE = ROOT/'result/queue_pair_kim_example_20261027'


def main():
    QUEUE.mkdir(exist_ok=False)
    try:
        assert not OUT.exists(), 'Output already exists'
        assert os.environ.get('OPENROUTER_API_KEY'), 'Missing API key'
        files = list((ROOT/'experiments').rglob('*.py'))
        files += [Path(__file__), ROOT/'analysis/pair_kim_rl_example.py']
        hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        commands = [[sys.executable, '-u', '-m', 'analysis.pair_kim_rl_example', stage, '--allow-api']
                    for stage in ('prepare', 'run')]
        checkpoint = PURE/'late10/checkpoint-1000.pt'
        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        write(QUEUE/'protocol.json', dict(previous=str(PREVIOUS), out=str(OUT), artifacts=str(ART),
            commands=commands, source_hashes=hashes, reference_checkpoint=str(checkpoint),
            reference_checkpoint_sha256=checkpoint_hash, cache_creation='after predecessor completes test80',
            settings='fresh seed42; 1000 epochs; vehicle300; late10; batch128 x 100 rollouts; stdON; epsilon .05 to .01; beta5; gateOFF'))
        while True:
            if (QUEUE/'STOP').exists():
                write(QUEUE/'status.json', dict(stage='cancelled_before_launch'))
                return
            status = read(PREVIOUS/'training/status.json')
            if (PREVIOUS/'training/failure.json').exists() or status['stage'] in ('failed', 'paused'):
                raise RuntimeError('Previous experiment stopped; refusing automatic launch')
            if status['stage'] == 'complete':
                complete = read(PREVIOUS/'training/late10/complete.json')
                assert complete['epoch'] == 1000 and complete['test_count'] == 80
                break
            write(QUEUE/'status.json', dict(stage='waiting', previous_stage=status['stage'],
                  previous_epoch=status.get('epoch'), pid=os.getpid()))
            time.sleep(30)
        for stage, command in zip(('generating_cache', 'training_and_test'), commands):
            assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == h for p,h in hashes.items()), 'Source changed after scheduling'
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_hash
            if (QUEUE/'STOP').exists():
                write(QUEUE/'status.json', dict(stage='cancelled_before_launch'))
                return
            with (QUEUE/f'{stage}.stdout.log').open('w') as stdout, (QUEUE/f'{stage}.stderr.log').open('w') as stderr:
                child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
                write(QUEUE/'status.json', dict(stage=stage, pid=child.pid, out=str(OUT)))
                assert child.wait() == 0, f'{stage} failed; inspect logs'
        complete = read(OUT/'training/late10/complete.json')
        assert complete['epoch'] == 1000 and complete['test_count'] == 80
        write(QUEUE/'status.json', dict(stage='complete', out=str(OUT)))
    except Exception as error:
        write(QUEUE/'status.json', dict(stage='failed', error=repr(error)))
        raise


if __name__ == '__main__':
    main()
