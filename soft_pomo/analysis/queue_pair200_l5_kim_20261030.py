"""Generate matched new-Kim caches, then fresh training and causal test80."""
import os
from pathlib import Path
import subprocess
import sys
from experiments.concurrent_fleet_pair.io import write
from experiments.concurrent_fleet_bias_guided.support import ROOT, read

PURE = ROOT/'result/pair200_late5_epsilon005_floor001_std_20261029'
OUT = ROOT/'result/pair200_late5_std_kim_example_20261030'
ART = ROOT/'result/pair200_late5_kim_cache_20261030'
QUEUE = ROOT/'result/queue_pair200_late5_kim_20261030'


def main():
    QUEUE.mkdir(exist_ok=False)
    try:
        assert os.environ.get('OPENROUTER_API_KEY'), 'Missing API key'
        assert not OUT.exists(), 'Output exists'
        done = read(PURE/'late5/complete.json')
        assert done['epoch'] == 1000 and done['test_count'] == 80
        common = ['--allow-api', '--late', '5', '--vehicle-cost', '200',
                  '--pure', str(PURE), '--out', str(OUT), '--art', str(ART)]
        write(QUEUE/'protocol.json', dict(pure=str(PURE), out=str(OUT), art=str(ART),
              initialization='fresh seed42, not Phase 2', epochs=1000,
              reference='one separate fixed case, checkpoint1000, CPU greedy',
              test='80 single greedy; no target reference; causal accident refresh'))
        for stage in ('prepare', 'run'):
            with (QUEUE/f'{stage}.stdout.log').open('w') as stdout, (QUEUE/f'{stage}.stderr.log').open('w') as stderr:
                child = subprocess.Popen([sys.executable, '-u', '-m', 'analysis.pair_kim_rl_example', stage] + common,
                                         cwd=ROOT, stdout=stdout, stderr=stderr)
                write(QUEUE/'status.json', dict(stage=stage, pid=child.pid))
                if child.wait() != 0:
                    raise RuntimeError(f'{stage} failed; inspect logs')
        done = read(OUT/'training/late5/complete.json')
        assert done['epoch'] == 1000 and done['test_count'] == 80
        write(QUEUE/'status.json', dict(stage='complete'))
    except Exception as error:
        write(QUEUE/'status.json', dict(stage='failed', error=repr(error)))
        raise


if __name__ == '__main__':
    main()
