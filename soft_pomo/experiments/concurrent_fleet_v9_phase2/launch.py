"""Sequential cache generation, phase2 training and causal online test."""
import subprocess
import sys
from experiments.concurrent_fleet_v9_phase2.support import ROOT, OUT, PURE
from experiments.concurrent_fleet_pair.io import write


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'launch.lock').open('x') as lock:
        lock.write('Single pipeline owner')
    tasks = [
        ['-m','experiments.concurrent_fleet_v9_phase2.prepare','--paid'],
        ['-m','experiments.concurrent_fleet_v9_phase2.train','--out',str(OUT/'training'),
         '--weights','10','--resume-from',str(PURE)],
    ]
    for stage, command in zip(('cache_generation','phase2_training_and_test'), tasks):
        write(OUT/'pipeline.json', dict(stage=stage))
        with (OUT/(stage+'.stdout.log')).open('w') as out, (OUT/(stage+'.stderr.log')).open('w') as err:
            result = subprocess.run([sys.executable,'-u',*command], cwd=ROOT, stdout=out, stderr=err)
        if result.returncode:
            write(OUT/'pipeline.json', dict(stage='failed', failed_stage=stage, code=result.returncode))
            return
    write(OUT/'pipeline.json', dict(stage='complete'))


if __name__ == '__main__': main()
