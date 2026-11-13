"""Wait for RC sweep, then masked Kim and masked Pure500 Phase 2."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.concurrent_fleet_pair.io import write
from experiments.concurrent_fleet_bias_guided.support import ROOT, read

PURE = ROOT/'result/rc_pair_returnmask_v200_l5_s42_20261031'
QUEUE = ROOT/'result/queue_mask_proposed_20261105'
PREVIOUS = ROOT/'result/queue_rc_returnmask_l5_20261102'
KIM = ROOT/'result/pair200_l5_mask_kim_20261105'
PHASE2 = ROOT/'result/pair200_l5_mask_phase2_20261105'


def stage(name):
    if name.startswith('kim-'):
        from analysis import pair_kim_rl_example as runner
        runner.PURE, runner.OUT = PURE, KIM
        runner.ART = ROOT/'result/pair200_l5_mask_kim_cache_20261105'
        runner.LATE, runner.VEHICLE_COST = 5, 200
        runner.prepare() if name == 'kim-prepare' else runner.run()
    else:
        from analysis import prepare_pair_phase2_feedback as prep
        from analysis import pair_phase2_feedback_run as runner
        art = ROOT/'result/pair200_l5_mask_phase2_cache_20261105'
        prep.PURE, prep.ART, prep.LATE = PURE, art, 5
        runner.PURE, runner.ART, runner.OUT = PURE, art, PHASE2
        runner.LATE, runner.VEHICLE_COST = 5, 200
        if name == 'phase2-prepare':
            prep.main(allow_api=True)
        elif name == 'phase2-test':
            runner.prepare_test()
        else:
            runner.run()


def main():
    assert os.environ.get('OPENROUTER_API_KEY'), 'Missing API key'
    assert not KIM.exists() and not PHASE2.exists(), 'Duplicate output'
    cfg = read(PURE/'protocol.json')['configs']['5']
    assert cfg['environment']['protect_last_service_vehicle']
    assert cfg['environment']['normalize_advantage_std']
    assert cfg['trainer']['seed'] == 42
    QUEUE.mkdir(exist_ok=False)
    files = list((ROOT/'experiments').rglob('*.py')) + list(ROOT.glob('*.py'))
    files += [Path(__file__)] + [ROOT/'analysis'/n for n in (
        'pair_kim_rl_example.py', 'prepare_pair_phase2_feedback.py', 'pair_phase2_feedback_run.py')]
    files += [PURE/f'late5/checkpoint-{epoch}.pt' for epoch in (500,1000)]
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    write(QUEUE/'protocol.json', dict(pure=str(PURE), previous=str(PREVIOUS),
        kim=str(KIM), phase2=str(PHASE2), config=cfg, hashes=hashes,
        schedule='sweep complete -> fresh Kim1000/test80 -> Pure500 Phase2 501-1000/test80'))
    try:
        while True:
            if (QUEUE/'STOP').exists():
                write(QUEUE/'status.json', dict(stage='cancelled')); return
            try:
                status = read(PREVIOUS/'status.json')
            except (PermissionError, ValueError):
                time.sleep(1); continue
            if status['stage'] == 'complete':
                break
            if status['stage'] in ('failed', 'cancelled_before_launch'):
                raise RuntimeError('Predecessor did not complete')
            write(QUEUE/'status.json', dict(stage='waiting', pid=os.getpid(), previous=status))
            time.sleep(30)
        for name in ('kim-prepare','kim-run','phase2-prepare','phase2-test','phase2-run'):
            if (QUEUE/'STOP').exists():
                write(QUEUE/'status.json', dict(stage='cancelled')); return
            assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items()), 'Source changed'
            with (QUEUE/f'{name}.stdout.log').open('w') as stdout, (QUEUE/f'{name}.stderr.log').open('w') as stderr:
                child = subprocess.Popen([sys.executable,'-u','-m','analysis.queue_mask_proposed_20261105','--stage',name], cwd=ROOT, stdout=stdout, stderr=stderr)
                write(QUEUE/'status.json', dict(stage=name,pid=child.pid))
                if child.wait()!=0:
                    raise RuntimeError(name+' failed; inspect logs')
            if name.endswith('-run'):
                out = KIM if name == 'kim-run' else PHASE2
                done = read(out/'training/late5/complete.json')
                assert done['epoch']==1000 and done['test_count']==80
                assert len(read(out/'training/late5/results.json'))==80
                actual = read(out/'training/protocol.json')['configs']['5']
                assert actual == cfg, 'Training config mismatch'
        write(QUEUE/'status.json', dict(stage='complete'))
    except Exception as error:
        write(QUEUE/'status.json', dict(stage='failed',error=repr(error)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['kim-prepare','kim-run','phase2-prepare','phase2-test','phase2-run'])
    args = parser.parse_args()
    stage(args.stage) if args.stage else main()
