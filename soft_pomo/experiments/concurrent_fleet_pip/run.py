"""Wait for the C5 queue, preflight, then fresh C10 pure-RL soft-PIP training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from experiments.concurrent_fleet_pair.io import write

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'result/v9_softpip_c10_300_late10_20261023_fix1'
PARENT = ROOT / 'result/v9_routeclip5_300_late10_20261023'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def hashes():
    paths = list(Path(__file__).parent.glob('*.py'))
    for module in ('stop_init', 'exploration', 'v9', 'v8', 'v7', 'v5', 'v3', 'pair'):
        paths.extend((ROOT / ('experiments/concurrent_fleet_' + module)).glob('*.py'))
    paths.extend(ROOT / p for p in ('train_vrptw.py', 'vrptw_env.py', 'event_observation.py'))
    paths.append(ROOT / 'result/rc101_learned_first_train20_20261017/late10/config.json')
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def train():
    from experiments.concurrent_fleet_stop_init import train as trainer
    from experiments.concurrent_fleet_pip.policy import FleetPolicy
    trainer.BiasedPolicy = FleetPolicy
    cfg = read(ROOT / 'result/rc101_learned_first_train20_20261017/late10/config.json')
    assert cfg['model']['logit_clipping'] == 10
    assert cfg['trainer']['seed'] == 42 and cfg['trainer']['epochs'] == 1000
    original = trainer.train_batch
    def measured(model, *a, **kw):
        model.risk_stats = next(model.parameters()).new_zeros(3)
        summary, detail = original(model, *a, **kw)
        total, count, nonzero = model.risk_stats.tolist()
        summary.update(pip_penalty_mean=total/max(count, 1),
                       pip_nonzero_fraction=nonzero/max(count, 1), pip_beta=1., pip_clip=2.)
        return summary, detail
    trainer.train_batch = measured
    dest = OUT / 'pure'
    args = argparse.Namespace(arm='stop_bias', stop_target=.15, max_hours=24.,
        entropy_beta=1., entropy_zero_epoch=500, vehicle_minutes=300,
        first_entropy_beta=0., out=dest, weights=[10], resume=dest.exists(),
        resume_from=None, smoke_updates=0)
    while True:
        trainer.run(args)
        stage = read(dest / 'status.json')['stage']
        if stage != 'needs_resume':
            assert stage == 'complete', stage
            break
        args.resume = True


def queue():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / 'queue.lock'
    with lock.open('x') as f:
        f.write(str(os.getpid()))
    try:
        expected = hashes()
        contract = dict(source_hashes=expected, parent=str(PARENT), epochs=1000,
            seed=42, vehicle_minutes=300, late_weight=10, clipping=10,
            stop_target=.15, dispatch_entropy='1 to 0 at500', first_entropy=0,
            method='soft fleet-aware one-step PIP-inspired; NOT PIP-D',
            reward='unchanged', risk_beta=1., max_logit_penalty=2.,
            travel_estimator='nominal base; no event multiplier or hidden future ETA',
            risk='positive increase of per-customer optimistic late minutes, capped at horizon',
            inference='same guidance; greedy one-shot test80; no LLM calls',
            reference='https://github.com/jieyibi/PIP-constraint/blob/main/POMO%2BPIP/envs/TSPTWEnv.py')
        if (OUT / 'contract.json').exists():
            assert read(OUT / 'contract.json') == contract
        else:
            write(OUT / 'contract.json', contract)
        for name in expected:
            dest = OUT / 'source' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest)
        while True:
            if (OUT / 'STOP').exists():
                write(OUT / 'queue_status.json', dict(stage='paused')); return
            parent = read(PARENT / 'queue_status.json')['stage']
            if parent in ('failed', 'paused'):
                raise RuntimeError('Parent queue failed/paused; not starting GPU training')
            write(OUT / 'queue_status.json', dict(stage='waiting', parent_stage=parent, pid=os.getpid()))
            if parent == 'complete' and not (PARENT / 'queue.lock').exists():
                for rel in ('pure/late10/complete.json', 'guided/phase2/training/late10/complete.json'):
                    done = read(PARENT / rel)
                    assert done['epoch'] == 1000 and done['test_count'] == 80
                break
            time.sleep(30)
        assert hashes() == expected, 'Source changed while waiting'
        for stage in ('preflight', 'train'):
            assert hashes() == expected, 'Source changed before next stage'
            write(OUT / 'queue_status.json', dict(stage=stage, pid=os.getpid()))
            module = 'experiments.concurrent_fleet_pip.tests' if stage == 'preflight' else 'experiments.concurrent_fleet_pip.run'
            args = ['--gpu'] if stage == 'preflight' else ['train']
            with (OUT / (stage+'.stdout.log')).open('a') as stdout, (OUT / (stage+'.stderr.log')).open('a') as stderr:
                subprocess.run([sys.executable, '-u', '-m', module, *args], cwd=ROOT,
                               stdout=stdout, stderr=stderr, check=True)
        write(OUT / 'queue_status.json', dict(stage='complete'))
    except Exception:
        write(OUT / 'queue_status.json', dict(stage='failed', error=traceback.format_exc()))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['queue', 'train'])
    globals()[parser.parse_args().stage]()
