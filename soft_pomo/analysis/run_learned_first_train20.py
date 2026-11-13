"""RC101 learned-first policy gradient: sampled training, single greedy evaluation."""
import argparse
import copy
import json
import math
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'result/bks_normalized_late10_train20_20261013/rc101'
OUT = ROOT / 'result/rc101_learned_first_train20_20261017'
sys.path.insert(0, str(ROOT / 'server'))
import train_cached as io


def prepare():
    if (OUT / 'manifest.json').exists():
        verify()
        return
    bundle = OUT / 'source'
    bundle.mkdir(parents=True, exist_ok=True)
    # Freeze the complete local modules used by the trainer, independently of old runs.
    for source in ROOT.glob('*.py'):
        shutil.copy2(source, bundle / source.name)
    shutil.copytree(BASE / 'data', bundle / 'data', dirs_exist_ok=True)
    shutil.copy2(BASE / 'split.json', bundle / 'split.json')
    shutil.copy2(Path(__file__), bundle / 'experiment_runner.py')
    shutil.copy2(ROOT / 'server/train_cached.py', bundle / 'train_cached.py')
    config = io.read(BASE / 'configs/pure.json')
    config['model']['learn_first_customer'] = True
    config['trainer'].update(use_augmentation=False, n_mc_samples=1)
    for weight in (1, 5, 10, 20):
        current = copy.deepcopy(config)
        current['environment']['late_penalty'] = weight
        io.write(OUT / f'late{weight}/config.json', current)
    io.write(OUT / 'manifest.json', {
        'method': 'learned-first sampled policy gradient, not original POMO',
        'training': 'from scratch, seed42, 100 sampled rollouts, 1000 epochs',
        'evaluation': 'test80, no augmentation, one greedy rollout including first customer',
        'files': {str(p.relative_to(OUT)): io.sha(p) for p in OUT.rglob('*')
                  if p.is_file() and p.name != 'manifest.json'},
    })


def verify():
    for name, digest in io.read(OUT / 'manifest.json')['files'].items():
        if io.sha(OUT / name) != digest:
            raise ValueError(f'Experiment input changed: {name}')


def build(weight, evaluation=False):
    verify()
    sys.path.insert(0, str(OUT / 'source'))
    import torch
    config = io.read(OUT / f'late{weight}/config.json')
    split = io.read(OUT / 'source/split.json')
    assert len(split['train']) == 20 and len(split['test']) == 80
    assert not {r['name'] for r in split['train']} & {r['name'] for r in split['test']}
    if evaluation:
        config['environment']['pomo_size'] = 1
        config['trainer']['train_instances'] = [r['name'] for r in split['test']]
        config['trainer']['pomo_auto'] = False
        config['trainer']['eval_top_k'] = None
    trainer = io.build_trainer(config, OUT / 'source', OUT / f'late{weight}/runtime')
    return trainer, config, torch


def smoke(weight):
    trainer, config, torch = build(weight)
    import train_vrptw as pure
    seen = []

    def observe(module, args, output):
        state = args[0]
        if state.selected_count == 1:
            chosen, probability = output
            assert (chosen > 0).all(), 'First customer must not be depot'
            assert probability is not None and probability.requires_grad
            probability.retain_grad()
            seen.append(probability)

    handle = trainer.model.register_forward_hook(observe)
    for kind in ('RAIN', 'ACC'):
        inst = next(i for i in trainer.instance_pool if '_' + kind + '_' in i['name'])
        seen.clear()
        reward, loss = trainer._train_one_batch(inst, 1)
        assert math.isfinite(reward) and math.isfinite(loss)
        assert len(seen) == 1 and seen[0].grad is not None
        assert torch.isfinite(seen[0].grad).all() and seen[0].grad.abs().sum() > 0
        print(f'PASS {kind}: first-customer gradient, reward={reward}, loss={loss}', flush=True)
    handle.remove()
    trainer.model.eval()
    trainer.env_params['pomo_size'] = 1
    with torch.no_grad():
        pure._set_seed(42)
        first = trainer._best_solution(inst)
        pure._set_seed(99)
        second = trainer._best_solution(inst)
    assert first == second, 'Greedy result must not depend on random forced starts'
    assert trainer.env.pomo_size == 1
    io.write(OUT / 'smoke.json', {'status': 'passed', 'weight': weight,
                                 'first_customer_gradient': True, 'deterministic_greedy': True})


def train(weight):
    trainer, config, torch = build(weight)
    out = OUT / f'late{weight}/training'
    contract = dict(bundle_sha256=io.sha(OUT / 'manifest.json'), config=config)
    checkpoint = out / 'checkpoint-last.pt'
    epoch = io.restore(trainer, checkpoint, contract) if checkpoint.exists() else 0
    io.write(out / 'settings.json', contract)
    while epoch < config['trainer']['epochs'] and not (OUT / 'STOP').exists():
        start = time.monotonic()
        trainer._current_epoch = epoch + 1
        reward, loss = trainer._train_one_epoch(epoch + 1)
        if not math.isfinite(reward) or not math.isfinite(loss):
            raise RuntimeError('Non-finite epoch')
        epoch += 1
        trainer.reward_log.append(reward)
        trainer.loss_log.append(loss)
        if epoch % 10 == 0 or (OUT / 'STOP').exists():
            io.save_checkpoint(trainer, epoch, checkpoint, contract)
        if epoch in (500, 1000):
            io.save_checkpoint(trainer, epoch, out / f'checkpoint-{epoch}.pt', contract)
        io.write(out / 'status.json', dict(epoch=epoch, target=1000, status='training'))
        print(f'lambda={weight} epoch={epoch} reward={reward:.6f} loss={loss:.6f} seconds={time.monotonic()-start:.1f}', flush=True)
    io.save_checkpoint(trainer, epoch, checkpoint, contract)
    io.write(out / 'status.json', dict(epoch=epoch, target=1000,
                                     status='complete' if epoch == 1000 else 'needs_resume'))


def evaluate(weight):
    trainer, config, torch = build(weight, evaluation=True)
    import train_vrptw as pure
    from vrptw_env import _solution_stats
    checkpoint = OUT / f'late{weight}/training/checkpoint-1000.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert state['epoch'] == 1000
    trainer.model.load_state_dict(state['model_state_dict'], strict=True)
    trainer.model.eval()
    out = OUT / f'late{weight}/eval_1000_test80_oneshot'
    io.write(out / 'contract.json', dict(config=config, checkpoint_sha256=io.sha(checkpoint),
                                       selection='single greedy, learned first customer'))
    entries = io.read(OUT / 'source/split.json')['test']
    results = io.read(out / 'results.json') if (out / 'results.json').exists() else []
    assert [r['name'] for r in results] == [e['name'] for e in entries[:len(results)]]
    for index, inst in enumerate(trainer.instance_pool):
        if index < len(results):
            continue
        entry = entries[index]
        assert inst['name'] == entry['name']
        pure._set_seed(42 + index)
        with torch.no_grad():
            routes, reward = trainer._best_solution(inst)
        assert trainer.env.pomo_size == 1
        routes = [[int(n) for n in route] for route in routes]
        stats = _solution_stats(inst, routes)
        assert stats['n_served'] == 100
        row = dict(name=inst['name'], kind=entry['kind'], routes=routes, reward=float(reward),
                   vehicles=len(stats['per_vehicle']), travel_time=float(stats['total_travel_time']),
                   total_late=float(stats['total_late']), late_count=int(stats['late_count']))
        row['cost'] = row['vehicles']/14 + (row['travel_time']+weight*row['total_late'])/1696.95
        assert abs(row['cost'] + row['reward']) < 1e-4
        results.append(row)
        io.write(out / 'results.json', results)
        io.write(out / 'status.json', dict(status='evaluating', completed=len(results), total=80))
    keys = ('reward', 'cost', 'vehicles', 'travel_time', 'total_late', 'late_count')
    io.write(out / 'summary.json', {k: dict(mean=statistics.mean(r[k] for r in results),
                                         std=statistics.stdev(r[k] for r in results)) for k in keys})
    io.write(out / 'status.json', dict(status='complete', completed=80))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'smoke', 'train', 'evaluate', 'queue'])
    parser.add_argument('--weight', type=int, choices=[1, 5, 10, 20], default=10)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare()
    elif args.stage == 'queue':
        verify()
        assert io.read(OUT / 'smoke.json')['status'] == 'passed'
        for weight in (1, 5, 10, 20):
            for stage in ('train', 'evaluate'):
                if (OUT / 'STOP').exists():
                    io.write(OUT / 'queue_status.json', dict(status='stopped', weight=weight, stage=stage))
                    return
                io.write(OUT / 'queue_status.json', dict(status='running', weight=weight, stage=stage))
                subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), stage,
                                '--weight', str(weight)], cwd=ROOT, check=True)
        io.write(OUT / 'queue_status.json', dict(status='complete'))
    else:
        {'smoke': smoke, 'train': train, 'evaluate': evaluate}[args.stage](args.weight)


if __name__ == '__main__':
    main()
