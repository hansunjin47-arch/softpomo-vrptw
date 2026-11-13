"""Isolated Pure RL reward-weight experiment; no LLM calls or baseline edits."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import sys
import time


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def prepare(run, variant='A', pool=80, objective='distance'):
    workspace = Path(__file__).resolve().parents[1]
    bundle = workspace / 'server_export/soft_pomo_rl_pure128'
    data = workspace / 'data/rc101_train80_test80_20260929'
    if pool != 80:
        bundle = workspace / f'server_export/soft_pomo_seed_sweep/train{pool}'
        data = bundle / 'data'
    if run.exists():
        raise FileExistsError(run)
    config = read(bundle / 'configs/pure.json')
    split = read(data / 'split.json' if pool == 80 else bundle / 'eval_addon/split.json')
    assert set(config['trainer']['train_instances']) == {e['name'] for e in split['train']}
    assert len(split['train']) == pool and len(split['test']) == 80
    assert not {e['name'] for e in split['train']} & {e['name'] for e in split['test']}
    source = run / 'source'
    (source / 'server').mkdir(parents=True)
    (source / 'data').mkdir()
    for path in bundle.glob('*.py'):
        shutil.copy2(path, source / path.name)
    shutil.copy2(bundle / 'server/train_cached.py', source / 'server/train_cached.py')
    for entry in split['train'] + split['test']:
        path = data / (entry['name'] + '.txt')
        if not path.exists() and pool != 80:
            path = bundle / 'eval_addon/data' / (entry['name'] + '.txt')
        assert sha(path) == entry['sha256'], path
        shutil.copy2(path, source / 'data' / path.name)
    shutil.copy2(Path(__file__), run / 'runner.py')
    write(run / 'baseline_config.json', config)
    config = copy.deepcopy(config)
    config['environment'].update(vehicle_penalty=6.0 if variant == 'A' else 4.0, late_penalty=3.0,
        D_max=12.0 if variant == 'A' else 8.0, Lt_max=5.0, dist_penalty_weight=1.0, time_penalty=0.0)
    if objective == 'travel_time':
        config['environment'].update(dist_penalty_weight=0.0, time_penalty=1.0,
            TT_max=12.0 if variant == 'A' else 8.0)
    config['trainer'].update(seed=42, epochs=1000, train_batch_size=128)
    write(run / 'config.json', config)
    write(run / 'split.json', split)
    write(run / 'protocol.json', dict(
        variant=variant,
        objective=(f'J_new = 0.06*K + {"T" if objective == "travel_time" else "D"}/2880 + 0.0025*L' if variant == 'A'
                   else f'J_new = 0.04*K + {"T" if objective == "travel_time" else "D"}/1920 + 0.0025*L'),
        optimized_travel_metric=objective,
        travel_time='Actual event-dependent driving time including depot returns; excludes waiting and service',
        interpretation='Weighted preference, NOT strict lexicographic ordering or LKH equivalence',
        distance='Free-flow geometric route length, excluding event multipliers',
        old_objective='J_E = 0.04*K + T/5760 + 0.0075*L',
        training=f'Fresh initialization; batch128; seed42; train{pool}; 1000 epochs',
        evaluation='Frozen test80; 8 augmentations x 100 starts; selection by J_new',
        api_calls=False))
    files = [p for p in source.rglob('*') if p.is_file()]
    files += [run / name for name in ('runner.py', 'config.json', 'split.json', 'protocol.json')]
    write(run / 'manifest.json', {str(p.relative_to(run)): sha(p) for p in files})
    print(f'Prepared {run}', flush=True)


def execute(run, smoke=False, resume=False):
    for name, digest in read(run / 'manifest.json').items():
        if sha(run / name) != digest:
            raise ValueError(f'Snapshot changed: {name}')
    source = run / 'source'
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / 'server'))
    import train_cached as runner
    import torch
    config = read(run / 'config.json')
    is_tt = read(run / 'protocol.json').get('optimized_travel_metric') == 'travel_time'
    if is_tt:
        env = config['environment']
        assert env['dist_penalty_weight'] == 0 and env['time_penalty'] == 1
        assert env['TT_max'] == (12 if read(run / 'protocol.json')['variant'] == 'A' else 8)
    contract = dict(config=config, manifest_sha256=sha(run / 'manifest.json'))
    out = run / ('smoke' if smoke else 'training')
    if out.exists() and not resume:
        raise FileExistsError(out)
    trainer = runner.build_trainer(config, source, out)
    write(out / 'settings.json', contract)
    print(f'GPU={torch.cuda.get_device_name(0)} torch={torch.__version__} batch=128', flush=True)
    print(json.dumps(config['environment']), flush=True)
    if smoke:
        for kind in ('RAIN', 'ACC'):
            inst = next(i for i in trainer.instance_pool if '_' + kind + '_' in i['name'])
            result = trainer._train_one_batch(inst, 128)
            assert all(math.isfinite(float(v)) for v in result)
            print(f'SMOKE {kind}: {result}', flush=True)
        runner.save_checkpoint(trainer, 0, out / 'checkpoint.pt', contract)
        runner.restore(trainer, out / 'checkpoint.pt', contract)
        if is_tt:
            from vrptw_env import _solution_stats
            # Use a known feasible policy only for reward validation; production
            # training starts in a separate process with freshly seeded weights.
            validation_checkpoint = Path(__file__).resolve().parents[1] / 'rc101_pure_train20_b128_s42_configA_1000_followup/training/checkpoint-1000.pt'
            validation_state = torch.load(validation_checkpoint, map_location='cpu', weights_only=False)
            trainer.model.load_state_dict(validation_state['model_state_dict'], strict=True)
            trainer.model.eval()
            checks = []
            for kind in ('RAIN', 'ACC'):
                inst = next(i for i in trainer.instance_pool if '_' + kind + '_' in i['name'])
                with torch.no_grad():
                    routes, reward = trainer._best_solution(inst)
                stats = _solution_stats(inst, routes)
                assert stats['n_served'] == 100
                travel = float(stats['total_travel_time'])
                distance = float(stats['total_dist'])
                expected = (env['vehicle_penalty']*len(stats['per_vehicle'])/100
                    + travel/(float(inst['T'])*env['TT_max'])
                    + env['late_penalty']*float(stats['total_late'])/(float(inst['T'])*env['Lt_max']))
                assert math.isclose(-float(reward), expected, rel_tol=1e-5, abs_tol=1e-4)
                distance_counterfactual = expected + (distance-travel)/(float(inst['T'])*env['TT_max'])
                checks.append(dict(kind=kind, actual_reward=float(reward), expected_cost=expected,
                    travel_time=travel, distance=distance, distance_substitution_cost=distance_counterfactual))
            assert any(abs(c['expected_cost']-c['distance_substitution_cost']) > 1e-3 for c in checks)
            write(out / 'travel_time_reward_validation.json', checks)
            print('PASS: actual rain/accident rewards match travel-time cost; distance substitution differs', flush=True)
        print('PASS: batch128 forward/backward, both events, checkpoint roundtrip', flush=True)
        return
    epoch = runner.restore(trainer, out / 'checkpoint-last.pt', contract) if resume else 0
    for epoch in range(epoch + 1, 1001):
        start = time.monotonic()
        trainer._current_epoch = epoch
        reward, loss = trainer._train_one_epoch(epoch)
        assert math.isfinite(reward) and math.isfinite(loss)
        trainer.reward_log.append(reward)
        trainer.loss_log.append(loss)
        if epoch == 1 or epoch % 10 == 0:
            runner.save_checkpoint(trainer, epoch, out / 'checkpoint-last.pt', contract)
        if epoch in (500, 1000):
            runner.save_checkpoint(trainer, epoch, out / f'checkpoint-{epoch}.pt', contract)
        write(run / 'status.json', dict(status='training', epoch=epoch, target=1000,
            reward=reward, loss=loss, seconds=time.monotonic() - start))
        print(f'Epoch {epoch}/1000 reward={reward:.6f} loss={loss:.6f} seconds={time.monotonic()-start:.1f}', flush=True)
    evaluate(run, trainer)


def evaluate(run, trainer, epoch=1000, output_root=None):
    import torch
    from train_vrptw import _set_seed
    from vrptw_env import _solution_stats, load_solomon
    trainer.model.eval()
    entries = read(run / 'split.json')['test']
    output_root = output_root or run
    out = output_root / f'eval_{epoch}_test80'
    rows = []
    env = read(run / 'config.json')['environment']
    for index, entry in enumerate(entries):
        _set_seed(42 + index)
        inst = load_solomon(str(run / 'source/data' / (entry['name'] + '.txt')))
        with torch.no_grad():
            routes, reward = trainer._best_solution(inst)
        stats = _solution_stats(inst, routes)
        assert sorted(int(n) for route in routes for n in route if n) == list(range(1, 101))
        row = dict(name=entry['name'], kind=entry['kind'], routes=routes,
            vehicles=len(stats['per_vehicle']), distance=float(stats['total_dist']),
            travel_time=float(stats['total_travel_time']), late_count=int(stats['late_count']),
            total_late=float(stats['total_late']), served=int(stats['n_served']), reward=float(reward))
        row['cost_new'] = (env['vehicle_penalty']*row['vehicles']/100
            + env['dist_penalty_weight']*row['distance']/(inst['T']*env['D_max'])
            + env.get('time_penalty', 0)*row['travel_time']/(inst['T']*env.get('TT_max', 1))
            + env['late_penalty']*row['total_late']/(inst['T']*env['Lt_max']))
        row['cost_E'] = .04*row['vehicles'] + row['travel_time']/5760 + .0075*row['total_late']
        if not math.isclose(-row['reward'], row['cost_new'], rel_tol=1e-4, abs_tol=1e-4):
            raise ValueError(f'Replay/reward mismatch: {row}')
        rows.append(row)
        write(out / 'results.json', rows)
        write(output_root / 'status.json', dict(status='evaluating', epoch=epoch, completed=len(rows), total=80))
        print(f'EVAL {len(rows)}/80 {row["name"]} J_new={row["cost_new"]:.6f} J_E={row["cost_E"]:.6f}', flush=True)
    metrics = ('vehicles', 'distance', 'travel_time', 'late_count', 'total_late', 'cost_new', 'cost_E')
    summary = {}
    for kind in ('ALL', 'RAIN', 'ACCIDENT'):
        group = [r for r in rows if kind == 'ALL' or r['kind'] == kind]
        summary[kind] = {m: dict(mean=statistics.mean(r[m] for r in group),
            std=statistics.stdev(r[m] for r in group)) for m in metrics}
    write(out / 'summary.json', summary)
    write(output_root / 'status.json', dict(status='complete', epoch=epoch, evaluated=80))


def evaluate_at(run, output, trainer, epoch):
    evaluate(run, trainer, epoch=epoch, output_root=output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'smoke', 'train'))
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--variant', choices=('A', 'C'), default='A')
    parser.add_argument('--pool', type=int, choices=(20, 80, 200), default=80)
    parser.add_argument('--objective', choices=('distance', 'travel_time'), default='distance')
    args = parser.parse_args()
    run = args.run.resolve()
    if args.mode == 'prepare':
        prepare(run, args.variant, args.pool, args.objective)
    else:
        try:
            execute(run, smoke=args.mode == 'smoke', resume=args.resume)
        except Exception as exc:
            write(run / 'failure.json', dict(mode=args.mode, error=repr(exc)))
            raise
