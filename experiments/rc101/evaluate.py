"""Evaluate a trusted local checkpoint with cached guidance; never call an API."""
import argparse
import copy
from pathlib import Path
import statistics

from server.train_cached import build_trainer, read, sha, verify_bundle, write

ROOT = Path(__file__).resolve().parent
METRICS = ('vehicles', 'travel_time', 'late_count', 'total_late', 'cost', 'reward')


def replay(inst, routes):
    from vrptw_env import _solution_stats
    customers = [int(n) for route in routes for n in route if n]
    if sorted(customers) != list(range(1, inst['n_customers'] + 1)):
        raise ValueError('Incomplete or duplicated service')
    stats = _solution_stats(inst, routes)
    row = dict(name=inst['name'], vehicles=len(stats['per_vehicle']),
               travel_time=stats['total_travel_time'], late_count=stats['late_count'],
               total_late=stats['total_late'], served=stats['n_served'], routes=routes)
    row['cost'] = .04 * row['vehicles'] + row['travel_time'] / 5760 + .0075 * row['total_late']
    row['reward'] = -row['cost']
    return row


def summarize(rows):
    return {kind: {
        'count': len(selected),
        'mean': {key: statistics.mean(r[key] for r in selected) for key in METRICS},
        'std': {key: statistics.stdev(r[key] for r in selected) if len(selected) > 1 else None
                for key in METRICS},
    } for kind in ('RAIN', 'ACCIDENT', 'ALL')
        if (selected := [r for r in rows if kind == 'ALL' or r['kind'] == kind])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', default='checkpoint-1000.pt', help='Filename inside --run')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--smoke', action='store_true', help='Only one rain and one accident')
    parser.add_argument('--cpu', action='store_true', help='CPU smoke evaluation only')
    args = parser.parse_args()
    if args.cpu and not args.smoke:
        parser.error('--cpu requires --smoke')
    digest = verify_bundle(ROOT)
    run = args.run.resolve()
    settings = read(run / 'settings.json')
    if settings['bundle_sha256'] != digest:
        raise ValueError('Training bundle differs from this evaluation bundle')
    config = copy.deepcopy(settings['config'])
    checkpoint = run / args.checkpoint
    out = (args.out or run / ('eval_smoke' if args.smoke else 'eval_test80')).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a new output directory')
    split = read(ROOT / 'data/split.json')
    entries = split['test']
    if len(entries) != 80 or set(e['name'] for e in entries) & set(config['trainer']['train_instances']):
        raise ValueError('Invalid test split')
    if args.smoke:
        entries = [next(e for e in entries if e['kind'] == kind) for kind in ('RAIN', 'ACCIDENT')]
    config['trainer'].update(use_augmentation=True, n_mc_samples=1, pomo_auto=False, eval_top_k=None)
    config['model']['eval_type'] = 'argmax'
    trainer = build_trainer(config, ROOT, out / 'runtime', cpu=args.cpu)
    import torch
    from train_vrptw import _set_seed
    from vrptw_env import load_solomon
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state.get('server_contract') != settings:
        raise ValueError('Checkpoint does not belong to this run')
    trainer.model.load_state_dict(state['model_state_dict'], strict=True)
    trainer.model.eval()
    write(out / 'protocol.json', dict(method=config['method'], epoch=state['epoch'],
        checkpoint_sha256=sha(checkpoint), bundle_sha256=digest, augmentation=8,
        starts=100, smoke=args.smoke, seed='42 + original test index'))
    names = [e['name'] for e in split['test']]
    rows = []
    for entry in entries:
        name = entry['name']
        _set_seed(42 + names.index(name))
        inst = load_solomon(str(ROOT / 'data' / (name + '.txt')))
        with torch.no_grad():
            routes, reward = (trainer._best_solution(inst) if config['method'] == 'pure'
                              else trainer._best_solution_aug(inst, n_aug=8))
        row = replay(inst, routes)
        row.update(kind=entry['kind'], decoder_reward=float(reward))
        if abs(row['reward'] - reward) > 1e-4:
            raise ValueError('Decoder reward and full-service replay objective differ')
        rows.append(row)
        write(out / 'results.json', rows)
        print(f'{len(rows)}/{len(entries)} {name}: J={row["cost"]:.6f}', flush=True)
    write(out / 'summary.json', summarize(rows))
    print(f'Complete: {out}', flush=True)


if __name__ == '__main__':
    main()
