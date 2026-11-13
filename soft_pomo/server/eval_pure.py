"""Offline frozen-test evaluation using the training bundle's original code."""
import argparse
import copy
import json
from pathlib import Path
import sys

ADDON = Path(__file__).resolve().parent
ROOT = ADDON.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'server'))
from train_cached import block_network, read, sha, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='Evaluate one rain and one accident')
    parser.add_argument('--cpu', action='store_true', help='CPU smoke check only')
    args = parser.parse_args()
    if args.cpu and not args.smoke:
        parser.error('--cpu requires --smoke')
    manifest = read(ADDON / 'manifest.json')
    for base, entries in [(ADDON, manifest['files']), (ROOT, manifest['required_sources'])]:
        for name, digest in entries.items():
            path = (base / name).resolve()
            if not path.is_relative_to(base.resolve()) or sha(path) != digest:
                raise ValueError(f'Integrity failure: {name}')
    run = args.run.resolve()
    settings = read(run / 'settings.json')
    config = settings['config']
    if config['method'] != 'pure':
        raise ValueError('This evaluator is for Pure RL only')
    checkpoint = run / f'checkpoint-{args.epoch}.pt'
    split = read(ADDON / 'split.json')
    entries = split['test']
    names = [e['name'] for e in entries]
    if len(names) != 80 or len(set(names)) != 80 or set(names) & set(config['trainer']['train_instances']):
        raise ValueError('Invalid frozen train/test membership')
    for entry in entries:
        if sha(ADDON / 'data' / (entry['name'] + '.txt')) != entry['sha256']:
            raise ValueError('Scenario hash mismatch')
    if args.smoke:
        entries = [next(e for e in entries if e['kind'] == kind) for kind in ('RAIN', 'ACCIDENT')]
    names = [e['name'] for e in entries]
    out = (args.out or run / ('eval_smoke' if args.smoke else f'eval_{args.epoch}_test80')).resolve()
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError(f'Output exists; use --resume or a new --out: {out}')
    protocol = dict(checkpoint_sha256=sha(checkpoint), settings_sha256=sha(run / 'settings.json'),
                    manifest_sha256=sha(ADDON / 'manifest.json'), epoch=args.epoch, names=names,
                    augmentation=8, pomo=100, seed='42 + original test index', llm=False,
                    smoke=args.smoke, cpu=args.cpu)
    if args.resume and read(out / 'protocol.json') != protocol:
        raise ValueError('Resume protocol changed')
    write(out / 'protocol.json', protocol)
    block_network()
    import torch
    import train_vrptw as training
    from vrptw_env import _solution_stats, load_solomon
    training._MLFLOW = False
    torch.set_num_threads(4)
    training._set_seed(42)
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: submit through Slurm with singularity --nv')
    tp = copy.deepcopy(config['trainer'])
    tp.update(data_dir=str(ROOT / 'data'), result_dir=str(out / 'runtime'),
              test_instances=[], use_cuda=not args.cpu,
              cuda_device_num=0, model_load={'enable': False}, use_augmentation=True,
              n_mc_samples=1, pomo_auto=False, eval_top_k=None)
    env = dict(config['environment'], pomo_size=100)
    trainer = training.VRPTWTrainer(env, dict(config['model'], eval_type='argmax'), config['optimizer'], tp)
    # Preserve legacy initialization; held-out instances are loaded separately, never trained on.
    trainer.test_pool = [load_solomon(str(ADDON / 'data' / (name + '.txt'))) for name in names]
    # Only the user's own trusted training checkpoint is accepted.
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['epoch'] != args.epoch or state['server_contract'] != settings:
        raise ValueError('Checkpoint does not match the saved training settings')
    trainer.model.load_state_dict(state['model_state_dict'], strict=True)
    trainer.model.eval()
    rows = read(out / 'results.json') if args.resume and (out / 'results.json').exists() else []
    if [r['name'] for r in rows] != names[:len(rows)]:
        raise ValueError('Result order mismatch')
    original_names = [e['name'] for e in split['test']]
    for index, inst in enumerate(trainer.test_pool):
        if index < len(rows):
            continue
        training._set_seed(42 + original_names.index(inst['name']))
        with torch.no_grad():
            routes, reward = trainer._best_solution(inst)
        stats = _solution_stats(inst, routes)
        customers = [int(n) for route in routes for n in route if n]
        if sorted(customers) != list(range(1, 101)) or stats['n_served'] != 100:
            raise ValueError('Incomplete or duplicated customer service')
        row = dict(name=inst['name'], kind=entries[index]['kind'],
                   vehicles=len(stats['per_vehicle']), travel_time=stats['total_travel_time'],
                   late_count=stats['late_count'], total_late=stats['total_late'],
                   served=stats['n_served'], reward=float(reward), routes=routes)
        row['cost'] = 0.04 * row['vehicles'] + row['travel_time'] / 5760 + 0.0075 * row['total_late']
        rows.append(row)
        write(out / 'results.json', rows)
        write(out / 'status.json', dict(status='evaluating', completed=len(rows), total=len(names)))
        print(f'EVAL {len(rows)}/{len(names)} {row["name"]} K={row["vehicles"]} '
              f'T={row["travel_time"]:.2f} Lt={row["total_late"]:.2f} J={row["cost"]:.6f}', flush=True)
    summary = {}
    for kind in ('RAIN', 'ACCIDENT', 'ALL'):
        selected = [r for r in rows if kind == 'ALL' or r['kind'] == kind]
        summary[kind] = dict(count=len(selected), zero_late=sum(r['late_count'] == 0 for r in selected),
            **{key: sum(r[key] for r in selected) / len(selected)
               for key in ('vehicles', 'travel_time', 'late_count', 'total_late', 'served', 'cost')})
    write(out / 'summary.json', dict(protocol=protocol, summary=summary))
    write(out / 'status.json', dict(status='complete', completed=len(rows), total=len(names)))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
