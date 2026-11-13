"""CPU-only dispatch diagnostics from an immutable training checkpoint snapshot."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import torch
from experiments.concurrent_fleet_v3.run import load, write
from experiments.concurrent_fleet_v5.env import TensorFleetEnv
from experiments.concurrent_fleet_v7.policy import FleetPolicy


@torch.no_grad()
def evaluate(model, inst, public, params, count, greedy):
    env = TensorFleetEnv(inst, count, params, 'cpu')
    starts, encoded = model.encode_events(inst, public)
    dispatch = []
    first = None
    for step in range(600):
        if bool(env.done.all()):
            break
        frame = env.frame()
        p = model.probabilities(frame, starts, encoded, env)
        action = p.argmax(1) if greedy else p.multinomial(1).squeeze(1)
        if first is None:
            first = action.tolist()
        for i in torch.where(frame['mask'][:, -1])[0].tolist():
            dispatch.append(dict(rollout=i, vehicles=int(env.k[i]),
                                 select_probability=float(p[i, -1]),
                                 new_probability=float(p[i, :env.n].sum()),
                                 entropy=float(-(p[i]*p[i].clamp(min=1e-30).log()).sum()),
                                 selected=bool(action[i] == 2*env.n+1),
                                 forced=not bool(frame['mask'][i, :env.n].any())))
        env.step(action, validate=True)
    else:
        raise RuntimeError('Diagnostic episode did not finish')
    return dict(metrics={k:float(v.double().mean()) for k,v in env.metrics().items()},
                vehicle_histogram=torch.bincount(env.k, minlength=env.v+1).tolist(),
                distinct_first_customers=len(set(first)), dispatch=dispatch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--weight', default='5')
    parser.add_argument('--samples', type=int, default=32)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Use a new diagnostic output directory')
    args.out.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.manual_seed(20261018)
    snapshot = args.out/'checkpoint.pt'
    shutil.copy2(args.run/f'late{args.weight}/checkpoint-last.pt', snapshot)
    saved = torch.load(snapshot, map_location='cpu', weights_only=False)
    protocol = json.loads((args.run/'protocol.json').read_text())
    config = protocol['configs'][args.weight]
    model = FleetPolicy(**config['model']).eval()
    model.load_state_dict(saved['model_state_dict'], strict=True)
    split = json.loads((args.run/'split.json').read_text())
    result = dict(role='TRAIN diagnostics, not test evaluation', device='cpu', precision='FP32',
                  note='CPU FP32 diagnostic trajectories can differ from CUDA AMP training probes.',
                  checkpoint_state=saved['state'], samples_per_instance=args.samples,
                  select_weight=model.select_weight.detach().tolist(),
                  select_bias=float(model.select_bias.detach()), results=[])
    for row in split['train']:
        inst, public = load(row)
        entry = dict(name=row['name'],
                     greedy=evaluate(model, inst, public, config['environment'], 1, True),
                     sampled=evaluate(model, inst, public, config['environment'], args.samples, False))
        result['results'].append(entry)
        write(args.out/'diagnostics.json', result)
        print(row['name'], 'greedy K', entry['greedy']['metrics']['vehicles'],
              'sample K', entry['sampled']['metrics']['vehicles'], flush=True)
    groups = defaultdict(list)
    for entry in result['results']:
        for d in entry['sampled']['dispatch']:
            groups[d['vehicles']].append(d)
    result['dispatch_by_vehicle_count'] = [dict(vehicles=k, decisions=len(rows),
        mean_select_probability=sum(d['select_probability'] for d in rows)/len(rows),
        observed_select_fraction=sum(d['selected'] for d in rows)/len(rows),
        forced_decisions=sum(d['forced'] for d in rows)) for k, rows in sorted(groups.items())]
    write(args.out/'diagnostics.json', result)
    print('DONE', args.out, 'checkpoint update', saved['state']['update'], flush=True)


if __name__ == '__main__':
    main()
