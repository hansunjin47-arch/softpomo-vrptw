"""Verify copied server checkpoint provenance, route replay, and training trends."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from vrptw_env import load_solomon, _solution_stats


def read(path):
    return json.loads(path.read_text())


def main():
    run = ROOT / 'server_export/server_pure128_results'
    data = ROOT / 'data/rc101_train80_test80_20260929'
    log = (ROOT / 'server_export/slurm-sunjinhan_pure128_to4000-198159.log').read_text()
    rewards = {int(e): float(r) for e, r in re.findall(r'^Epoch (\d+)/4000 reward=([-\d.]+)', log, re.M)}
    report, rows_by_epoch, configs, protocols = {}, {}, [], []
    for epoch in (1000, 2000, 3000, 4000):
        path = run / f'checkpoint-{epoch}.pt'
        protocol = read(run / f'eval_{epoch}_test80/protocol.json')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == protocol['checkpoint_sha256']
        state = torch.load(path, map_location='cpu', weights_only=False)
        assert state['epoch'] == epoch == protocol['epoch']
        config = state['server_contract']['config']
        config['trainer'].pop('epochs')
        configs.append(config)
        comparable = {k:v for k,v in protocol.items() if k not in ('epoch','checkpoint_sha256','settings_sha256')}
        protocols.append(comparable)
        rows = read(run / f'eval_{epoch}_test80/results.json')
        rows_by_epoch[epoch] = rows
        lengths, route_loads, returns = [], [], []
        for row in rows:
            stats = _solution_stats(load_solomon(str(data / (row['name'] + '.txt'))), row['routes'])
            assert abs(stats['total_travel_time'] - row['travel_time']) < 1e-7
            assert abs(stats['total_late'] - row['total_late']) < 1e-7
            assert sorted(n for r in row['routes'] for n in r[1:-1]) == list(range(1, 101))
            assert all(v['load'] <= stats['cap'] + 1e-4 for v in stats['per_vehicle'])
            lengths.extend(len(r)-2 for r in row['routes'])
            route_loads.extend(float(v['load']/stats['cap']) for v in stats['per_vehicle'])
            returns.extend(v['arr_depot'] for v in stats['per_vehicle'])
        report[epoch] = dict(checkpoint_and_replay_verified=True,
            mean_reward_last50=statistics.mean(rewards[e] for e in range(epoch-49,epoch+1)) if epoch > 1000 else None,
            mean_customers_per_route=statistics.mean(lengths), route_size_counts=dict(Counter(lengths)),
            mean_capacity_utilization=statistics.mean(route_loads), mean_return_time=statistics.mean(returns),
            vehicle_range=[min(r['vehicles'] for r in rows),max(r['vehicles'] for r in rows)])
        del state
    assert all(config == configs[0] for config in configs)
    # An intentional runner patch changed the bundle manifest after epoch 1000.
    assert all({k:v for k,v in p.items() if k != 'manifest_sha256'} ==
               {k:v for k,v in protocols[0].items() if k != 'manifest_sha256'} for p in protocols)
    comparisons = {}
    for a,b in ((2000,3000),(3000,4000)):
        pairs = list(zip(rows_by_epoch[a], rows_by_epoch[b]))
        assert all(x['name'] == y['name'] for x,y in pairs)
        comparisons[f'{a}_to_{b}'] = dict(
            more_vehicles=sum(y['vehicles'] > x['vehicles'] for x,y in pairs),
            fewer_vehicles=sum(y['vehicles'] < x['vehicles'] for x,y in pairs),
            higher_cost=sum(y['cost'] > x['cost'] for x,y in pairs))
    windows = {str(end):statistics.mean(rewards[e] for e in range(end-99,end+1)) for end in range(1100,4001,100)}
    result = dict(checkpoints=report, paired=comparisons, reward_100epoch_windows=windows,
                  settings_equal_except_epoch_target=True)
    out = ROOT / 'analysis/server_pure128_audit.json'
    out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
