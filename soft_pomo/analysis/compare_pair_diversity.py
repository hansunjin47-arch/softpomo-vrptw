"""Descriptive diversity audit of saved training samples, not frozen-policy tests."""
import json
from pathlib import Path
import statistics as st

RUNS = {
    'pair': Path('result/pair300_pure_optimized_20261025'),
    'stop': Path('server_export/be_completed_received_20261022_091410/rc101_v9_bias_entropy_vehicle300_late10_20261021/runs/vehicle300'),
}


def main():
    report = {}
    for mode, root in RUNS.items():
        records = []
        for update in range(6000, 6461, 20):
            d = json.loads((root/f'late10/early_history/update_{update:06d}.json').read_text())
            metric = d['rollout_metrics']
            orders, sets, routes = [], [], []
            for rollout in range(4):
                first = []
                vehicles = {}
                for step in d['decision_history']:
                    row = step[rollout]
                    if mode == 'pair':
                        time, vehicle, dest, new, live = row
                    else:
                        time, vehicle, action, live = row
                        if action == 201:
                            continue
                        new = action < 100
                        dest = action + 1 if new else action - 100
                    if not live:
                        continue
                    if new:
                        first.append(int(dest))
                    vehicles.setdefault(int(vehicle), []).append(int(dest))
                orders.append(tuple(first))
                sets.append(tuple(sorted(first)))
                routes.append(tuple(sorted(tuple(v) for v in vehicles.values())))
            histogram = d['vehicle_histogram']
            records.append(dict(
                update=update, instance=d['name'],
                reward_std=st.pstdev(metric['reward']),
                reward_unique=len(set(round(x, 4) for x in metric['reward'])),
                vehicle19_fraction=histogram[19]/sum(histogram),
                vehicle_below19_fraction=sum(histogram[:19])/sum(histogram),
                first_orders=len(set(orders)), first_sets=len(set(sets)),
                canonical_routes=len(set(routes))))
        keys = [k for k in records[0] if k not in ('update', 'instance')]
        report[mode] = dict(samples=len(records), means={k:st.mean(r[k] for r in records) for k in keys},
                            records=records)
        print(mode, json.dumps(report[mode]['means']))
    Path('analysis/pair_diversity_saved_20261025.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
