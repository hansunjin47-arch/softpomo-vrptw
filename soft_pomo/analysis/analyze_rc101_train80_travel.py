"""Replay matched test80 routes and decompose travel and Config E costs."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vrptw_env import load_solomon, _solution_stats


def read(path):
    return json.loads((ROOT / path).read_text())


def main():
    history = 'analysis/rc101_history_train80_20260929'
    models = dict(
        history=read(history + '/results.json'),
        pure=read('result/rc101_pure_train80_fresh1000_20260929/eval_1000_saved_routes/results.json'),
        kim=read('result/rc101_kim_train80_fresh1000_20260929/eval_1000_test80/results.json'))
    common = read(history + '/routes.json')
    cases = {}
    for model, rows in models.items():
        cases[model] = {}
        for row in rows:
            inst = load_solomon(str(ROOT / 'data/rc101_train80_test80_20260929' / (row['name'] + '.txt')))
            routes = common if model == 'history' else row['routes']
            s = _solution_stats(inst, routes)
            assert s['n_served'] == 100
            assert abs(s['total_travel_time'] - row['travel_time']) < 1e-4
            assert abs(s['total_late'] - row['total_late']) < 1e-4
            depot = sum(float(s['tt'][r[0], r[1]]) + float(s['tt'][r[-2], r[-1]]) for r in routes)
            base = sum(float(s['tt'][a, b]) for r in routes for a, b in zip(r, r[1:]))
            k, t, late = len(routes), s['total_travel_time'], s['total_late']
            cases[model][row['name']] = dict(kind=row['kind'], vehicles=k,
                travel=t, base=base, event_extra=t-base, depot_legs=depot,
                customer_legs=base-depot, late=late, late_count=s['late_count'],
                depot_late=sum(v['ret_late'] for v in s['per_vehicle']),
                cost=.04*k+t/(24*s['T'])+9*late/(5*s['T']))
    assert set(cases['history']) == set(cases['pure']) == set(cases['kim'])
    metrics = [k for k in next(iter(cases['kim'].values())) if k != 'kind']
    summary = {}
    for model, rows in cases.items():
        summary[model] = {}
        for group in ['ALL', 'RAIN', 'ACCIDENT']:
            rs = [r for r in rows.values() if group == 'ALL' or r['kind'] == group]
            summary[model][group] = {k: float(np.mean([r[k] for r in rs])) for k in metrics}
    paired = {}
    for reference in ['history', 'pure']:
        diffs = {name: {k: row[k]-cases[reference][name][k] for k in metrics}
                 for name, row in cases['kim'].items()}
        paired[reference] = dict(changes={k: dict(
            better=sum(r[k] < -1e-6 for r in diffs.values()),
            same=sum(abs(r[k]) <= 1e-6 for r in diffs.values()),
            worse=sum(r[k] > 1e-6 for r in diffs.values())) for k in metrics},
            both_zero_late=[n for n in diffs if cases['kim'][n]['late'] < 1e-6 and cases[reference][n]['late'] < 1e-6],
            largest_travel_gaps=sorted([dict(name=n, **r) for n, r in diffs.items()], key=lambda r:r['travel'], reverse=True)[:5])
        zero = paired[reference]['both_zero_late']
        paired[reference]['both_zero_mean_delta'] = {k: float(np.mean([diffs[n][k] for n in zero])) for k in metrics} if zero else {}
    out = ROOT / 'analysis/rc101_train80_travel_audit'
    out.mkdir(exist_ok=True)
    (out/'details.json').write_text(json.dumps(cases, indent=2), encoding='utf-8')
    report = dict(summary=summary, paired=paired)
    (out/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
