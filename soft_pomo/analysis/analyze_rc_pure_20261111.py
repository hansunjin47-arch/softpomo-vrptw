"""Read-only paired Test80 cost decomposition for masked RC Pure RL."""
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    history = read(ROOT/'analysis/rc101_pilot_history_20_80_20260923_104517/results.json')
    baseline = {r['name']: r for r in history}
    report = []
    for weight in (100, 200):
        all_rows = []
        for seed in range(42, 47):
            paths = list((ROOT/'result').glob(f'rc_pair_returnmask_v{weight}_l5_s{seed}_*/late5/results.json'))
            assert len(paths) == 1
            path = paths[0]
            cfg = read(path.parents[1]/'protocol.json')['configs']['5']
            assert cfg['trainer']['seed'] == seed
            assert cfg['environment']['protect_last_service_vehicle']
            assert cfg['environment']['late_penalty'] == 5
            assert abs(cfg['environment']['vehicle_penalty'] * 1696.95 / 100 - weight) < 1e-6
            rows = read(path)
            assert len(rows) == 80 and {r['name'] for r in rows} == set(baseline)
            paired = []
            for r in rows:
                b = baseline[r['name']]
                assert r['unserved'] == 0
                cost = weight*r['vehicles'] + r['travel_time'] + 5*r['total_late']
                assert abs(-cost/1696.95 - r['reward']) < 1e-5
                paired.append(dict(name=r['name'], seed=seed, **{k:r[k] for k in
                    ('reward','vehicles','travel_time','total_late')},
                    delta_vehicle_cost=weight*(r['vehicles']-b['vehicles']),
                    delta_travel_cost=r['travel_time']-b['travel_time'],
                    delta_late_cost=5*(r['total_late']-b['total_late']),
                    delta_cost=cost-(weight*b['vehicles']+b['travel_time']+5*b['total_late'])))
            all_rows.extend(paired)
        subsets = {'all': all_rows}
        subsets.update({f'seed{s}':[r for r in all_rows if r['seed']==s] for s in range(42,47)})
        subsets.update({kind:[r for r in all_rows if f'_{kind}_' in r['name']] for kind in ('RAIN','ACC')})
        aggregate = {}
        for label, rows in subsets.items():
            aggregate[label] = dict(count=len(rows), **{k:mean(r[k] for r in rows) for k in
                ('reward','vehicles','travel_time','total_late','delta_vehicle_cost',
                 'delta_travel_cost','delta_late_cost','delta_cost')},
                wins=sum(r['delta_cost'] < -1e-5 for r in rows))
        item = dict(weight=weight, late=5, mask=True, summaries=aggregate,
            worst_cases=sorted(all_rows,key=lambda r:r['delta_cost'],reverse=True)[:10])
        report.append(item)
        print(json.dumps(dict(weight=weight,summaries=aggregate)),flush=True)
    out = ROOT/'result/rc_pure_diagnosis_20261111'
    out.mkdir(exist_ok=True)
    (out/'test_comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
