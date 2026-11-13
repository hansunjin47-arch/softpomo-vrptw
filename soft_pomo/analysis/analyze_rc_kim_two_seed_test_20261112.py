"""Report-only paired Test80 analysis; not used for curriculum or guide tuning."""
import hashlib
from collections import Counter
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'result/rc_kim_two_seed_test_audit_20261112'
KIM = ROOT / 'result/rc101_pair100_l5_mask_kim_test_20261112'
KEYS = ('reward', 'vehicles', 'travel_time', 'total_late', 'unserved')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(rows):
    return {k: mean(r[k] for r in rows) for k in KEYS}


def pairing(treatment, control):
    old = {r['name']: r for r in control}
    assert set(old) == {r['name'] for r in treatment}
    rows = []
    for r in treatment:
        delta = {k: r[k] - old[r['name']][k] for k in KEYS}
        costs = dict(vehicle=100 * delta['vehicles'], travel=delta['travel_time'], late=5 * delta['total_late'])
        costs['total'] = sum(costs.values())
        assert abs(-costs['total'] / 1696.95 - delta['reward']) < 1e-5
        rows.append(dict(name=r['name'], delta=delta, cost_delta=costs))
    groups = {}
    for kind in ('all', 'RAIN', 'ACC'):
        selected = [r for r in rows if kind == 'all' or f'_{kind}_' in r['name']]
        groups[kind] = dict(n=len(selected),
            delta={k: mean(r['delta'][k] for r in selected) for k in KEYS},
            cost_delta={k: mean(r['cost_delta'][k] for r in selected) for k in costs},
            wins=sum(r['delta']['reward'] > 1e-6 for r in selected),
            losses=sum(r['delta']['reward'] < -1e-6 for r in selected),
            ties=sum(abs(r['delta']['reward']) <= 1e-6 for r in selected),
            late_increase_cases=sum(r['delta']['total_late'] > 1e-6 for r in selected))
    return dict(groups=groups, paired=rows,
                worst=sorted(rows, key=lambda r: r['cost_delta']['total'], reverse=True)[:5],
                best=sorted(rows, key=lambda r: r['cost_delta']['total'])[:5])


def main():
    assert read(KIM / 'status.json')['stage'] == 'complete'
    lkh_path = ROOT / 'analysis/rc101_pilot_history_20_80_20260923_104517/results.json'
    lkh = read(lkh_path)
    for r in lkh:
        assert r['served'] == 100
        r['unserved'] = 0
        r['reward'] = -(100 * r['vehicles'] + r['travel_time'] + 5 * r['total_late']) / 1696.95
    hashes = {str(lkh_path.relative_to(ROOT)): sha(lkh_path)}
    results = {}
    pooled = {'Kim': [], 'Pure RL': []}
    for seed, date in ((42, '20261031'), (43, '20261102')):
        pure_dir = ROOT / f'result/rc_pair_returnmask_v100_l5_s{seed}_{date}'
        pure_path = pure_dir / 'late5/results.json'
        kim_path = KIM / f'seed{seed}/results.json'
        assert read(KIM / f'seed{seed}/complete.json')['test_count'] == 80
        cfg = read(pure_dir / 'protocol.json')['configs']['5']
        assert cfg['trainer']['seed'] == seed
        assert cfg['environment']['late_penalty'] == 5 and cfg['environment']['protect_last_service_vehicle']
        assert abs(cfg['environment']['vehicle_penalty'] - 10000 / 1696.95) < 1e-8
        pure, kim = read(pure_path), read(kim_path)
        assert len(pure) == len(kim) == len(lkh) == 80
        for path, rows in ((pure_path, pure), (kim_path, kim)):
            hashes[str(path.relative_to(ROOT))] = sha(path)
            for r in rows:
                assert r['unserved'] == 0
                assert abs(r['reward'] + (100 * r['vehicles'] + r['travel_time'] + 5 * r['total_late']) / 1696.95) < 1e-5
        results[str(seed)] = dict(means={'Kim': summary(kim), 'Pure RL': summary(pure), 'LKH-3': summary(lkh)},
            vehicle_histograms={label: dict(sorted(Counter(r['vehicles'] for r in rows).items()))
                                for label, rows in (('Kim', kim), ('Pure RL', pure), ('LKH-3', lkh))},
            kim_minus_pure=pairing(kim, pure), kim_minus_lkh=pairing(kim, lkh))
        pooled['Kim'].extend(kim)
        pooled['Pure RL'].extend(pure)
    report = dict(setting='RC101 concurrent vehicleweight100 lambda5 maskON; original Test80; two seeds42/43 only',
        scope='Reporting/diagnosis only; not used to select prompts, curriculum targets, coefficients or schedules',
        seed_results=results, means={k: summary(v) for k, v in pooled.items()}, lkh=summary(lkh),
        input_hashes=hashes, limits='Two seeds are not five-seed confirmation; 160 rows are not160 independent scenarios.')
    OUT.mkdir(exist_ok=True)
    (OUT / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for seed, result in results.items():
        print(seed, json.dumps(dict(means=result['means'], delta=result['kim_minus_pure']['groups'])))
    print('POOLED', report['means'], 'LKH', report['lkh'])


if __name__ == '__main__':
    main()
