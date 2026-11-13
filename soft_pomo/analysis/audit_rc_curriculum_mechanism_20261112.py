"""CPU-only Train20 audit of completed curriculum comparisons."""
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'result/rc_curriculum_mechanism_audit_20261112'
KEYS = ('reward', 'vehicles', 'travel_time', 'total_late', 'unserved')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def compare(control, treatment):
    c = {r['name']: r for r in read(control)}
    t = {r['name']: r for r in read(treatment)}
    assert c.keys() == t.keys() and len(c) == 20
    rows = []
    for name in c:
        delta = {k: t[name][k] - c[name][k] for k in KEYS}
        costs = dict(vehicle=100 * delta['vehicles'], travel=delta['travel_time'],
                     late=5 * delta['total_late'])
        costs['total'] = sum(costs.values())
        rows.append(dict(name=name, delta=delta, cost_delta=costs))
    groups = {}
    for group in ('all', 'RAIN', 'ACC'):
        selected = [r for r in rows if group == 'all' or f'_{group}_' in r['name']]
        groups[group] = dict(n=len(selected),
            delta={k: mean(r['delta'][k] for r in selected) for k in KEYS},
            cost_delta={k: mean(r['cost_delta'][k] for r in selected) for k in costs},
            wins=sum(r['delta']['reward'] > 1e-6 for r in selected),
            losses=sum(r['delta']['reward'] < -1e-6 for r in selected))
    positive = sorted((r for r in rows if r['delta']['total_late'] > 0),
                      key=lambda r: r['delta']['total_late'], reverse=True)
    total_positive = sum(r['delta']['total_late'] for r in positive)
    return dict(control=str(control.relative_to(ROOT)), treatment=str(treatment.relative_to(ROOT)),
                means={arm: {k: mean(r[k] for r in values.values()) for k in KEYS}
                       for arm, values in (('control', c), ('treatment', t))},
                groups=groups, late_increase_cases=len(positive),
                top3_positive_late_share=sum(r['delta']['total_late'] for r in positive[:3]) / total_positive if total_positive else 0,
                paired=rows)


def main():
    results = {}
    for label, folder in (
        ('seed44_K17', 'rc_pure_k17_curriculum_20261112'),
        ('seed42_K17', 'rc_pure42_k17_curriculum_20261112'),
        ('seed45_K15', 'rc_pure45_k15_curriculum_20261112')):
        base = ROOT / 'result' / folder
        assert read(base / 'status.json')['stage'] == 'complete'
        results[label] = compare(base / 'control/free_epoch040.json', base / 'curriculum/free_epoch040.json')
    base = ROOT / 'result/rc_pure_k18_curriculum_20261112'
    results['seed44_K18'] = compare(ROOT / 'result/rc_pure_k17_curriculum_20261112/control/free_epoch040.json',
                                   base / 'curriculum/free_epoch040.json')
    results['seed44_K18']['checkpoint40_exists'] = (base / 'curriculum/checkpoint-40.pt').is_file()
    results['seed44_K18']['summary_last_free_epoch'] = max(
        r['epoch'] for r in read(base / 'summary.json')['evaluations'] if r['mode'] == 'free')
    OUT.mkdir(exist_ok=True)
    (OUT / 'report.json').write_text(json.dumps(dict(evaluation='Train20 only; no Test accessed', comparisons=results), indent=2), encoding='utf-8')
    for label, result in results.items():
        print(label, json.dumps(dict(means=result['means'], groups=result['groups'],
              late_increase_cases=result['late_increase_cases'], top3_positive_late_share=result['top3_positive_late_share'])))


if __name__ == '__main__':
    main()
