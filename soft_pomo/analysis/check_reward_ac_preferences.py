"""Rescore existing test80 routes; no training, decoding, or API calls."""
import json
import math
from pathlib import Path
import statistics as st
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vrptw_env import load_solomon


def main():
    data = ROOT / 'data/rc101_train80_test80_20260929'
    split = json.loads((data / 'split.json').read_bytes())
    inst = load_solomon(str(data / (split['test'][0]['name'] + '.txt')))
    tt = inst['tt'].numpy() * inst['T']
    history = json.loads((ROOT / 'analysis/rc101_train80_travel_audit/details.json').read_bytes())['history']
    methods = {'History': {name: dict(K=r['vehicles'], D=r['base'], T=r['travel'], L=r['late'])
                           for name, r in history.items()}}
    paths = {f'Pure128_{ep}': f'server_export/server_pure128_results/eval_{ep}_test80/results.json'
             for ep in (1000, 2000, 3000)}
    paths.update(Kim64='result/rc101_kim_train80_fresh1000_20260929/eval_1000_test80/results.json',
                 Phase2_64='result/rc101_pure500_train80_phase2_20260930/eval_1000_test80/results.json')
    for label, path in paths.items():
        rows = json.loads((ROOT / path).read_bytes())
        assert len(rows) == 80 and all(r['served'] == 100 for r in rows)
        methods[label] = {}
        for r in rows:
            assert sorted(n for route in r['routes'] for n in route if n) == list(range(1, 101))
            dist = sum(float(tt[a, b]) for route in r['routes'] for a, b in zip(route, route[1:]))
            methods[label][r['name']] = dict(K=r['vehicles'], D=dist, T=r['travel_time'], L=r['total_late'])
    names = list(methods['History'])
    report = dict(scope='Fixed existing solutions, not predictions of retraining. Guided runs batch64; Pure batch128.',
                  means={}, paired_vs_history={}, rankings={}, flips={})
    for label, rows in methods.items():
        assert set(rows) == set(names)
        for r in rows.values():
            r.update(E=.04*r['K']+r['T']/5760+.0075*r['L'],
                     A=.06*r['K']+r['D']/2880+.0025*r['L'],
                     C=.04*r['K']+r['D']/1920+.0025*r['L'])
        report['means'][label] = {k: st.mean(r[k] for r in rows.values()) for k in next(iter(rows.values()))}
        report['means'][label]['A_terms'] = [st.mean(r['K']*.06 for r in rows.values()),
            st.mean(r['D']/2880 for r in rows.values()), st.mean(r['L']*.0025 for r in rows.values())]
        if label != 'History':
            report['paired_vs_history'][label] = {c: sum(rows[n][c] < methods['History'][n][c] for n in names)
                                                 for c in ('E', 'A', 'C')}
    for c in ('E', 'A', 'C'):
        report['rankings'][c] = sorted(methods, key=lambda m: report['means'][m][c])
    for a in methods:
        for b in methods:
            if a >= b:
                continue
            report['flips'][a+' vs '+b] = sum((methods[a][n]['A']-methods[b][n]['A'])*
                                            (methods[a][n]['C']-methods[b][n]['C']) < 0 for n in names)
    report['best_among_saved'] = {c: {m: sum(min(methods, key=lambda k: methods[k][n][c]) == m
                                                for n in names) for m in methods} for c in ('E','A','C')}
    target = ROOT / 'analysis/reward_ac_preferences.json'
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
