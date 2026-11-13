"""Replay saved A/C/BKS routes to decompose travel and lateness without training."""
import hashlib
import json
from pathlib import Path
import statistics as st
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'result/bks_normalized_late10_train20_20261013/rc101'
sys.path.insert(0, str(FROZEN))
from vrptw_env import load_solomon, _solution_stats


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def main():
    sources = {v: ROOT / f'result/rc101_pure_train20_b128_s42_config{v}_travel_time_1000_20261010_v2/eval_1000_test80/results.json' for v in ('A', 'C')}
    sources['BKS10'] = FROZEN / 'eval_1000_test80_gpu/results.json'
    sources['BKS1'] = ROOT / 'server_export/rc101_bks_verified_20261014_111512/rc101_bks_train20_80_e1000_20261013/train20_pure/eval_1000_test80_gpu/results.json'
    records = {v: {r['name']: r for r in read(p)} for v, p in sources.items()}
    split = read(FROZEN / 'split.json')
    names = {r['name'] for r in split['test']}
    assert all(set(rows) == names for rows in records.values())
    detailed = {v: {} for v in records}
    for row in split['test']:
        name = row['name']
        path = FROZEN / 'data' / (name + '.txt')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
        inst = load_solomon(str(path))
        for version, rows in records.items():
            saved = rows[name]
            stats = _solution_stats(inst, saved['routes'])
            for sk, rk in [('total_travel_time','travel_time'), ('total_late','total_late'), ('late_count','late_count')]:
                assert abs(stats[sk] - saved[rk]) < 1e-4, (version, name, sk)
            vehicles = stats['per_vehicle']
            m = dict(vehicles=len(vehicles), travel=saved['travel_time'], lateness=saved['total_late'],
                depot_travel=0., internal_travel=0., event_extra=stats['total_travel_time']-float(stats['total_dist']),
                customer_late=0., depot_late=0., customer_late_count=0, depot_late_count=0,
                latter_half_late=0., latter_half_late_count=0, first_stop_late_count=0,
                waiting=0., max_customer_late=0., max_customers=max(len(v['stops']) for v in vehicles),
                mean_customers=100/len(vehicles), mean_load=st.mean(float(v['load']/stats['cap']) for v in vehicles))
            for v in vehicles:
                stops = v['stop_details']
                ret = v['arr_depot'] - stops[-1]['depart']
                m['depot_travel'] += stops[0]['travel_time'] + ret
                m['internal_travel'] += sum(s['travel_time'] for s in stops[1:])
                m['depot_late'] += v['ret_late']
                m['depot_late_count'] += int(v['ret_late'] > 0)
                for i, s in enumerate(stops):
                    m['customer_late'] += s['late']
                    m['customer_late_count'] += int(s['late'] > 0)
                    m['waiting'] += s['wait']
                    m['max_customer_late'] = max(m['max_customer_late'], s['late'])
                    if (i+1)/len(stops) > .5:
                        m['latter_half_late'] += s['late']
                        m['latter_half_late_count'] += int(s['late'] > 0)
                    if i == 0:
                        m['first_stop_late_count'] += int(s['late'] > 0)
            assert abs(m['depot_travel']+m['internal_travel']-m['travel']) < 1e-5
            detailed[version][name] = m
    summary = {v: {key: st.mean(r[key] for r in rows.values()) for key in next(iter(rows.values()))} for v, rows in detailed.items()}
    for v, m in summary.items():
        m['latter_half_share_customer_late'] = m['latter_half_late']/m['customer_late'] if m['customer_late'] else None
        m['scores_minutes'] = {label: a*m['vehicles'] + m['travel'] + b*m['lateness']
            for label, a, b in [('A',172.8,7.2), ('C',76.8,4.8), ('BKS10',1696.95/14,10), ('BKS1',1696.95/14,1)]}
    paired = {}
    for a,b in [('A','C'),('C','BKS10')]:
        counts = {k: 0 for k in ['fewer_vehicles','equal_vehicles','more_vehicles','lower_travel','lower_depot_travel']}
        reassigned = []
        for name in names:
            x,y = detailed[a][name], detailed[b][name]
            counts['fewer_vehicles' if y['vehicles']<x['vehicles'] else 'equal_vehicles' if y['vehicles']==x['vehicles'] else 'more_vehicles'] += 1
            counts['lower_travel'] += int(y['travel']<x['travel'])
            counts['lower_depot_travel'] += int(y['depot_travel']<x['depot_travel'])
            ra = [set(r)-{0} for r in records[a][name]['routes']]
            rb = [set(r)-{0} for r in records[b][name]['routes']]
            w = np.array([[len(x&y) for y in rb] for x in ra])
            i,j = linear_sum_assignment(-w)
            reassigned.append(100-int(w[i,j].sum()))
        paired[a+'->'+b] = dict(counts, mean_min_reassigned=st.mean(reassigned))
    out = dict(summary=summary, paired=paired, scenario_details=detailed,
        note='Latter half means customer rank / route customer count > 0.5; depot excluded. Observational replay, not causal counterfactual.',
        sources={v:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for v,p in sources.items()})
    (ROOT/'analysis/rc101_reward_route_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(dict(summary=summary,paired=paired),indent=2))


if __name__ == '__main__':
    main()
