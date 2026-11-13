"""Offline paired score and route audit for the completed A evaluation."""
import json
import statistics as st
from collections import Counter
from pathlib import Path
import numpy as np
import diagnose_tt_ac_routes as audit

ROOT = audit.ROOT
OUT = ROOT/'analysis/depot_customer_only_A_20261012.json'


def main():
    versions = {'old': 'ontology_routes_only_ac_travel_time_v1',
                'new': 'ontology_routes_only_customer_only_ac_v1'}
    output, records = {}, {}
    for label, version in versions.items():
        root = ROOT/f'result/rc101_phase2_train20_b128_s42_configA_{version}'
        rows = audit.read(root/'eval_1000_test80_gpu/results.json')
        records[label] = {r['name']: r for r in rows}
        detail, late_nodes = [], Counter()
        for row in rows:
            name = row['name']
            cache = audit.read(root/'eval_1000_test80/cache'/f'{name}_rl_cluster.json')
            instance = audit.load_solomon(str(audit.SOURCE/'data'/f'{name}.txt'))
            stats = audit._solution_stats(instance, row['routes'])
            assert abs(stats['total_travel_time']-row['travel_time']) < .02
            assert abs(stats['total_late']-row['total_late']) < .02
            groups = {int(n):k for k,g in cache.items() if not k.startswith('_') for n in g}
            values, biases, margins, ranges = [], [], [], []
            for k,g in cache.items():
                if k.startswith('_'):
                    continue
                depot = cache['_depot'][k]
                vals = list(g.values())
                ranges.append(max(vals)-min(vals))
                for v in vals:
                    b = -5 if v < depot else 5*v
                    values.append(v)
                    biases.append(b)
                    margins.append(b-5*depot)
            vehicles = stats['per_vehicle']
            stops = [s for v in vehicles for s in v['stop_details']]
            late_nodes.update(s['node'] for s in stops if s['late']>0)
            detail.append(dict(name=name, score=st.mean(values), score_sd=st.stdev(values),
                group_range=st.mean(ranges), negative=sum(b<0 for b in biases),
                depot_score=st.mean(cache['_depot'].values()), customer_bias=st.mean(biases),
                customer_minus_depot=st.mean(margins),
                base=float(stats['total_dist']), event=row['travel_time']-float(stats['total_dist']),
                depot_legs=sum(float(stats['tt'][0,v['stops'][0]])+v['ret_dist'] for v in vehicles),
                load=st.mean(v['load']/stats['cap'] for v in vehicles),
                small_routes=sum(len(v['stops'])<=3 for v in vehicles),
                off_group=sum(groups[n]!=groups[v['stops'][0]] for v in vehicles for n in v['stops']),
                late_depots=sum(v['ret_late']>0 for v in vehicles),
                depot_lateness=sum(v['ret_late'] for v in vehicles)))
        output[label] = {'mean': {k:st.mean(d[k] for d in detail) for k in detail[0] if k!='name'},
            'top_late_customers':late_nodes.most_common(10), 'scenarios':detail}
    names = list(records['old'])
    assert set(names)==set(records['new']) and len(names)==80
    output['paired'] = {k:{'better':sum(records['new'][n][k]<records['old'][n][k]-1e-6 for n in names),
                          'same':sum(abs(records['new'][n][k]-records['old'][n][k])<=1e-6 for n in names),
                          'worse':sum(records['new'][n][k]>records['old'][n][k]+1e-6 for n in names)}
                        for k in ['vehicles','travel_time','late_count','total_late','cost']}
    reassigned, inversions = [], []
    for n in names:
        p,o = audit.relations(records['old'][n]['routes'])
        q,r = audit.relations(records['new'][n]['routes'])
        reassigned.append(len(p^q)/len(p|q))
        inversions.append(sum(((a,b) in o and (b,a) in r) or ((b,a) in o and (a,b) in r) for a,b in p&q))
    output['route_changes'] = dict(pair_reassignment=st.mean(reassigned), shared_pair_order_inversions=st.mean(inversions))
    OUT.write_text(json.dumps(output,indent=2,default=lambda x:x.item()),encoding='utf-8')
    print(json.dumps({k:({a:b for a,b in v.items() if a!='scenarios'}) for k,v in output.items()},indent=2,default=lambda x:x.item()))


if __name__=='__main__':
    main()
