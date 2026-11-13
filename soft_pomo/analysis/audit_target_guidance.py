"""Reference applicability at actual visits, plus recorded direct action flips."""
from collections import Counter
import json
import statistics as st
import argparse
from analyze_ontology_v5_results import ROOT, FROZEN, read, replay, load_solomon

SOURCE = ROOT / 'result/rc101_ontology_v5_train20_b128_seed43/eval_1000_test80'
TARGETS = (68, 91, 80, 70, 100)
RUNS = {10: 'rc101_ontology_v5_b128_s44_scale10_diagnostic',
        5: 'rc101_ontology_v5_b128_s44_scale5'}


def main(all_customers=False):
    targets = tuple(range(1, 101)) if all_customers else TARGETS
    rows, summaries = {}, {}
    for scale, folder in RUNS.items():
        rows[str(scale)] = []
        records = read(ROOT / 'result' / folder / 'eval_1000_test80/results.json')
        for record in records:
            name = record['name']
            cache = read(SOURCE / 'cache' / (name + '_rl_cluster.json'))
            member = {int(n): g for g, d in cache.items() if not g.startswith('_') for n in d}
            score = {int(n): s for g, d in cache.items() if not g.startswith('_') for n, s in d.items()}
            refs = read(SOURCE / 'requests' / name / 'reference.json')['clusters']
            obs = read(FROZEN / 'eval_addon/data' / (name + '.observations.json'))
            inst = load_solomon(str(FROZEN / 'eval_addon/data' / (name + '.txt')))
            _, stats = replay(inst, record)
            _, ref_stats = replay(inst, dict(name=name, kind=record['kind'],
                routes=[[0]+r+[0] for r in refs]))
            ref_visits = {s['node']: s for v in ref_stats['per_vehicle'] for s in v['stop_details']}
            changes = read(ROOT / 'result/rc101_ontology_bias_direction_audit' /
                           f'scale{scale}' / (name + '.json'))['changes']
            for vehicle in stats['per_vehicle']:
                route = vehicle['stops']
                active = member[route[0]]
                previous_departure = 0
                for pos, stop in enumerate(vehicle['stop_details']):
                    node = stop['node']
                    if node in targets:
                        reference = refs[int(member[node])]
                        prefix_match = route[:pos+1] == reference[:reference.index(node)+1]
                        own = member[node] == active
                        disclosed = obs['event_type'] == 'RAIN' or previous_departure >= obs['start_min']
                        depot = cache['_depot'][active]
                        if pos == 0:
                            status, bias = 'first_customer', 0
                        elif not disclosed:
                            status, bias = 'before_disclosure', 0
                        elif not own:
                            status, bias = 'foreign_group', 0
                        elif score[node] < depot:
                            status, bias = 'negative', -5
                        else:
                            status, bias = 'own_group', 5 * score[node]
                        rows[str(scale)].append(dict(name=name,node=node,
                            reference_group=member[node], reference_route=refs[int(member[node])],
                            prefix_match=prefix_match,
                            reference_start_match=route[0]==refs[int(active)][0],
                            raw_score=score[node], own_depot_score=cache['_depot'][member[node]],
                            own_group=own, disclosed=disclosed,status=status,bias=bias,
                            active_group=active, active_depot_score=depot,
                            actual_route=route, position=pos, first=route[0],
                            selection_time=previous_departure,arrival=stop['arrival'],
                            deadline=stop['tw_close'],late=stop['late'],
                            reference_late=ref_visits[node]['late'],
                            route_return_late=vehicle['ret_late'],
                            direct_changes=[c for c in changes if c['old']==node or c['new']==node]))
                    previous_departure = stop['depart']
        summaries[str(scale)] = {}
        for node in targets:
            rr = [r for r in rows[str(scale)] if r['node']==node]
            assert len(rr)==80
            summaries[str(scale)][str(node)] = dict(
                mean_score=st.mean(r['raw_score'] for r in rr),
                status=dict(Counter(r['status'] for r in rr)),
                foreign_total=sum(not r['own_group'] for r in rr),
                late_count=sum(r['late']>1e-7 for r in rr),
                mean_late=st.mean(r['late'] for r in rr), max_late=max(r['late'] for r in rr),
                mean_route_return_late=st.mean(r['route_return_late'] for r in rr),
                typical_routes=[dict(route=list(k),count=v) for k,v in
                                Counter(tuple(r['actual_route']) for r in rr).most_common(2)],
                flips_toward=sum(c['new']==node for r in rr for c in r['direct_changes']),
                flips_away=sum(c['old']==node for r in rr for c in r['direct_changes']),
                negative_flips_away=sum(c['old']==node and c['old_bias']<0
                                        for r in rr for c in r['direct_changes']))
    applicability = {}
    for scale, visits in rows.items():
        buckets = {}
        for r in visits:
            bucket = r['status']
            if bucket == 'own_group':
                bucket += '_prefix_match' if r['prefix_match'] else '_prefix_mismatch'
            buckets.setdefault(bucket, []).append(r)
        applicability[scale] = {k: dict(visits=len(v), late_visits=sum(x['late']>1e-7 for x in v),
            total_customer_late=sum(x['late'] for x in v),
            reference_late_at_these_visits=sum(x['reference_late'] for x in v),
            start_match=sum(x['reference_start_match'] for x in v)) for k,v in buckets.items()}
        assert all(abs(r['late']-r['reference_late'])<1e-6 for r in visits if r['prefix_match'])
        applicability[scale]['top_late_customers'] = sorted(
            [dict(node=int(n), late_count=s['late_count'], total_late=s['mean_late']*80)
             for n,s in summaries[scale].items()], key=lambda x:x['total_late'], reverse=True)[:10]
        assert len(visits) == 80 * len(targets)
    report = dict(scope='Actual selected visits and previously recorded same-state action flips. '
                       'No complete record of unselected candidates; no causal attribution.',
                  applicability=applicability,summaries=summaries,visits=rows)
    filename = 'all_customer_guidance_audit.json' if all_customers else 'target_guidance_audit.json'
    (ROOT/'analysis'/filename).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(applicability if all_customers else summaries,indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--all-customers', action='store_true')
    main(parser.parse_args().all_customers)
