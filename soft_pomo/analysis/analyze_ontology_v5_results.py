"""Replay-based descriptive analysis of v5; no API calls or model updates."""
import hashlib
import json
from pathlib import Path
import statistics as st
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'server_export/soft_pomo_seed_sweep/train20'
sys.path.insert(0, str(FROZEN))
from vrptw_env import load_solomon, _solution_stats
import numpy as np

NEW = ROOT / 'result/rc101_ontology_v5_train20_b128_seed43/eval_1000_test80'
OLD = ROOT / 'result/rc101_relation_train20_cache_v1/eval_1000_test80'
KEYS = ('vehicles', 'travel_time', 'late_count', 'total_late', 'cost',
        'customer_late', 'depot_late', 'base_travel', 'event_travel', 'base_late')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def replay(inst, row):
    stats = _solution_stats(inst, row['routes'])
    base = _solution_stats(dict(inst, preset_events=[]), row['routes'])
    assert stats['n_served'] == 100
    assert sorted(n for r in row['routes'] for n in r if n) == list(range(1, 101))
    assert all(v['load'] <= stats['cap'] + 1e-4 for v in stats['per_vehicle'])
    for key, target in [('travel_time', 'total_travel_time'), ('total_late', 'total_late'), ('late_count', 'late_count')]:
        if key in row:
            assert abs(row[key] - stats[target]) < 1e-6, (row['name'], key)
    out = dict(name=row['name'], kind=row['kind'], vehicles=len(stats['per_vehicle']),
        travel_time=stats['total_travel_time'], total_late=stats['total_late'], late_count=stats['late_count'],
        customer_late=sum(s['late'] for v in stats['per_vehicle'] for s in v['stop_details']),
        depot_late=sum(v['ret_late'] for v in stats['per_vehicle']), base_travel=base['total_travel_time'],
        event_travel=stats['total_travel_time'] - base['total_travel_time'], base_late=base['total_late'])
    out['cost'] = .04 * out['vehicles'] + out['travel_time'] / 5760 + .0075 * out['total_late']
    return out, stats


def cache_info(folder, name):
    cache = read(folder / 'cache' / (name + '_rl_cluster.json'))
    groups = {k: v for k, v in cache.items() if not k.startswith('_')}
    member = {int(n): k for k, group in groups.items() for n in group}
    scores = {int(n): value for group in groups.values() for n, value in group.items()}
    return groups, member, scores, cache['_depot']


def guidance_details(name, stats, observation):
    groups, member, scores, depots = cache_info(NEW, name)
    reference = read(NEW / 'requests' / name / 'reference.json')['clusters']
    visits, routes = [], []
    for route_index, vehicle in enumerate(stats['per_vehicle']):
        first = vehicle['stops'][0]
        group = member[first]
        time = 0.0
        route_start_matches = first == reference[int(group)][0]
        routes.append(dict(first=first, group=int(group), matches_reference_start=route_start_matches,
                           stops=vehicle['stops'], load=vehicle['load']))
        for position, stop in enumerate(vehicle['stop_details']):
            n = stop['node']
            disclosed = observation['event_type'] == 'RAIN' or time >= observation['start_min']
            active = position > 0 and disclosed
            own_group = member[n] == group
            visits.append(dict(node=n, route=route_index, first=first, active_group=int(group),
                own_group=int(member[n]), in_active_group=own_group, bias_active=active,
                reference_start_matches=route_start_matches, own_score=scores[n],
                applied_score=(scores[n] if own_group else 0.0) if active else 0.0,
                depot_score=depots[group] if active else 0.0, departure=time,
                arrival=stop['arrival'], deadline=stop['tw_close'], late=stop['late']))
            time = stop['depart']
        visits.append(dict(node=0, route=route_index, first=first, departure=time,
            arrival=vehicle['arr_depot'], late=vehicle['ret_late']))
    return dict(routes=routes, visits=visits)


def summarize(rows):
    late = [r['total_late'] for r in rows]
    return dict(n=len(rows), means={k: st.mean(r[k] for r in rows) for k in KEYS},
        std={k: st.stdev(r[k] for r in rows) for k in KEYS}, zero_late=sum(x == 0 for x in late),
        late_quantiles=dict(zip(('median', 'p90', 'p95', 'max'), np.quantile(late, [.5,.9,.95,1]).tolist())),
        top5_late_fraction=sum(sorted(late, reverse=True)[:5])/sum(late),
        vehicles_distribution={str(k): sum(r['vehicles'] == k for r in rows) for k in sorted(set(r['vehicles'] for r in rows))})


def main():
    raw = dict(v5=read(NEW/'results.json'), v1=read(OLD/'results.json'))
    archive_path = ROOT/'server_export/train20_seed43_results.tar.gz'
    with tarfile.open(archive_path) as archive:
        raw['pure'] = json.load(archive.extractfile('train20/runs/pure128_4000_seed43/eval_1000_test80/results.json'))
    names = [r['name'] for r in raw['v5']]
    assert len(names) == 80 and all([r['name'] for r in rows] == names for rows in raw.values())
    raw['reference500'] = [dict(name=r['name'], kind=r['kind'], routes=[[0]+c+[0] for c in
        read(NEW/'requests'/r['name']/'reference.json')['clusters']]) for r in raw['v5']]
    rows = {k: [] for k in raw}
    detail, examples, cache_summary = {}, {}, {}
    for index, name in enumerate(names):
        inst = load_solomon(str(FROZEN/'eval_addon/data'/(name+'.txt')))
        detail[name] = {}
        for method, records in raw.items():
            row, stats = replay(inst, records[index])
            rows[method].append(row)
            detail[name][method] = dict(metrics=row,
                late_visits=[dict(route=j, first=v['stops'][0], node=s['node'], late=s['late'])
                    for j,v in enumerate(stats['per_vehicle']) for s in v['stop_details'] if s['late'] > 0],
                depot_late=[dict(route=j, first=v['stops'][0], late=v['ret_late'])
                    for j,v in enumerate(stats['per_vehicle']) if v['ret_late'] > 0])
            if method == 'v5':
                observation = read(FROZEN/'eval_addon/data'/(name+'.observations.json'))
                detail[name]['guidance'] = guidance_details(name, stats, observation)
    for method, folder in [('v1',OLD),('v5',NEW)]:
        scores, depots, margins, ranges = [], [], [], []
        for name in names:
            groups, member, values, ds = cache_info(folder, name)
            scores.extend(values.values()); depots.extend(ds.values())
            margins.extend(v-ds[member[n]] for n,v in values.items())
            ranges.extend(max(g.values())-min(g.values()) for g in groups.values())
        cache_summary[method] = dict(customer_mean=st.mean(scores), depot_mean=st.mean(depots),
            margin_mean=st.mean(margins), group_range_mean=st.mean(ranges),
            uniform_groups=sum(x<1e-9 for x in ranges), groups=len(ranges))
    comparisons = {}
    for other in ('pure','v1','reference500'):
        comparisons[other] = {}
        for kind in ('ALL','RAIN','ACCIDENT'):
            pairs = [(a,b) for a,b in zip(rows[other],rows['v5']) if kind=='ALL' or a['kind']==kind]
            comparisons[other][kind] = {k: dict(mean_delta=st.mean(b[k]-a[k] for a,b in pairs),
                improved=sum(b[k]<a[k]-1e-8 for a,b in pairs), tied=sum(abs(b[k]-a[k])<=1e-8 for a,b in pairs),
                worsened=sum(b[k]>a[k]+1e-8 for a,b in pairs)) for k in KEYS}
        for key in ('cost','total_late'):
            ranked=sorted(zip(rows[other],rows['v5']),key=lambda p:p[1][key]-p[0][key])
            examples[other+'_'+key] = dict(better=[dict(name=a['name'],old=a[key],new=b[key],delta=b[key]-a[key]) for a,b in ranked[:5]],
                worse=[dict(name=a['name'],old=a[key],new=b[key],delta=b[key]-a[key]) for a,b in ranked[-5:][::-1]])
    visits=[s for d in detail.values() for s in d['guidance']['visits'] if s['node']]
    routes=[r for d in detail.values() for r in d['guidance']['routes']]
    applied={}
    for key, subset in [('all',visits),('late',[s for s in visits if s['late']>0])]:
        applied[key]=dict(count=len(subset), foreign=sum(not s['in_active_group'] for s in subset),
            active_foreign=sum(s['bias_active'] and not s['in_active_group'] for s in subset),
            active_own=sum(s['bias_active'] and s['in_active_group'] for s in subset),
            inactive=sum(not s['bias_active'] for s in subset),
            late_foreign=sum(s['late'] for s in subset if not s['in_active_group']),
            late_active_foreign=sum(s['late'] for s in subset if s['bias_active'] and not s['in_active_group']))
    report=dict(scope='Descriptive fixed-route replay. No causal attribution to ontology, depot scores, or assignment mismatch.',
        summary={m:{k:summarize([r for r in rs if k=='ALL' or r['kind']==k]) for k in ('ALL','RAIN','ACCIDENT')} for m,rs in rows.items()},
        comparisons=comparisons, examples=examples, test_cache_summary=cache_summary,
        applied_guidance=applied, reference_start_matches=sum(r['matches_reference_start'] for r in routes),
        final_routes=len(routes), scenarios=detail,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (NEW/'results.json',OLD/'results.json',archive_path,Path(__file__))})
    output=NEW/'detailed_analysis.json'
    output.write_text(json.dumps(report,indent=2,default=scalar,allow_nan=False),encoding='utf-8')
    compact={k:v for k,v in report.items() if k not in ('scenarios','source_sha256','comparisons')}
    print(json.dumps(compact,indent=2))


if __name__=='__main__':
    main()
