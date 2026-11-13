"""Offline structural route audit; does not modify policies or caches."""
import collections
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'server_export/kim_tt_ac_train20_20261010/A'
sys.path.insert(0, str(SOURCE))
from vrptw_env import load_solomon, _solution_stats

VERSION = 'ontology_routes_only_ac_travel_time_v1'
OUT = ROOT / 'analysis/tt_ac_route_diagnosis_20261011'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def relations(routes):
    pairs, ordered = set(), set()
    for route in routes:
        nodes = [n for n in route if n]
        for i, a in enumerate(nodes):
            for b in nodes[i+1:]:
                pairs.add(tuple(sorted((a, b))))
                ordered.add((a, b))
    return pairs, ordered


def main():
    OUT.mkdir(exist_ok=True)
    history = ROOT / 'analysis/rc101_pilot_history_20_80_20260923_104517'
    hrows = read(history / 'results.json')
    hroutes = read(history / 'routes.json')
    output = {}
    for variant in ('A', 'C'):
        paths = {
            'pure': ROOT / f'result/rc101_pure_train20_b128_s42_config{variant}_travel_time_1000_20261010_v2/eval_1000_test80/results.json',
            'kim': ROOT / f'result/rc101_kim_train20_config{variant}_{VERSION}/eval_1000_test80_gpu/results.json',
            'phase2': ROOT / f'result/rc101_phase2_train20_b128_s42_config{variant}_{VERSION}/eval_1000_test80_gpu/results.json',
        }
        rows = {method: read(path) for method, path in paths.items()}
        rows['history'] = [dict(row, routes=hroutes) for row in hrows]
        names = [r['name'] for r in rows['pure']]
        instances = {n: load_solomon(str(SOURCE / 'data' / (n + '.txt'))) for n in names}
        result = {}
        for method, records in rows.items():
            assert [r['name'] for r in records] == names
            details, late_nodes = [], collections.Counter()
            for row in records:
                name = row['name']
                stats = _solution_stats(instances[name], row['routes'])
                assert abs(stats['total_travel_time'] - row['travel_time']) < .02
                assert abs(stats['total_late'] - row['total_late']) < .02
                vehicles = stats['per_vehicle']
                stops = [s for v in vehicles for s in v['stop_details']]
                for s in stops:
                    if s['late'] > 0:
                        late_nodes[s['node']] += 1
                d = dict(name=name, base=float(stats['total_dist']),
                    event=row['travel_time']-float(stats['total_dist']),
                    depot_legs=sum(float(stats['tt'][0,v['stops'][0]])+v['ret_dist'] for v in vehicles),
                    load_mean=statistics.mean(v['load']/stats['cap'] for v in vehicles),
                    small_routes=sum(len(v['stops'])<=3 for v in vehicles),
                    wait=sum(s['wait'] for s in stops),
                    late_customers=sum(s['late']>0 for s in stops),
                    late_depots=sum(v['ret_late']>0 for v in vehicles),
                    customer_lateness=sum(s['late'] for s in stops), depot_lateness=sum(v['ret_late'] for v in vehicles))
                if method in ('kim','phase2'):
                    cache_root = paths[method].parent
                    if method == 'phase2':
                        cache_root = cache_root.parent / 'eval_1000_test80'
                    suffix = '_cluster.json' if method == 'kim' else '_rl_cluster.json'
                    cache = read(cache_root / 'cache' / (name+suffix))
                    group = {int(n): k for k, vals in cache.items() if not k.startswith('_') for n in vals}
                    d['off_start_group'] = sum(group[n]!=group[v['stops'][0]] for v in vehicles for n in v['stops'])
                    d['negative_customers'] = sum(float(c)<float(cache['_depot'][k]) for k,vals in cache.items() if not k.startswith('_') for c in vals.values())
                    d['depot_score'] = statistics.mean(cache['_depot'].values())
                    cp, co = relations(row['routes'])
                    pp, po = relations(rows['pure'][len(details)]['routes'])
                    shared = cp & pp
                    d['pair_reassignment'] = len(cp ^ pp)/len(cp | pp)
                    d['order_inversions_shared'] = sum((a,b) in co and (b,a) in po or (b,a) in co and (a,b) in po for a,b in shared)
                details.append(d)
            result[method] = dict(mean={k:statistics.mean(d[k] for d in details) for k in details[0] if k!='name'},
                top_late_customers=late_nodes.most_common(10), scenarios=details)
            if method in ('kim','phase2'):
                pairs = list(zip(rows['pure'], records))
                result[method]['paired_vs_pure'] = {
                    k:dict(better=sum(b[k]<a[k]-1e-6 for a,b in pairs),same=sum(abs(b[k]-a[k])<=1e-6 for a,b in pairs),worse=sum(b[k]>a[k]+1e-6 for a,b in pairs))
                    for k in ('vehicles','travel_time','late_count','total_late')}
                result[method]['joint_nonworse'] = sum(all(b[k]<=a[k]+1e-6 for k in ('vehicles','travel_time','late_count')) and any(b[k]<a[k]-1e-6 for k in ('vehicles','travel_time','late_count')) for a,b in pairs)
        output[variant] = result
        print(variant, json.dumps({m:{k:v for k,v in d.items() if k!='scenarios'} for m,d in result.items()}, default=lambda x: x.item()), flush=True)
    (OUT/'structural.json').write_text(json.dumps(output,indent=2,default=lambda x: x.item()),encoding='utf-8')


if __name__ == '__main__':
    main()
