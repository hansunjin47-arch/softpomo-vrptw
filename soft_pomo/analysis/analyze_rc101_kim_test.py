"""Paired saved-route diagnostics; no inference, solver or LLM calls."""
import json
import argparse
from pathlib import Path
import sys
from collections import Counter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vrptw_env import load_solomon, _solution_stats


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate',choices=['kim','phase2'],default='kim')
    parser.add_argument('--baseline',choices=['500','1000'],default='1000')
    args=parser.parse_args()
    out = ROOT/('analysis/rc101_kim_test_diagnostics' if args.candidate=='kim' and args.baseline=='1000'
                else f'analysis/rc101_{args.candidate}_vs_pure{args.baseline}_diagnostics')
    out.mkdir(exist_ok=True)
    candidate_run='rc101_kim_independent1000' if args.candidate=='kim' else 'rc101_pure500_fullconfidence_phase2'
    kim = json.loads((ROOT/f'result/{candidate_run}/eval_1000_test80/results.json').read_text())
    baseline_dir='eval_1000_saved_routes' if args.baseline=='1000' else 'eval_500'
    pure = {r['name']: r for r in json.loads((ROOT/f'result/rc101_pilot_discrete_pure1000_20260923_115506/{baseline_dir}/results.json').read_text())}
    rows = []
    for k in kim:
        p = pure[k['name']]
        inst = load_solomon(str(ROOT/'data/rc101_event_pilot_discrete_v1'/(k['name']+'.txt')))
        maps = []
        depot = []
        for row in [p, k]:
            stats = _solution_stats(inst, row['routes'])
            assert abs(stats['total_late']-row['total_late']) < 1e-6
            assert stats['late_count'] == row['late_count']
            depot.append(dict(late_count=sum(v['ret_late']>0 for v in stats['per_vehicle']),
                total_late=sum(v['ret_late'] for v in stats['per_vehicle']),
                late_routes=[dict(route=row['routes'][i],arrival=v['arr_depot'],late=v['ret_late'])
                    for i,v in enumerate(stats['per_vehicle']) if v['ret_late']>0]))
            nodes = {}
            for vehicle, pv in enumerate(stats['per_vehicle']):
                route = [n for n in row['routes'][vehicle] if n]
                for pos, stop in enumerate(pv['stop_details']):
                    if stop['node']:
                        nodes[stop['node']] = dict(stop, position=pos+1, route=route)
            maps.append(nodes)
        a,b = maps
        new = sorted(n for n in b if b[n]['late'] > 1e-6 and a[n]['late'] <= 1e-6)
        fixed = sorted(n for n in a if a[n]['late'] > 1e-6 and b[n]['late'] <= 1e-6)
        remain = sorted(n for n in a if a[n]['late'] > 1e-6 and b[n]['late'] > 1e-6)
        late_nodes = sorted(set(new+fixed+remain))
        rows.append(dict(name=k['name'],kind=k['kind'],pure={x:p[x] for x in ['vehicles','travel_time','late_count','total_late']},
            kim={x:k[x] for x in ['vehicles','travel_time','late_count','total_late']},
            delta={x:k[x]-p[x] for x in ['vehicles','travel_time','late_count','total_late']},
            new_late=new,fixed_late=fixed,remaining_late=remain,
            depot=dict(pure=depot[0],kim=depot[1]),
            customer_details={str(n):dict(pure=a[n],kim=b[n]) for n in late_nodes},
            event=inst['preset_events'][0]))
    summary = {}
    for group in ['RAIN','ACCIDENT','ALL']:
        rs=[r for r in rows if group=='ALL' or r['kind']==group]
        changes={}
        for key in ['vehicles','travel_time','late_count','total_late']:
            ds=[r['delta'][key] for r in rs]
            changes[key]=dict(better=sum(d < -1e-6 for d in ds),same=sum(abs(d)<=1e-6 for d in ds),
                worse=sum(d > 1e-6 for d in ds),mean=float(np.mean(ds)),median=float(np.median(ds)))
        worst=sorted(rs,key=lambda r:r['delta']['total_late'],reverse=True)
        summary[group]=dict(count=len(rs),changes=changes,
            candidate=args.candidate,baseline=args.baseline,
            depot_total_late={m:sum(r['depot'][m]['total_late'] for r in rs) for m in ['pure','kim']},
            new_late_customers=sum(len(r['new_late']) for r in rs),
            fixed_late_customers=sum(len(r['fixed_late']) for r in rs),
            new_late_frequencies=Counter(n for r in rs for n in r['new_late']).most_common(10),
            fewer_vehicles_more_lateness=sum(r['delta']['vehicles']<0 and r['delta']['total_late']>1e-6 for r in rs),
            fewer_vehicles_no_more_lateness=sum(r['delta']['vehicles']<0 and r['delta']['total_late']<=1e-6 for r in rs),
            top3_positive_lateness_delta=sum(max(0,r['delta']['total_late']) for r in worst[:3]),
            all_positive_lateness_delta=sum(max(0,r['delta']['total_late']) for r in rs),
            worst=[{k:r[k] for k in ['name','pure','kim','delta','new_late','fixed_late']} for r in worst[:3]],
            best=[{k:r[k] for k in ['name','pure','kim','delta']} for r in sorted(rs,key=lambda r:r['delta']['total_late'])[:2]])
    (out/'details.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
