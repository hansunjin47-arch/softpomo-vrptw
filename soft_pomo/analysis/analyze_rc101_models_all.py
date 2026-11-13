"""Audit saved routes and confidence caches without new inference or API calls."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vrptw_env import load_solomon, _solution_stats

OUT = ROOT/'analysis/rc101_comprehensive_20260928'
PURE = 'result/rc101_pilot_discrete_pure1000_20260923_115506'
RUNS = dict(kim='rc101_kim_independent1000',
            phase2='rc101_pure500_fullconfidence_phase2',
            v2='rc101_pure500_return_aware_v2')


def read(path):
    return json.loads((ROOT/path).read_text())


def main():
    OUT.mkdir(exist_ok=True)
    models = {k: read(f'result/{v}/eval_1000_test80/results.json') for k, v in RUNS.items()}
    models.update(pure500=read(PURE+'/eval_500/results.json'),
                  pure1000=read(PURE+'/eval_1000_saved_routes/results.json'))
    details, summaries = {}, {}
    for model, rows in models.items():
        cases = {}
        for row in rows:
            name = row['name']
            inst = load_solomon(str(ROOT/'data/rc101_event_pilot_discrete_v1'/(name+'.txt')))
            replay = _solution_stats(inst, row['routes'])
            assert abs(replay['total_late']-row['total_late']) < 1e-6
            assert abs(replay['total_travel_time']-row['travel_time']) < 1e-6
            assert replay['n_served'] == 100
            depot_late = sum(v['ret_late'] for v in replay['per_vehicle'])
            nodes = {str(s['node']): dict(arrival=s['arrival'], late=s['late'],
                     deadline=s['tw_close'], route=v['stops'])
                     for v in replay['per_vehicle'] for s in v['stop_details']}
            cost = .04*row['vehicles'] + row['travel_time']/(24*inst['T']) + 9*row['total_late']/(5*inst['T'])
            cases[name] = dict(row, customer_late=row['total_late']-depot_late,
                              depot_late=depot_late, nodes=nodes, config_e_cost=cost,
                              late_returns=[dict(route=v['stops'], late=v['ret_late'])
                                            for v in replay['per_vehicle'] if v['ret_late']>0])
        details[model] = cases
        summaries[model] = {}
        for kind in ['ALL', 'RAIN', 'ACCIDENT']:
            rs = [r for r in cases.values() if kind=='ALL' or r['kind']==kind]
            summaries[model][kind] = {key: float(np.mean([r[key] for r in rs])) for key in
                ['vehicles','travel_time','late_count','total_late','customer_late','depot_late','config_e_cost']}
            summaries[model][kind].update(zero=sum(r['late_count']==0 for r in rs),
                p95_lateness=float(np.percentile([r['total_late'] for r in rs],95)),
                max_lateness=max(r['total_late'] for r in rs))
    paired = {}
    for a,b in [('pure500','pure1000'),('pure1000','kim'),('pure500','phase2'),
                ('phase2','v2'),('pure1000','v2'),('kim','v2')]:
        ds=[]
        for name,x in details[a].items():
            y=details[b][name]
            delta={k:y[k]-x[k] for k in ['vehicles','travel_time','total_late','customer_late','depot_late','config_e_cost']}
            new=[n for n in x['nodes'] if x['nodes'][n]['late']<=1e-6 and y['nodes'][n]['late']>1e-6]
            fixed=[n for n in x['nodes'] if x['nodes'][n]['late']>1e-6 and y['nodes'][n]['late']<=1e-6]
            ds.append(dict(name=name,delta=delta,new=new,fixed=fixed))
        paired[a+'->'+b]=dict(
            changes={k:dict(better=sum(r['delta'][k]<-1e-6 for r in ds),
                           same=sum(abs(r['delta'][k])<=1e-6 for r in ds),
                           worse=sum(r['delta'][k]>1e-6 for r in ds)) for k in ds[0]['delta']},
            joint_success=sum(r['delta']['vehicles']<0 and r['delta']['total_late']<=1e-6 for r in ds),
            new_customer_pairs=sum(len(r['new']) for r in ds),fixed_customer_pairs=sum(len(r['fixed']) for r in ds),
            worst=sorted(ds,key=lambda r:r['delta']['total_late'],reverse=True)[:5])
    confidence={}
    for model in ['phase2','v2']:
        confidence[model]={}
        for split, folder in [('train','cache'),('test','eval_1000_test80/cache')]:
            depot, customers=[],[]
            for path in (ROOT/'result'/RUNS[model]/folder).glob('*_rl_cluster.json'):
                cache=json.loads(path.read_text())
                depot.extend(cache['_depot'].values())
                customers.extend(v for k,g in cache.items() if not k.startswith('_') for v in g.values())
            confidence[model][split]=dict(depot_mean=float(np.mean(depot)),
                depot_zero=sum(v==0 for v in depot),depot_count=len(depot),customer_mean=float(np.mean(customers)))
    result=dict(summary=summaries,paired=paired,confidence=confidence)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (OUT/'details.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
