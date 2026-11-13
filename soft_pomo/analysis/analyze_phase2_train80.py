"""Paired, saved-route diagnostic for train80 Phase2; no new inference."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vrptw_env import load_solomon, _solution_stats


def read(path):
    return json.loads((ROOT/path).read_text())


def main():
    run = 'result/rc101_pure500_train80_phase2_20260930'
    paths = dict(phase2=run+'/eval_1000_test80/results.json',
        pure500='result/rc101_pure_train80_fresh1000_20260929/eval_500/results.json',
        pure1000='result/rc101_pure_train80_fresh1000_20260929/eval_1000_saved_routes/results.json',
        kim='result/rc101_kim_train80_fresh1000_20260929/eval_1000_test80/results.json')
    cases = {}
    for model, path in paths.items():
        cases[model] = {}
        for row in read(path):
            inst = load_solomon(str(ROOT/'data/rc101_train80_test80_20260929'/(row['name']+'.txt')))
            s = _solution_stats(inst, row['routes'])
            assert abs(s['total_late']-row['total_late']) < 1e-5
            assert abs(s['total_travel_time']-row['travel_time']) < 1e-5
            base = sum(float(s['tt'][a,b]) for r in row['routes'] for a,b in zip(r,r[1:]))
            depot = sum(v['ret_late'] for v in s['per_vehicle'])
            nodes = {str(d['node']): dict(d, route=v['stops']) for v in s['per_vehicle'] for d in v['stop_details']}
            cases[model][row['name']] = dict(row, base=base, event_extra=row['travel_time']-base,
                depot_late=depot, customer_late=row['total_late']-depot, nodes=nodes,
                cost=.04*row['vehicles']+row['travel_time']/(24*s['T'])+9*row['total_late']/(5*s['T']))
    metrics = ['vehicles','travel_time','base','event_extra','total_late','depot_late','customer_late','cost']
    summary = {}
    for model, rows in cases.items():
        summary[model] = {}
        for group in ['ALL','RAIN','ACCIDENT']:
            rs = [r for r in rows.values() if group=='ALL' or r['kind']==group]
            summary[model][group] = dict(
                **{k:float(np.mean([r[k] for r in rs])) for k in metrics},
                p95=float(np.percentile([r['total_late'] for r in rs],95)),
                maximum=max(r['total_late'] for r in rs),
                top5_lateness_share=sum(sorted([r['total_late'] for r in rs],reverse=True)[:5])/max(1e-9,sum(r['total_late'] for r in rs)))
    paired = {}
    for model in ['pure500','pure1000','kim']:
        diffs = [{k:r[k]-cases[model][n][k] for k in metrics} | dict(name=n)
                 for n,r in cases['phase2'].items()]
        paired[model] = dict(changes={k:dict(better=sum(d[k]<-1e-6 for d in diffs),
            worse=sum(d[k]>1e-6 for d in diffs),same=sum(abs(d[k])<=1e-6 for d in diffs)) for k in metrics},
            worst_late=sorted(diffs,key=lambda r:r['total_late'],reverse=True)[:5])
    worst = sorted(cases['phase2'].values(),key=lambda r:r['total_late'],reverse=True)[:5]
    examples = []
    for row in worst:
        n = row['name']
        cache = read(run+'/eval_1000_test80/cache/'+n+'_rl_cluster.json')
        ref = read(run+'/eval_1000_test80/requests/'+n+'/reference.json')
        examples.append(dict(name=n, models={m:{k:rs[n][k] for k in metrics} for m,rs in cases.items()},
            late_nodes=[dict(d, previous={m:rs[n]['nodes'][node] for m,rs in cases.items() if m!='phase2'})
                        for node,d in row['nodes'].items() if d['late']>0],
            routes=row['routes'],cache=cache,reference_clusters=ref['clusters']))
    confidence = {}
    for split, folder in [('train','cache'),('test','eval_1000_test80/cache')]:
        depots, scores = [], []
        for path in (ROOT/run/folder).glob('*_rl_cluster.json'):
            c = json.loads(path.read_text())
            depots.extend(c['_depot'].values())
            scores.extend(v for k,g in c.items() if not k.startswith('_') for v in g.values())
        confidence[split] = dict(depot_count=len(depots),depot_zero=sum(v==0 for v in depots),
            depot_mean=float(np.mean(depots)),customer_mean=float(np.mean(scores)))
    out = ROOT/'analysis/rc101_phase2_train80_audit'
    out.mkdir(exist_ok=True)
    report = dict(summary=summary,paired=paired,confidence=confidence)
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (out/'examples.json').write_text(json.dumps(examples,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
