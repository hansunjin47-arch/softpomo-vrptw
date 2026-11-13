"""Compare archived LKH BASE routes under current replay and three RL evaluations."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vrptw_env import load_solomon,_solution_stats

sources={
    'Pure1000':'rc_pure1000_train_eval_20260920_195014',
    'Full confidence':'rc_full_confidence_train_eval_20260920_215555',
    'Selective confidence':'rc_selective_train_eval_20260920_194238',
}
data={k:json.loads((ROOT/'analysis'/v/'results.json').read_text())['results'] for k,v in sources.items()}
archive=json.loads((ROOT/'lkh_eval_results/rc1_full.json').read_text())
lkh=[]
for x in archive['instances']:
    if x['name'].lower() not in [f'rc{i}' for i in range(102,109)]: continue
    inst=load_solomon(str(ROOT.parent/'data/Solomon'/(x['name']+'.txt')))
    routes=[[0]+[n for n in r if n]+[0] for r in x['routes']]
    assert sorted(n for r in routes for n in r if n)==list(range(1,101))
    s=_solution_stats(inst,routes)
    lkh.append(dict(name=x['name'].upper(),scenario='BASE',vehicles=len(routes),
        distance=sum(float(v['rdist']) for v in s['per_vehicle']),late_stops=int(s['late_count']),
        total_late=float(s['total_late']),archived_distance=x['D'],archived_late=x['Lc'],routes=routes))
data['LKH-3 replay']=lkh
lines=['# RC train set: LKH-3 / full / selective', '',
       'LKH source: lkh_eval_results/rc1_full.json, 30s per solve. BASE only. Not a proof of optimality.',
       'Archived LKH uses rounded integer travel times. Replayed below with current floating-point travel and strict arrival deadline.',
       'No matching RAIN2/ACC2 oracle route archive was found. Do not treat BASE replay under events as an oracle.', '',
       '## Seven-instance sums', '',
       '| Scenario | Model | Vehicles sum | Distance sum | Late count sum | Total late sum |',
       '|---|---|---:|---:|---:|---:|']
summary=[]
for scenario in ['BASE','_RAIN2','_ACC2']:
    for model,rows in data.items():
        selected=[r for r in rows if r['scenario']==scenario]
        if not selected: continue
        row=dict(model=model,scenario=scenario,**{k:sum(r[k] for r in selected) for k in ['vehicles','distance','late_stops','total_late']})
        summary.append(row)
        lines.append(f"| {scenario} | {model} | {row['vehicles']} | {row['distance']:.2f} | {row['late_stops']} | {row['total_late']:.2f} |")
lines+=['','## BASE individual results','','| Instance | Model | Vehicles | Distance | Late count | Total late |','|---|---|---:|---:|---:|---:|']
for i in range(102,109):
    for model,rows in data.items():
        r=next(r for r in rows if r['name']==f'RC{i}')
        lines.append(f"| RC{i} | {model} | {r['vehicles']} | {r['distance']:.2f} | {r['late_stops']} | {r['total_late']:.2f} |")
out=ROOT/'analysis/rc_full_confidence_train_eval_20260920_215555'
(out/'lkh_comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(out/'lkh_comparison.json').write_text(json.dumps(dict(summary=summary,lkh_replay=lkh),indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
print('LKH archived sums',sum(r['archived_distance'] for r in lkh),sum(r['archived_late'] for r in lkh))
