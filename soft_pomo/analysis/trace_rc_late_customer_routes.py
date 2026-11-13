"""All originally late customers, using archived Pure500 solutions in every scenario."""
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vrptw_env import load_solomon, _solution_stats

old_dir=ROOT/'result_soft_kim_1000_b64_20260918_000534/eval_pure500_20260918_105232/rc1/rc102-rc108/plots'
new_dir=ROOT/'result_soft_rc1_selective_cluster_v2_b32_20260920_130657/training/rc102-rc108/plots'
cache_dir=ROOT/'result_soft/llm_cache_rc101_selective_advance_cluster_v2_sonnet46/evaluation'

def read(folder,name,inst):
    text=(folder/f'{name}_solution.txt').read_text(encoding='utf-8')
    routes=[[int(n.strip()) for n in row.split('->')] for row in re.findall(r'^\s*V\d+:\s*(.*?)\s*\[',text,re.M)]
    assert sorted(n for r in routes for n in r if n)==list(range(1,101))
    stats=_solution_stats(inst,routes)
    nodes={}
    for i,(r,v) in enumerate(zip(routes,stats['per_vehicle'])):
        for pos,s in enumerate(v['stop_details']):
            nodes[s['node']]=dict(s,vehicle=i+1,position=pos+1,route=r[1:-1])
    return nodes,stats

def route_text(route,target):
    return ' -> '.join(f'**{n}**' if n==target else str(n) for n in route)

report=['# Pure500 Late Customers: Complete RC101 Route Changes','',
        'All BASE/RAIN2/ACC2 baselines are actual archived Pure500 solution routes, not accident estimates. '
        'Timing is replayed under current instance files; lateness means arrival strictly after close. '
        'Vehicle indices are display labels, not persistent vehicle identities. Depot omitted from route arrows. '
        'A position change across different routes is not a controlled insertion experiment.','']
results=[]
for suffix in ['', '_RAIN2', '_ACC2']:
    name='RC101'+suffix
    inst=load_solomon(str(ROOT.parent/'data/Solomon'/f'{name.lower()}.txt'))
    old,oldstats=read(old_dir,name,inst); new,newstats=read(new_dir,name,inst)
    guidance=json.loads((cache_dir/('RC101' if suffix=='_ACC2' else name)/'attempt1/cache.json').read_text())['advance']
    late=sorted(n for n,d in old.items() if d['late']>0)
    report += [f'## {name}', '',f'Originally late customers: {late}', '',
               '| Customer | Old position -> new | Arrival old -> new | Lateness old -> new | BASE/scenario score |',
               '|---|---|---|---|---|']
    records=[]
    for n in late:
        a,b=old[n],new[n]
        report.append(f"| {n} | {a['position']} -> {b['position']} | {a['arrival']:.2f} -> {b['arrival']:.2f} | {a['late']:.2f} -> {b['late']:.2f} | {guidance.get(str(n),0)} |")
        new_late_peers=[dict(node=m,late=new[m]['late']) for m in b['route'] if m!=n and old[m]['late']==0 and new[m]['late']>0]
        records.append(dict(node=n,old=a,new=b,score=guidance.get(str(n),0),newly_late_peers=new_late_peers))
    for r in records:
        n=r['node'];a=r['old'];b=r['new']
        report += ['',f'### Customer {n}',f"- Before: {route_text(a['route'],n)}",f"- After: {route_text(b['route'],n)}",
            f"- Newly late customers in the final route: {', '.join(str(p['node'])+' ('+format(p['late'],'.2f')+')' for p in r['newly_late_peers']) or 'none'}. This is co-occurrence, not causal attribution."]
    returns=[]
    for label,stats in [('Pure500',oldstats),('Phase2',newstats)]:
        for i,v in enumerate(stats['per_vehicle']):
            if v['ret_late']>0:
                returns.append(dict(run=label,vehicle=i+1,last_customer=v['prev_last'],arrival=v['arr_depot'],late=v['ret_late']))
    report += ['', 'Depot-return lateness (separate from customer lateness):', '```json',json.dumps(returns,indent=2),'```','']
    results.append(dict(name=name,records=records,returns=returns))
    print(name)
    for r in records:
        a,b=r['old'],r['new']
        print(r['node'],a['route'],'=>',b['route'], 'arrival',round(a['arrival'],2),round(b['arrival'],2),'late',round(a['late'],2),round(b['late'],2))
    print('RETURNS',returns)
(ROOT/'analysis/rc_late_customer_route_changes.md').write_text('\n'.join(report),encoding='utf-8')
(ROOT/'analysis/rc_late_customer_route_changes.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
