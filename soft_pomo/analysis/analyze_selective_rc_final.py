"""Saved-route evidence only; inferred bonus eligibility is not a causal trace."""
import json
import re
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vrptw_env import load_solomon,_solution_stats

run=ROOT/'result_soft_rc1_selective_cluster_v2_b32_20260920_130657'
cache=ROOT/'result_soft/llm_cache_rc101_selective_advance_cluster_v2_sonnet46/evaluation'
refs=ROOT/'analysis/rc_pure500_reference_20260920_124625'
results=[]
for suffix in ['', '_RAIN2', '_ACC2']:
    name='RC101'+suffix
    inst=load_solomon(str(ROOT.parent/'data/Solomon'/f'{name.lower()}.txt'))
    ref=json.loads((refs/f'{"RC101" if suffix=="_ACC2" else name}.json').read_text())
    clusters=ref['clusters']
    base_name='RC101' if suffix=='_ACC2' else name
    base=json.loads((cache/base_name/'attempt1/cache.json').read_text())
    stages=[]
    if suffix=='_ACC2':
        for p in sorted(cache.glob('RC101_ACC2_wave_*/attempt1/cache.json')):
            d=json.loads(p.read_text());stages.append((d['events_visible_until']*float(inst['T']),d['advance']))
    text=(run/'training/rc102-rc108/plots'/f'{name}_solution.txt').read_text(encoding='utf-8')
    routes=[[int(n.strip()) for n in row.split('->')] for row in re.findall(r'^\s*V\d+:\s*(.*?)\s*\[',text,re.M)]
    stats=_solution_stats(inst,routes)
    oldstats=_solution_stats(inst,[[0]+c+[0] for c in clusters])
    membership={n:k for k,c in enumerate(clusters) for n in c}
    old={s['node']:s for v in oldstats['per_vehicle'] for s in v['stop_details']}
    nodes=[]
    for ri,(route,v) in enumerate(zip(routes,stats['per_vehicle'])):
        anchor=membership[route[1]]
        for pos,s in enumerate(v['stop_details']):
            n=s['node']; k=membership[n]
            nodes.append(dict(node=n,reference_cluster=k+1,anchor_cluster=anchor+1,
                selected_base=base['advance'].get(str(n),0),
                eligible_cluster=pos>0 and k==anchor,
                reference_position=clusters[k].index(n)+1,position=pos+1,
                reference_arrival=old[n]['arrival'],arrival=s['arrival'],
                reference_late=old[n]['late'],late=s['late'],route_index=ri+1,route=route))
    selected=[n for n in nodes if n['selected_base']>0]
    summary=dict(name=name,reference_note='ACC reference is BASE routes replayed under ACC, not a measured Pure500 ACC rollout' if suffix=='_ACC2' else 'Pure500 reference',
        old_late=sum(n['reference_late']>0 for n in nodes),new_late=sum(n['late']>0 for n in nodes),
        fixed=[n['node'] for n in nodes if n['reference_late']>0 and n['late']==0],
        introduced=[n['node'] for n in nodes if n['reference_late']==0 and n['late']>0],
        introduced_amount=sum(n['late'] for n in nodes if n['reference_late']==0),
        selected_count=len(selected),selected_eligible=sum(n['eligible_cluster'] for n in selected),
        selected_earlier=sum(n['arrival']<n['reference_arrival'] for n in selected),
        selected_old_late=sum(n['reference_late'] for n in selected),
        selected_new_late=sum(n['late'] for n in selected),
        depot_late=sum(v['ret_late'] for v in stats['per_vehicle']),
        cross_cluster_customers=sum(n['reference_cluster']!=n['anchor_cluster'] for n in nodes),
        nodes=nodes,stages=stages)
    results.append(summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['nodes','stages']}))
    print('SELECTED',json.dumps(selected))
    print('LATE',[(n['node'],round(n['late'],2),n['route_index']) for n in nodes if n['late']>0])
(ROOT/'analysis/selective_rc_final_diagnostics.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
