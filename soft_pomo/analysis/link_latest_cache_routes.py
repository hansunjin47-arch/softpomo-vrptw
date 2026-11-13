"""Link saved final BASE routes to reference slack and cluster-scoped bias."""
import json
import re
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vrptw_env import load_solomon,_solution_stats

results=[]
for family,prefix,pool in [('r1','R','r102-r112'),('rc1','RC','rc102-rc108')]:
    name=prefix+'101'
    cache=json.loads((ROOT/f'result_soft/llm_cache_{family}_conf_sonnet46/{name}_rl_cluster.json').read_text())
    text=(ROOT/f'result_soft_{family}_conf_sonnet46/{pool}/plots/{name}_solution.txt').read_text(encoding='utf-8')
    routes=[list(map(int,re.findall(r'\d+',r))) for r in re.findall(r'V\d+: (.*?)\s+\[',text)]
    stats=_solution_stats(load_solomon(str(ROOT.parent/'data/Solomon'/(name.lower()+'.txt'))),routes)
    membership={n:k for k,v in cache.items() if k.isdigit() for n in v}
    seen=set();nodes=[]
    for route,v in zip(routes,stats['per_vehicle']):
        anchor=membership[str(route[1])]
        for pos,s in enumerate(v['stop_details']):
            n=str(s['node']);own=membership[n]
            active=bool(set(cache[anchor])-seen) and pos>0
            bias=5*cache[anchor].get(n,0) if active else 0
            nodes.append(dict(node=int(n),reference_cluster=int(own),anchor_cluster=int(anchor),
                reference_slack=cache['_stats'][own]['node_detail'][n]['slack'],
                cached_score=cache[own][n],inferred_added_bias=bias,
                late=s['late'],route=route,position=pos+1))
            seen.add(n)
    oldlate=[n for n in nodes if n['reference_slack']<0]
    newly=[n for n in nodes if n['reference_slack']>=0 and n['late']>0]
    result=dict(name=name,reference_late_customers=len(oldlate),fixed=sum(n['late']==0 for n in oldlate),
        new_late_customers=len(newly),new_late_amount=sum(n['late'] for n in newly),
        persistent_late_amount=sum(n['late'] for n in oldlate),
        depot_late_count=sum(v['ret_late']>0 for v in stats['per_vehicle']),
        depot_late_amount=sum(v['ret_late'] for v in stats['per_vehicle']),
        late_customers_without_own_group_bias=sum(n['late']>0 and n['inferred_added_bias']==0 for n in nodes),
        nodes=nodes)
    results.append(result)
    print(json.dumps({k:v for k,v in result.items() if k!='nodes'}))
    print('late with no own-group bonus',[(n['node'],round(n['late'],2),n['cached_score']) for n in nodes if n['late']>0 and n['inferred_added_bias']==0])
(ROOT/'analysis/latest_cache_route_links.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
