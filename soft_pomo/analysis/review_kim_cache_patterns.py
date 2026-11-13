"""Read-only descriptive audit of Kim confidence caches and scenario deltas."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vrptw_env import load_solomon

RUN=ROOT/'result_soft/llm_cache_kim_initial_v1_20260917_214827'


def main():
    groups=defaultdict(lambda:dict(scores=[],depots=[],sizes=[],counts=Counter(),hist=Counter()))
    deltas=defaultdict(lambda:dict(changes=[],counts=Counter(),depot_changes=[]))
    instances=[]
    for family in ['c1','rc1','r1']:
        for path in sorted((RUN/family).glob('*_cluster.json')):
            scenario='RAIN2' if 'RAIN2' in path.name else 'BASE'
            name=path.stem.split('_')[0]
            data=json.loads(path.read_text())
            inst=load_solomon(str(ROOT.parent/f'data/Solomon/{name.lower()}.txt'))
            inspect(data,inst,groups[family+'/'+scenario])
            if scenario=='BASE':
                clusters=[v for k,v in data.items() if k.isdigit()]
                loads=[sum(float(inst['node_demand'][int(n)-1]) for n in c) for c in clusters]
                instances.append(dict(name=name,clusters=len(clusters),max_load=max(loads),
                    overloaded=sum(x>1.00001 for x in loads),singletons=sum(len(c)==1 for c in clusters)))
            else:
                base=json.loads((RUN/family/f'{name}_cluster.json').read_text())
                compare(base,data,deltas[family+'/RAIN2'])
        for name in [r['name'] for r in instances if ('rc1' if r['name'].startswith('RC') else 'c1' if r['name'].startswith('C') else 'r1')==family]:
            inst=load_solomon(str(ROOT.parent/f'data/Solomon/{name.lower()}.txt'))
            previous=json.loads((RUN/family/f'{name}_cluster.json').read_text())
            paths=sorted(p for p in (RUN/family).glob(f'{name}_ACC2_acc_refresh_*.json')
                         if p.stem.rsplit('_',1)[-1].replace('.','',1).isdigit())
            for wave,path in enumerate(paths,1):
                data=json.loads(path.read_text())
                inspect(data,inst,groups[family+f'/ACC{wave}'])
                indices=json.loads(path.with_name(path.stem+'_input.json').read_text())['original_cluster_indices']
                compare(previous,data,deltas[family+f'/ACC{wave}'],set(map(str,indices)))
                previous=data
    output=dict(groups={},deltas={},instances=instances)
    for key,g in groups.items():
        scores=g.pop('scores'); depots=g.pop('depots'); sizes=g.pop('sizes')
        output['groups'][key]=dict(g,mean=statistics.mean(scores),stdev=statistics.pstdev(scores),
            minimum=min(scores),maximum=max(scores),nodes=len(scores),clusters=len(sizes),
            mean_size=statistics.mean(sizes),max_size=max(sizes),
            depot_nonzero=sum(d!=0 for d in depots),depot_max=max(depots),
            common_scores=g['hist'].most_common(6))
    for key,d in deltas.items():
        v=d.pop('changes'); dv=d.pop('depot_changes')
        output['deltas'][key]=dict(d,nodes=len(v),changed=sum(abs(x)>1e-8 for x in v),
            increased=sum(x>1e-8 for x in v),decreased=sum(x< -1e-8 for x in v),
            mean_delta=statistics.mean(v),mean_abs_delta=statistics.mean(map(abs,v)),
            max_abs_delta=max(map(abs,v)),depot_changed=sum(abs(x)>1e-8 for x in dv))
    path=ROOT/'analysis/kim_initial_cache_patterns.json'
    path.write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2))


def inspect(data,inst,g):
    c=g['counts']; c['files']+=1
    for k,values in data.items():
        if not k.isdigit(): continue
        ns=list(values); scores=list(values.values()); depot=data['_depot'][k]
        g['scores']+=scores; g['depots'].append(depot); g['sizes'].append(len(ns))
        g['hist'].update(map(str,scores))
        c['constant_clusters']+=len(set(scores))==1
        c['below_depot']+=sum(v<depot for v in scores)
        c['equal_depot']+=sum(v==depot for v in scores)
        c['zero']+=scores.count(0); c['one']+=scores.count(1)
        for label,key in [('id',lambda n:int(n)),('tw',lambda n:(float(inst['node_tw_close'][int(n)-1]),float(inst['node_tw_open'][int(n)-1]),int(n)))]:
            order=sorted(ns,key=key)
            c[label+'_monotonic']+=all(values[a]>=values[b] for a,b in zip(order,order[1:]))
            for i,a in enumerate(order):
                for b in order[i+1:]:
                    c[label+'_pairs']+=1
                    c[label+'_ties']+=values[a]==values[b]
                    c[label+'_agree']+=values[a]>values[b]


def compare(previous,data,g,indices=None):
    for k,values in data.items():
        if not k.isdigit() or indices is not None and k not in indices: continue
        change=[v-previous[k][n] for n,v in values.items()]
        g['changes']+=change
        g['depot_changes'].append(data['_depot'][k]-previous['_depot'][k])
        g['counts']['clusters']+=1
        g['counts']['unchanged_clusters']+=all(abs(v)<1e-8 for v in change)
        ns=list(values)
        for i,a in enumerate(ns):
            for b in ns[i+1:]:
                old=previous[k][a]-previous[k][b]; new=values[a]-values[b]
                g['counts']['pairs']+=1
                g['counts']['strict_reversals']+=old*new < -1e-12


if __name__=='__main__':main()
