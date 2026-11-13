"""Descriptive audit; score ordering is not a routing-quality label."""
import json
from pathlib import Path
import statistics as st
from collections import Counter, defaultdict
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vrptw_env import load_solomon

NEW=ROOT/'result_soft/llm_cache_kim500_sonnet46_c1_20260918_113254'
OLD=ROOT/'result_soft/llm_cache_kim_initial_v1_20260917_214827/c1'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def flat(d):
    return {int(n):v for k,c in d.items() if k.isdigit() for n,v in c.items()}


def main():
    global NEW, OLD
    rc = '--rc' in sys.argv
    r = '--r' in sys.argv
    if r:
        NEW=ROOT/'result_soft/llm_cache_kim500_sonnet46_r1_v3_20260918_230759'
        OLD=ROOT/'result_soft/llm_cache_kim_initial_v1_20260917_214827/r1'
    if rc:
        NEW=ROOT/'result_soft/llm_cache_kim500_sonnet46_rc1_20260918_123744'
        OLD=ROOT/'result_soft/llm_cache_kim_initial_v1_20260917_214827/rc1'
    groups=defaultdict(lambda:dict(scores=[],depots=[],counts=Counter(),steps=Counter()))
    records=[]
    names=[f'RC{i}' for i in range(101,109)] if rc else [f'C{i}' for i in range(101,110)]
    if r:
        names=[f'R{i}' for i in range(101,113)]
    for name in names:
        folder=NEW/('evaluation' if name.endswith('101') else 'training')
        inst=load_solomon(str(ROOT.parent/f'data/Solomon/{name.lower()}.txt'))
        base=read(folder/f'{name}_rl_cluster.json')
        previous=base
        paths=[folder/f'{name}_rl_cluster.json',folder/f'{name}_RAIN2_rl_cluster.json']+sorted(p for p in folder.glob(f'{name}_ACC2_acc_refresh_*.json') if p.stem.rsplit('_',1)[-1].replace('.','',1).isdigit())
        for index,path in enumerate(paths):
            label=['BASE','RAIN2','ACC1','ACC2','ACC3'][index]
            d=read(path)
            oldpath=OLD/(f'{name}_cluster.json' if index==0 else f'{name}_RAIN2_cluster.json' if index==1 else path.name)
            old=read(oldpath)
            values=flat(d); oldvalues=flat(old)
            record=dict(name=name,scenario=label,clusters=len(d['_depot']),old_clusters=len(old['_depot']),
                mean=st.mean(values.values()),old_mean=st.mean(oldvalues.values()),
                old_mae=st.mean(abs(v-oldvalues[n]) for n,v in values.items()),
                reference_late=sum(v['late_count'] for v in d['_stats'].values()),
                reference_total_late=sum(v['total_late'] for v in d['_stats'].values()))
            for version,data in [('old',old),('new',d)]:
                split='test' if name.endswith('101') else 'train'
                g=groups[(split+'/' if rc or r else '')+version+'/'+label]; c=g['counts']
                for k,vs in data.items():
                    if not k.isdigit(): continue
                    ns=list(vs); scores=list(vs.values())
                    g['scores']+=scores; g['depots'].append(data['_depot'][k]); c['clusters']+=1
                    for order_name,order in [('route',ns),('tw',sorted(ns,key=lambda n:(float(inst['node_tw_close'][int(n)-1]),float(inst['node_tw_open'][int(n)-1]),int(n))))]:
                        if version=='old' and order_name=='route': continue
                        c[order_name+'_monotone']+=all(vs[a]>=vs[b] for a,b in zip(order,order[1:]))
                        for i,a in enumerate(order):
                            for b in order[i+1:]:
                                c[order_name+'_pairs']+=1
                                c[order_name+'_ties']+=vs[a]==vs[b]
                                c[order_name+'_agree']+=vs[a]>vs[b]
                    if version=='new':
                        g['steps'].update(str(round(vs[a]-vs[b],4)) for a,b in zip(ns,ns[1:]))
            if index>=2:
                pf=flat(previous); affected={n for e in load_solomon(str(ROOT.parent/f'data/Solomon/{name.lower()}_acc2.txt'))['preset_events'] if e['type']=='ACCIDENT' and e['trigger_time']/float(inst['T'])<=float(path.stem.rsplit('_',1)[-1])+1e-6 for n in e['nodes']}
                changes={n:round(v-pf[n],5) for n,v in values.items() if abs(v-pf[n])>1e-8}
                flips=[]
                for k,vs in d.items():
                    if not k.isdigit(): continue
                    ns=list(vs)
                    for i,a in enumerate(ns):
                        for b in ns[i+1:]:
                            if (vs[a]-vs[b])*(previous[k][a]-previous[k][b]) < -1e-10:
                                flips.append([int(a),int(b)])
                record.update(changed=len(changes),changed_outside_revealed_nodes=sum(n not in affected for n in changes),
                    increases=sum(v>0 for v in changes.values()),decreases=sum(v<0 for v in changes.values()),
                    reversals=len(flips),largest_changes=sorted(changes.items(),key=lambda x:-abs(x[1]))[:6])
                previous=d
            records.append(record)
    summary={}
    for key,g in groups.items():
        summary[key]=dict(mean=st.mean(g['scores']),minimum=min(g['scores']),maximum=max(g['scores']),
            nonzero_depots=sum(v!=0 for v in g['depots']),depot_max=max(g['depots']),
            counts=g['counts'],common_adjacent_steps=g['steps'].most_common(6))
    report=dict(groups=summary,records=records)
    (ROOT/('analysis/r_kim500_confidence_patterns.json' if r else 'analysis/rc_kim500_confidence_patterns.json' if rc else 'analysis/kim500_confidence_patterns.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
