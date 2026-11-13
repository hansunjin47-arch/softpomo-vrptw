"""Read-only score-pattern review of the September 20 confidence caches."""
import json
import math
import re
import statistics
from collections import defaultdict, Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def main():
    aggregates=defaultdict(lambda:dict(scores=[],depot=[],count=Counter()))
    mismatches=[]
    waves=[]
    for family in ['r1','rc1']:
        folder=ROOT/f'result_soft/llm_cache_{family}_conf_sonnet46'
        for path in sorted(folder.glob('*_rl_cluster.json')):
            name=path.stem.removesuffix('_rl_cluster')
            split='test' if name.split('_')[0].endswith('101') else 'train'
            scenario='RAIN2' if '_RAIN2' in name else 'BASE'
            d=read(path)
            ids=[int(n) for k,v in d.items() if k.isdigit() for n in v]
            assert sorted(ids)==list(range(1,101)),path
            g=aggregates[f'{family}/{split}/{scenario}'];c=g['count']
            c['files']+=1
            for k,vs in d.items():
                if not k.isdigit():continue
                ns=list(d['_stats'][k]['node_detail'])
                assert set(ns)==set(vs)
                assert all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=1 for v in vs.values())
                g['scores']+=list(vs.values());g['depot'].append(d['_depot'][k]);c['clusters']+=1
                slack={n:d['_stats'][k]['node_detail'][n]['slack'] for n in ns}
                negative=[n for n in ns if slack[n]<0]
                positive=[n for n in ns if slack[n]>=0]
                c['late_nodes']+=len(negative)
                c['late_clusters']+=bool(negative)
                c['all_late_outrank_on_time']+=bool(negative and positive) and min(vs[n] for n in negative)>max(vs[n] for n in positive)
                c['mixed_late_clusters']+=bool(negative and positive)
                c['route_monotone']+=all(vs[a]>=vs[b] for a,b in zip(ns,ns[1:]))
                if len(negative)!=d['_stats'][k]['late_count']:
                    mismatches.append(dict(name=name,cluster=k,reported=d['_stats'][k]['late_count'],negative=len(negative)))
                for i,a in enumerate(ns):
                    for b in ns[i+1:]:
                        if vs[a]!=vs[b]:
                            c['route_pairs']+=1;c['route_agree']+=vs[a]>vs[b]
                        if (slack[a]<0)!=(slack[b]<0):
                            late,ontime=(a,b) if slack[a]<0 else (b,a)
                            c['late_pairs']+=1;c['late_priority_agree']+=vs[late]>vs[ontime]
            if scenario!='BASE':continue
            previous=d
            for index,wpath in enumerate(sorted(folder.glob(name+'_ACC2_acc_refresh_*.json'))):
                wave=read(wpath); wg=aggregates[f'{family}/{split}/ACC{index+1}'];wc=wg['count'];wc['files']+=1
                assert {k:set(v) for k,v in wave.items() if k.isdigit()}=={k:set(v) for k,v in d.items() if k.isdigit()}
                prompt=(folder/f'{name}_ACC2_acc_prompt_{index}.txt').read_text(encoding='utf-8')
                prompted={int(n) for line in prompt.splitlines() if '[RL visit order,' in line for n in re.findall(r'\d+',line.split(']')[-1])}
                changed=[]
                for k,vs in wave.items():
                    if not k.isdigit():continue
                    wc['clusters']+=1;wg['depot'].append(wave['_depot'][k]);wg['scores']+=list(vs.values())
                    for n,v in vs.items():
                        assert math.isfinite(v) and (0<=v<=1 or v==-1)
                        wc['negative_scores']+=v<0
                        if abs(v-previous[k][n])>1e-6:changed.append(int(n))
                waves.append(dict(name=name,wave=index+1,changed=len(changed),prompted=len(prompted),changed_outside_prompt=sum(n not in prompted for n in changed)))
                previous=wave
    result={}
    for key,g in aggregates.items():
        result[key]=dict(mean=statistics.mean(g['scores']),minimum=min(g['scores']),maximum=max(g['scores']),positive_depots=sum(v>0 for v in g['depot']),depot_max=max(g['depot']),counts=dict(g['count']))
    out=dict(summary=result,late_count_mismatches=mismatches,waves=waves)
    (ROOT/'analysis/pure500_conf_cache_patterns.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
