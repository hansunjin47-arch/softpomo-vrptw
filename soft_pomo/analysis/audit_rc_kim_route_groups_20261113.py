"""Replay saved Test actions, never infer new routes or call LLM. Compare groups/order."""
import json
from collections import defaultdict
from pathlib import Path
import statistics as st
import torch
from experiments.concurrent_fleet_v3 import run as data
from experiments.concurrent_fleet_bias_guided.support import load
from experiments.concurrent_fleet_pair_kim.run import OnlineEnv
from experiments.concurrent_fleet_v9_guided.prompts import parse_test_response

ROOT=Path(__file__).resolve().parents[1]
B=ROOT/'server_export/rc101_pair100_200_l5_mask_kim_s42_46_20261111'
OUT=ROOT/'result/rc_kim_route_group_audit_20261113'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,indent=2),encoding='utf-8')
def mean(x):return st.mean(x) if x else None

def measures(routes,groups,scores):
    membership={n:k for k,g in enumerate(groups) for n in g}
    owner={n:v for v,r in routes.items() for n in r}
    assert len(owner)==100 and sum(map(len,routes.values()))==100
    follow=[membership[n]==membership[r[0]] for r in routes.values() for n in r[1:]]
    pairs=[owner[a]==owner[b] for g in groups for i,a in enumerate(g) for b in g[i+1:]]
    choose=lambda n:n*(n-1)/2
    rowpairs=sum(choose(len(r)) for r in routes.values())
    colpairs=sum(choose(len(g)) for g in groups)
    intersection=sum(choose(sum(n in g for n in r)) for r in routes.values() for g in groups)
    expected=rowpairs*colpairs/choose(100)
    ari=(intersection-expected)/(.5*(rowpairs+colpairs)-expected)
    d=dict(anchor_follow_rate=mean(follow),group_pair_same_vehicle=mean(pairs),
        adjusted_rand_index=ari,
        vehicles_per_group=mean([len({owner[n] for n in g}) for g in groups]),
        groups_per_vehicle=mean([len({membership[n] for n in r}) for r in routes.values()]),
        vehicles=len(routes))
    if scores is not None:
        ordered=[];top=[]
        for r in routes.values():
            # Anchor is selected before guidance is active, so exclude it.
            seq=[n for n in r[1:] if membership[n]==membership[r[0]]]
            for i,a in enumerate(seq):
                later=seq[i+1:]
                if later:top.append(scores[a]>=max(scores[b] for b in later))
                for b in later:
                    if abs(scores[a]-scores[b])>1e-9:ordered.append(scores[a]>scores[b])
        d.update(confidence_descending_pair_rate=mean(ordered),order_pairs=len(ordered),
                 highest_of_later_assigned_rate=mean(top))
    return d

@torch.no_grad()
def main():
    torch.set_num_threads(1);data.SOURCE=B/'dataset'
    groups=read(B/'artifacts/vehicle100/groups.json')
    cfg=read(ROOT/'server_export/rc101_pair100_200_l5_mask_kim_s42_46_20261111_received/runs/vehicle100/seed42/protocol.json')['configs']['5']
    rows=read(B/'dataset/split.json')['test'];allrows=[];summary=[]
    for seed in range(42,46):
        kimroot=ROOT/('result/rc101_pair100_l5_mask_kim_test_20261112' if seed<44 else 'result/rc101_pair100_l5_mask_kim_test_additional_20261113')/f'seed{seed}/instances'
        pure=next((ROOT/'result').glob(f'rc_pair_returnmask_v100_l5_s{seed}_*'))/'late5/test_history'
        for i,row in enumerate(rows):
            name=row['name'];folder=kimroot/name
            cachefile=OUT/'routes'/f's{seed}_{name}.json'
            calls=read(folder/'calls.json');call=calls[0]
            context=json.loads(call['request']['messages'][1]['content'])
            initial=parse_test_response(call['response'],groups,context['selectable_customers'])
            scores={n:float(initial[k][n]) for k,g in enumerate(groups) for n in g} if 'RAIN' in name else None
            if cachefile.exists():kr={int(k):v for k,v in read(cachefile).items()}
            else:
                inst,reports=load(row)
                env=OnlineEnv(inst,1,cfg['environment'],'cuda',groups=groups,scores=initial,beta=5.,reports=reports,on_disclosure=lambda *args:None)
                result=read(folder/'result.json');kr=defaultdict(list)
                for h in result['history']:
                    assert abs(float(env.time[0])-h['clock_min'])<.002,(seed,name,'clock')
                    action=torch.tensor([h['action_slot']],device='cuda')
                    new,v,n=env.decode_action(action)
                    if int(n[0])>0:kr[int(v[0])].append(int(n[0]))
                    env.step(action,validate=True)
                assert bool(env.done.all())
                assert abs(float(env.metrics()['reward'][0])-result['metrics']['reward'])<1e-4
                write(cachefile,kr)
            pr=defaultdict(list)
            for h in read(pure/(name+'.json')):
                clock,v,n,new,live=h[0]
                if live and n>0:pr[int(v)].append(int(n))
            for method,routes in [('Pure',pr),('Kim',kr)]:
                allrows.append(dict(seed=seed,name=name,kind=row['kind'],method=method,**measures(routes,groups,scores)))
            if i%20==19:print('replayed',seed,i+1,flush=True)
        for method in ('Pure','Kim'):
            rs=[r for r in allrows if r['seed']==seed and r['method']==method]
            rain=[r for r in rs if 'RAIN' in r['name']]
            s=dict(seed=seed,method=method,count=len(rs),rain_count=len(rain))
            for k in ('vehicles','anchor_follow_rate','group_pair_same_vehicle','vehicles_per_group','groups_per_vehicle','adjusted_rand_index'):
                s[k]=mean([r[k] for r in rs])
            for k in ('confidence_descending_pair_rate','highest_of_later_assigned_rate'):
                s[k]=mean([r[k] for r in rain if r[k] is not None])
            summary.append(s);print(json.dumps(s),flush=True)
        write(OUT/'status.json',dict(stage='running',completed_seeds=seed-41,target=4))
    write(OUT/'report.json',dict(scope='Actual saved Test80 per seed42-45. Group composition all80. Order only Rain40 with shared fixed initial scores; exclude anchor, cross-anchor-group customers and score ties. Pair order is descriptive, not feasible-action or causal conformity. No inference, API or training.',summary=summary,instances=allrows))
    write(OUT/'status.json',dict(stage='complete',completed_seeds=4,target=4))

if __name__=='__main__':main()
