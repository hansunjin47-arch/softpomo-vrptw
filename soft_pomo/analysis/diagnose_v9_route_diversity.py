"""CPU-only sampled route diversity probe; no training or LLM calls."""
import json
import sys
from collections import Counter
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.concurrent_fleet_stop_init.policy import FleetPolicy
from experiments.concurrent_fleet_v9.env import TensorFleetEnv
from experiments.concurrent_fleet_v3.run import SOURCE,load,write


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    run=ROOT/'result/v9_stopbias_entropy1_700_late10_20261021'
    cfg=json.loads((run/'protocol.json').read_text())['configs']['10']
    # Use an immutable saved checkpoint, not checkpoint-last while it is rewritten.
    saved=torch.load(run/'late10/checkpoint-500.pt',map_location='cpu',weights_only=False)
    model=FleetPolicy(**cfg['model']).cpu().eval()
    model.load_state_dict(saved['model_state_dict'])
    split=json.loads((SOURCE/'split.json').read_text())['train']
    output=[]
    for index in [0,3,10]:
        torch.manual_seed(20261022+index)
        inst,public=load(split[index])
        env=TensorFleetEnv(inst,100,cfg['environment'],device='cpu')
        starts,encoded=model.encode_events(inst,public)
        routes=[[[] for _ in range(env.v)] for _ in range(100)]
        stats={k:[] for k in ['dispatch_binary','new_customer','routing_destination']}
        def record(key,probs,mask,valid):
            n=mask.sum(1)
            eligible=valid&(n>1)
            probs=probs/probs.sum(1,keepdim=True).clamp_min(1e-30)
            h=-(probs*probs.clamp_min(1e-30).log()).sum(1)
            for i in eligible.nonzero().flatten().tolist():
                stats[key].append((float(probs[i].max()),float(h[i]),float(h[i]/n[i].float().log())))
        for step in range(600):
            if bool(env.done.all()):break
            frame=env.frame();live=~env.done
            p=model.probabilities(frame,starts,encoded,env)
            dispatch=env.dispatch_open&live
            binary=torch.stack([p[:,:env.n].sum(1),p[:,-1]],1)
            bm=torch.stack([frame['mask'][:,:env.n].any(1),frame['mask'][:,-1]],1)
            record('dispatch_binary',binary,bm,dispatch)
            record('new_customer',p[:,:env.n],frame['mask'][:,:env.n],dispatch)
            record('routing_destination',p[:,env.n:2*env.n+1],frame['mask'][:,env.n:2*env.n+1],live&~dispatch)
            action=p.multinomial(1).squeeze(1)
            v,_,_=env.next_vehicle()
            v=torch.where(action<env.n,env.k,v)
            for i in live.nonzero().flatten().tolist():
                a=int(action[i])
                if a==2*env.n+1:continue
                dest=a+1 if a<env.n else a-env.n
                if dest:routes[i][int(v[i])].append(dest)
            env.step(action,validate=True)
        else:raise RuntimeError('Step bound')
        metrics={k:v.tolist() for k,v in env.metrics().items()}
        # Canonicalize vehicle labels so dispatch order alone is not route diversity.
        signatures=[tuple(sorted(tuple(r) for r in fleet if r)) for fleet in routes]
        counts=Counter(signatures)
        summary={}
        for key,values in stats.items():
            summary[key]=dict(decisions=len(values),mean_max_probability=sum(x[0] for x in values)/len(values) if values else None,
                fraction_max_above_099=sum(x[0]>.99 for x in values)/len(values) if values else None,
                mean_entropy=sum(x[1] for x in values)/len(values) if values else None,
                mean_normalized_entropy=sum(x[2] for x in values)/len(values) if values else None)
        r=torch.tensor(metrics['reward'])
        row=dict(name=split[index]['name'],unique_routes=len(counts),most_common_route_count=counts.most_common(1)[0][1],
            vehicle_histogram=dict(Counter(metrics['vehicles'])),reward_mean=float(r.mean()),reward_std=float(r.std()),
            reward_min=float(r.min()),reward_max=float(r.max()),unique_rewards=len(set(metrics['reward'])),
            choices=summary,metrics=metrics,routes=routes)
        output.append(row)
        print(json.dumps({k:v for k,v in row.items() if k not in ['metrics','routes']}),flush=True)
    write(ROOT/'analysis/v9_route_diversity/report.json',dict(checkpoint=500,device='CPU FP32',rollouts=100,
        note='Three TRAIN scenarios; branching states only; NEW entropy conditional on NEW; no optimizer or live model changes.',results=output))


if __name__=='__main__':main()
