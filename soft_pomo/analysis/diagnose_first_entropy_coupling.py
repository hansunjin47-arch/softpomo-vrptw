"""Frozen checkpoint gradient diagnostics, no optimizer steps."""
import json
import argparse
import math
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.concurrent_fleet_stop_init.policy import FleetPolicy
from experiments.concurrent_fleet_v9.env import TensorFleetEnv
from experiments.concurrent_fleet_v3.run import SOURCE,load
from experiments.concurrent_fleet_exploration.objective import first_customer_entropy,dispatch_entropy,entropy_beta
from experiments.concurrent_fleet_pair.io import write


def dot(a,b):
    return sum(float((x.double()*y.double()).sum()) for x,y in zip(a,b) if x is not None and y is not None)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path)
    ap.add_argument('--epoch',type=int,default=50)
    ap.add_argument('--beta',type=float,default=1.)
    ap.add_argument('--output',type=Path,default=ROOT/'analysis/first_entropy_coupling/report.json')
    args=ap.parse_args()
    torch.set_num_threads(2)
    bases=dict(old=ROOT/'server_export/be100_received_20261022_100545/runs/vehicle100',new=ROOT/'result/v9_firstentropy100_late10_20261022')
    if args.run:bases=dict(new=args.run)
    cfg=json.loads((bases['new']/'protocol.json').read_text())['configs']['10']
    rows=json.loads((SOURCE/'split.json').read_text())['train'];out=[]
    for label,base in bases.items():
        for idx in [0,3,10]:
            saved=torch.load(base/f'late10/checkpoint-{args.epoch}.pt',map_location='cpu',weights_only=False)
            model=FleetPolicy(**cfg['model']).train();model.load_state_dict(saved['model_state_dict'])
            torch.manual_seed(20261022+idx)
            inst,public=load(rows[idx]);env=TensorFleetEnv(inst,100,cfg['environment'],device='cpu')
            starts,encoded=model.encode_events(inst,public)
            logs=[];hs=[torch.zeros(100),torch.zeros(100)];cs=[torch.zeros(100),torch.zeros(100)]
            stop=[];odds=[];direct=[]
            first_sets=torch.zeros(100,env.n,dtype=torch.bool)
            first_orders=[[] for _ in range(100)]
            for step in range(600):
                if bool(env.done.all()):break
                frame=env.frame();live=~env.done;p=model.probabilities(frame,starts,encoded,env)
                for i,fn in enumerate([dispatch_entropy,first_customer_entropy]):
                    h,v=fn(p,frame['mask'],env.dispatch_open,live,env.n);hs[i]=hs[i]+h;cs[i]=cs[i]+v.float()
                both=live & env.dispatch_open & frame['mask'][:,-1] & frame['mask'][:,:env.n].any(1)
                if bool(both.any()):
                    ps=p[both,-1];pn=p[both,:env.n].sum(1)
                    stop.append(ps.detach());odds.append(ps.clamp_min(1e-30).log()-pn.clamp_min(1e-30).log())
                    q=(p[both,:env.n]/pn[:,None]).detach();hq=-(q*q.clamp_min(1e-30).log()).sum(1)
                    # Independent NEW-logit entropy ascent: derivative of logsumexp(NEW).
                    direct.append((q.square()*(-q.clamp_min(1e-30).log()-hq[:,None])).sum(1))
                with torch.no_grad():a=p.multinomial(1).squeeze(1)
                chosen=live & env.dispatch_open & (a<env.n)
                first_sets[env.ids[chosen],a[chosen]]=True
                for ri in env.ids[chosen].tolist():first_orders[ri].append(int(a[ri]))
                logs.append(torch.where(live,p[env.ids,a],1.).log());env.step(a)
            assert bool(env.done.all())
            r=env.metrics()['reward'].float();adv=r-r[0];adv-=adv.mean()
            h=[(a/b.clamp_min(1)).mean() for a,b in zip(hs,cs)]
            losses=[-(adv*torch.stack(logs).sum(0)).mean(),-entropy_beta(args.epoch+1,1,500)*h[0],-args.beta*h[1]]
            params=list(model.parameters());grads=[torch.autograd.grad(x,params,retain_graph=True,allow_unused=True) for x in losses]
            go=torch.autograd.grad(torch.cat(odds).mean(),params,allow_unused=True)
            norms=[dot(g,g)**.5 for g in grads]
            result=dict(model=label,name=rows[idx]['name'],checkpoint=args.epoch,first_beta=args.beta,stop_bias=float(model.stop_bias.detach()),
                unique_first_sets=int(torch.unique(first_sets,dim=0).shape[0]),
                unique_first_orders=len(set(tuple(x) for x in first_orders)),
                fixed_set_uniform_order_entropy=math.lgamma(26)/25,
                reward=float(r.mean()),vehicles=float(env.k.float().mean()),entropy=[float(x.detach()) for x in h],
                mean_stop_probability=float(torch.cat(stop).mean()),
                norms=dict(zip(['policy','stop','first'],norms)),
                first_policy_cosine=dot(grads[0],grads[2])/max(norms[0]*norms[2],1e-30),
                first_stop_cosine=dot(grads[1],grads[2])/max(norms[1]*norms[2],1e-30),
                stop_logodds_direction_per_unit_gradient={k:-dot(go,g)/max(n,1e-30) for k,g,n in zip(['policy','stop','first'],grads,norms)},
                independent_logits_NEW_logmass_direction=float(torch.cat(direct).mean()))
            out.append(result);print(json.dumps(result),flush=True)
    write(args.output,dict(results=out,
        scope='Frozen checkpoints, CPU FP32 batch1 rollout100, 3 train cases, next-epoch STOP beta; no optimizer updates',
        caution='Directional derivatives are local Euclidean gradient descent, NOT Adam updates or causal training attribution. States differ between policies.'))


if __name__=='__main__':main()
