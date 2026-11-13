"""Compare cached confidence distributions at identical Pure500 CPU states."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from experiments.concurrent_fleet_bias_guided.support import GuidedBiasPolicy,cached_env,read,load
from experiments.concurrent_fleet_bias_guided.kim300 import PURE,OUT as BASE
from experiments.concurrent_fleet_v3.run import SOURCE,write

OUT=ROOT/'analysis/phase2_probabilities300_20261023'


def summarize(parts):
    result={}
    for k,values in parts.items():
        x=torch.cat(values).double()
        result[k]=dict(count=x.numel(),mean=float(x.mean()),max=float(x.max()),
                       p50=float(x.quantile(.5)),p95=float(x.quantile(.95)))
    return result


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    cfg=read(PURE/'protocol.json')['configs']['10']
    model=GuidedBiasPolicy(**cfg['model']).eval()
    saved=torch.load(PURE/'late10/checkpoint-500.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(saved['model_state_dict'],strict=True)
    Env=cached_env(BASE/'phase2/artifacts')
    rows=read(SOURCE/'split.json')['train']
    sample_names={'RC101_RAIN_P018','RC101_RAIN_P017','RC101_ACC_P041'}
    report=dict(protocol='Pure500 CPU FP32; same states; OFF-policy greedy20 and sampled100 x 3 train scenarios; beta ON=5; no updates/API',runs=[])
    pools={}
    for mode in ['greedy','sampled']:
        pooled={}
        for row in rows:
            if mode=='sampled' and row['name'] not in sample_names:continue
            torch.manual_seed(42)
            inst,public=load(row);count=1 if mode=='greedy' else 100
            env=Env(inst,count,cfg['environment'],device='cpu')
            starts,encoded=model.encode_events(inst,public)
            parts={};all_routes=0;dispatch_max=0.;formula_error=0.
            for step in range(600):
                if bool(env.done.all()):break
                frame=env.frame();live=~env.done
                env.guidance.beta=0.;off=model.probabilities(frame,starts,encoded,env)
                env.guidance.beta=5.;on=model.probabilities(frame,starts,encoded,env)
                dispatch=live & env.dispatch_open
                if dispatch.any():dispatch_max=max(dispatch_max,float((on[dispatch]-off[dispatch]).abs().max()))
                route=live & ~env.dispatch_open
                all_routes+=int(route.sum())
                eligible=route & (frame['mask'].sum(1)>=2)
                if eligible.any():
                    p=off[eligible].double();q=on[eligible].double()
                    mask=frame['mask'][eligible]
                    bias=env.guidance.action_bias()[eligible].double()
                    predicted=torch.softmax(p.clamp_min(1e-300).log().masked_fill(~mask,-torch.inf)+bias,1)
                    formula_error=max(formula_error,float((predicted-q).abs().max()))
                    a=p.argmax(1);ids=torch.arange(p.size(0));top=p.topk(2,dim=1).values
                    values=dict(tv=.5*(q-p).abs().sum(1),
                        kl_off_on=(p*(p.clamp_min(1e-300).log()-q.clamp_min(1e-300).log())).sum(1),
                        top1_off=p[ids,a],top1_on_same_action=q[ids,a],
                        off_action_probability_change=q[ids,a]-p[ids,a],
                        argmax_flip=(a!=q.argmax(1)).double(),
                        logit_margin=(top[:,0].clamp_min(1e-300).log()-top[:,1].clamp_min(1e-300).log()),
                        bias_range=bias.masked_fill(~mask,-torch.inf).max(1).values-bias.masked_fill(~mask,torch.inf).min(1).values,
                        active_bias=((bias>0)&mask).any(1).double(),
                        entropy_off=-(p*p.clamp_min(1e-300).log()).sum(1),
                        entropy_on=-(q*q.clamp_min(1e-300).log()).sum(1))
                    for k,v in values.items():parts.setdefault(k,[]).append(v.cpu())
                action=off.argmax(1) if mode=='greedy' else torch.multinomial(off,1).squeeze(1)
                env.step(action)
            else:raise RuntimeError('Rollout bound exceeded')
            assert dispatch_max==0
            assert formula_error<1e-5,formula_error
            for k,v in parts.items():pooled.setdefault(k,[]).extend(v)
            report['runs'].append(dict(name=row['name'],mode=mode,rollouts=count,all_route_states=all_routes,
                dispatch_probability_difference=dispatch_max,formula_max_error=formula_error,statistics=summarize(parts)))
            print(mode,row['name'],summarize(parts)['tv'],flush=True)
            write(OUT/'report.json',report)
        pools[mode]=summarize(pooled)
    report['pooled']=pools
    assert all(torch.equal(v.cpu(),saved['model_state_dict'][k]) for k,v in model.state_dict().items())
    write(OUT/'report.json',report)
    write(OUT/'complete.json',dict(optimizer_updates=0,paid_calls=0,unchanged_parameters=True))


if __name__=='__main__':main()
