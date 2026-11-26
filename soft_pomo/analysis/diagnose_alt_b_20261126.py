"""Offline Train20 diagnosis of B using public observations and the training caches."""
import argparse

import json
import statistics as st
from pathlib import Path
import torch
from experiments.fleet_alt_rl.train import Replay, ActionValue, setup, data_module, load, read, q_values
from collections import defaultdict
from experiments.fleet_alt_rl.state import action_bias, observe


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('out',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(2)
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    _,src,_,cfg,env_cls=setup(5, portable=True)
    data_module.SOURCE=src
    net=ActionValue(**cfg['model']).eval()
    net.load_state_dict(ck['model'])


    items={r['name']:load(r) for r in read(src/'split.json')['train']}
    if ck.get('format') == 'b_diagnostics_v1':
        epoch, summary = ck['epoch'], ck['replay_summary']
    else:
        rep = ck['replay'].rows
        epoch = ck['state']['epoch']
        summary = dict(count=len(rep), terminal=sum(r[-1] for r in rep), names=len({r[0] for r in rep}))
    output = dict(epoch=epoch, replay=summary, modes={})
    args.out.parent.mkdir(exist_ok=True, parents=True)
    modes=['train','soft_only','sharper','greedy_guided','greedy_q']
    for mode in modes:
        torch.manual_seed(46)
        metrics=[]; diag=defaultdict(list)
        for inst,public in items.values():
            env=env_cls(inst,4 if mode in ('train','soft_only','sharper') else 1,cfg['environment'],device='cpu')
            geo=tuple(v/env.horizon for v in (env.base,env.open,env.close))
            enc=net.encode_events(inst,public)
            for _ in range(600):
                if env.done.all():break
                s=observe(env,True);q=q_values(net,(inst,public,geo),s,enc)
                mask=s['mask'].clone();mask[:,0]|=~mask.any(1)
                guide=action_bias(s,5)
                guided=(q+guide).masked_fill(~mask,-torch.inf)
                pure=q.masked_fill(~mask,-torch.inf)
                greedy=guided.argmax(1)
                if mode in ('train','soft_only','sharper'):
                    p=(guided / (0.25 if mode=='sharper' else 1.0)).softmax(1)
                    if mode=='train':p=.95*p+.05*mask/mask.sum(1,keepdim=True)
                    act=p.multinomial(1).squeeze(1)
                    for phase,rows in [('dispatch',s['dispatch']),('route',~s['dispatch'])]:
                        eligible=rows & ~env.done & (mask.sum(1)>1)
                        if eligible.any():
                            diag[phase+'_top1'].extend(p.amax(1)[eligible].tolist())
                            diag[phase+'_feasible'].extend(mask.sum(1)[eligible].tolist())
                            diag[phase+'_flip'].extend((greedy!=pure.argmax(1))[eligible].float().tolist())
                            gap=pure.topk(2).values
                            diag[phase+'_qgap'].extend((gap[:,0]-gap[:,1])[eligible].tolist())
                else:act=greedy if mode=='greedy_guided' else pure.argmax(1)
                env.step(act,validate=True)
            assert env.done.all()
            m=env.metrics()
            metrics.append({k:m[k].float().mean().item() for k in ('reward','vehicles','travel_time','total_late','unserved')})
        output['modes'][mode]={'metrics':{k:st.mean(r[k] for r in metrics) for k in metrics[0]},
                               'decisions':{k:st.mean(v) for k,v in diag.items()}}
        args.out.write_text(json.dumps(output,indent=2))
        print(output['epoch'],mode,output['modes'][mode]['metrics'],flush=True)

if __name__=='__main__':main()

