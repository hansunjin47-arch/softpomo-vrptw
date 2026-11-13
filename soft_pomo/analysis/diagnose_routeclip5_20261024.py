"""Read-only curve and frozen-policy diagnosis for the completed C5 run."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from experiments.concurrent_fleet_bias_guided.routeclip5 import RouteClip5Policy
from experiments.concurrent_fleet_stop_init.policy import FleetPolicy
from experiments.concurrent_fleet_v9.env import TensorFleetEnv
from experiments.concurrent_fleet_v3.run import SOURCE,load,write

C5=ROOT/'result/v9_routeclip5_300_late10_20261023/pure'
C10=ROOT/'server_export/be_completed_received_20261022_091410/rc101_v9_bias_entropy_vehicle300_late10_20261021/runs/vehicle300'
OUT=ROOT/'analysis/routeclip5_diagnosis_20261024'


def read(p):return json.loads(p.read_text())


def curves(root):
    rows=[json.loads(l) for l in (root/'late10/training.jsonl').read_text().splitlines()]
    output=[]
    for lo,hi in [(1,5),(6,20),(41,50),(91,100),(191,200),(491,500),(791,800),(991,1000)]:
        selected=[r for r in rows if lo<=r['epoch']<=hi]
        weights=sum(r['batch_size'] for r in selected)
        keys=['reward','vehicles','travel_time','total_late','reward_group_std','dispatch_entropy','advantage_abs_mean']
        output.append(dict(epochs=[lo,hi],**{k:sum(r[k]*r['batch_size'] for r in selected)/weights for k in keys},
                           skips=sum(r['optimizer_step_skipped'] for r in selected)))
    return dict(windows=output,skips=sum(r['optimizer_step_skipped'] for r in rows),
                first_k_below20=next((r['epoch'] for r in rows if r['epoch']>5 and r['vehicles']<20),None))


@torch.no_grad()
def probes(root,epoch,cls):
    cfg=read(root/'protocol.json')['configs']['10']
    model=cls(**cfg['model']).eval()
    saved=torch.load(root/f'late10/checkpoint-{epoch}.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(saved['model_state_dict'])
    counterpart=(FleetPolicy if cls==RouteClip5Policy else RouteClip5Policy)(**cfg['model']).eval()
    counterpart.load_state_dict(saved['model_state_dict'])
    rows=read(SOURCE/'split.json')['train']
    rows=[r for r in rows if r['name'] in {'RC101_RAIN_P018','RC101_RAIN_P017','RC101_ACC_P041'}]
    results=[]
    for row in rows:
        inst,public=load(row)
        for mode in ['greedy','sampled']:
            torch.manual_seed(42)
            env=TensorFleetEnv(inst,1 if mode=='greedy' else 100,cfg['environment'],device='cpu')
            starts,enc=model.encode_events(inst,public)
            bs,be=counterpart.encode_events(inst,public)
            values={}
            for step in range(600):
                if env.done.all():break
                frame=env.frame();p=model.probabilities(frame,starts,enc,env)
                valid=(~env.done)&(~env.dispatch_open)&(frame['mask'].sum(1)>1)
                if valid.any():
                    q=counterpart.probabilities(frame,bs,be,env)[valid].double()
                    pv=p[valid].double(); top=pv.topk(2,dim=1).values
                    stats=dict(top1=top[:,0],entropy=-(pv*pv.clamp_min(1e-300).log()).sum(1),
                               margin=(top[:,0]/top[:,1].clamp_min(1e-300)).log(),
                               flipped=(pv.argmax(1)!=q.argmax(1)).double(),counterpart_top1=q.max(1).values)
                    for key,value in stats.items():values.setdefault(key,[]).append(value)
                action=p.argmax(1) if mode=='greedy' else p.multinomial(1).squeeze(1)
                env.step(action)
            else:raise RuntimeError('bound')
            metrics=env.metrics()
            results.append(dict(name=row['name'],mode=mode,
                **{k:float(torch.cat(v).mean()) for k,v in values.items()},
                **{k:float(metrics[k].double().mean()) for k in ['reward','vehicles','travel_time','total_late']},
                reward_std=float(metrics['reward'].std(unbiased=False))))
    print(root.name,epoch,results,flush=True)
    return results


def main():
    torch.set_num_threads(1)
    report=dict(scope='CPU FP32, train3 greedy and 100 samples; no optimizer or LLM',
                curves=dict(c5=curves(C5),c10=curves(C10)),probes={})
    write(OUT/'report.json',report)
    for name,root,ep,cls in [('c5_100',C5,100,RouteClip5Policy),('c5_500',C5,500,RouteClip5Policy),
                              ('c5_1000',C5,1000,RouteClip5Policy),('c10_500',C10,500,FleetPolicy)]:
        report['probes'][name]=probes(root,ep,cls)
        write(OUT/'report.json',report)
    print('COMPLETE',OUT,flush=True)


if __name__=='__main__':main()
