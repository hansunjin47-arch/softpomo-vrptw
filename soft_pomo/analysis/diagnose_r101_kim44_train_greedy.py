"""Evaluate the final Kim44 on train20 using frozen training confidences, no API."""
import statistics
import torch
from analysis.evaluate_r101_mask_kim_20261109 import BUNDLE, RECEIVED
from experiments.concurrent_fleet_pair_kim.run import GuidedEnv, GuidedPolicy
from experiments.concurrent_fleet_pair import train
from experiments.concurrent_fleet_v3 import run as data_module
from experiments.concurrent_fleet_bias_guided.support import ROOT, read, load
from experiments.concurrent_fleet_v9_guided.prompts import parse_response
from experiments.concurrent_fleet_pair.io import write


def main():
    torch.set_num_threads(1)
    data_module.SOURCE=BUNDLE/'dataset'
    run=RECEIVED/'runs/seed44'
    cfg=read(run/'protocol.json')['configs']['5']
    groups=read(BUNDLE/'artifacts/groups.json')
    class CachedEnv(GuidedEnv):
        def __init__(self,inst,count,params,device='cuda'):
            cache=read(BUNDLE/'artifacts/train_cache'/(inst['name']+'.json'))
            super().__init__(inst,count,params,device,groups=groups,scores=parse_response(cache['initial'],groups))
            if cache['post_disclosure'] is not None:
                t=cache['revealed_at_min']
                self.update_event_confidence(parse_response(cache['post_disclosure'],groups),revealed_at_min=t,effective_time_min=t)
    train.TensorFleetEnv=CachedEnv
    model=GuidedPolicy(**cfg['model']).cuda().eval()
    model.load_state_dict(torch.load(run/'late5/checkpoint-1000.pt',map_location='cpu',weights_only=False)['model_state_dict'])
    results=[]
    for row in read(run/'split.json')['train']:
        inst,public=load(row)
        env,_,_=train.collect(model,inst,public,cfg['environment'],1,amp=True,greedy=True)
        metrics={k:float(v[0]) for k,v in env.metrics().items()}
        results.append(dict(name=row['name'],**metrics))
        print(row['name'],metrics['reward'],metrics['vehicles'],metrics['total_late'],flush=True)
    summary={k:statistics.mean(r[k] for r in results) for k in ['reward','vehicles','travel_time','total_late','unserved']}
    write(ROOT/'analysis/r101_kim44_train_greedy_20261109.json',dict(summary=summary,results=results,amp=True,cache='offline train cache',seed=44,epoch=1000))
    print('SUMMARY',summary,flush=True)


if __name__=='__main__':main()
