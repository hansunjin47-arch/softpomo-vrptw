"""Matched CPU sampled-rollout probes; no training or test-set selection."""
import json
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.concurrent_fleet_stop_init.policy import FleetPolicy
from experiments.concurrent_fleet_v9.env import TensorFleetEnv
from experiments.concurrent_fleet_exploration.objective import first_customer_entropy
from experiments.concurrent_fleet_v3.run import SOURCE,load
from experiments.concurrent_fleet_pair.io import write


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    bases=dict(old=ROOT/'server_export/be100_received_20261022_100545/runs/vehicle100',
               new=ROOT/'result/v9_firstentropy100_late10_20261022')
    configs={k:json.loads((v/'protocol.json').read_text())['configs']['10'] for k,v in bases.items()}
    assert configs['old']==configs['new']
    cfg=configs['new'];rows=json.loads((SOURCE/'split.json').read_text())['train'];output=[]
    for epoch in [10,20,50]:
        if not all((b/f'late10/checkpoint-{epoch}.pt').exists() for b in bases.values()):continue
        for label,base in bases.items():
            checkpoint=torch.load(base/f'late10/checkpoint-{epoch}.pt',map_location='cpu',weights_only=False)
            assert checkpoint['epoch']==epoch
            model=FleetPolicy(**cfg['model']).eval();model.load_state_dict(checkpoint['model_state_dict'])
            for idx in [0,3,10]:
                torch.manual_seed(20261022+idx)
                inst,public=load(rows[idx]);env=TensorFleetEnv(inst,100,cfg['environment'],device='cpu')
                starts,encoded=model.encode_events(inst,public)
                sets=torch.zeros(100,env.n,dtype=torch.bool);hs=torch.zeros(100);cs=torch.zeros(100)
                for step in range(600):
                    if bool(env.done.all()):break
                    frame=env.frame();live=~env.done;p=model.probabilities(frame,starts,encoded,env)
                    h,v=first_customer_entropy(p,frame['mask'],env.dispatch_open,live,env.n)
                    hs+=h;cs+=v.float();a=p.multinomial(1).squeeze(1)
                    chosen=live & env.dispatch_open & (a<env.n)
                    sets[env.ids[chosen],a[chosen]]=True
                    env.step(a)
                assert bool(env.done.all())
                unique,counts=torch.unique(sets,dim=0,return_counts=True)
                item=dict(epoch=epoch,model=label,name=rows[idx]['name'],unique_first_sets=len(unique),
                          most_common_set_fraction=float(counts.max())/100,
                          first_entropy=float((hs/cs.clamp_min(1)).mean()),
                          metrics={k:float(v.double().mean()) for k,v in env.metrics().items()})
                output.append(item);print(json.dumps(item),flush=True)
    write(ROOT/'analysis/first_entropy100_comparison/report.json',dict(
        scope='3 TRAIN cases x100 stochastic rollouts; CPU FP32 eval mode, matching seeds; no updates',results=output))


if __name__=='__main__':main()
