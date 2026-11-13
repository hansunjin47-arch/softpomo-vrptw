"""CPU smoke tests; no changes to the active training process."""
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.concurrent_fleet_v3.run import SOURCE,load
from experiments.concurrent_fleet_v9.policy import FleetPolicy as Base
from experiments.concurrent_fleet_stop_init.policy import FleetPolicy,calibrate
from experiments.concurrent_fleet_stop_init import train
from experiments.concurrent_fleet_v9.env import TensorFleetEnv
from experiments.concurrent_fleet_pair.io import write
from train_vrptw import _set_seed


def main():
    torch.set_num_threads(2)
    cfg=json.loads((ROOT/'result/concurrent_fleet_v9_train20_late10_vehicle700_20261019/protocol.json').read_text())['configs']['10']
    split=json.loads((SOURCE/'split.json').read_text())
    _set_seed(42);base=Base(**cfg['model'])
    _set_seed(42);model=FleetPolicy(**cfg['model'])
    for k,v in base.state_dict().items():assert torch.equal(v,model.state_dict()[k]),k
    cases=[load(r) for r in split['train']]
    rng=torch.get_rng_state().clone()
    report=calibrate(model,cases,cfg['environment'])
    assert torch.equal(rng,torch.get_rng_state())
    assert abs(report['mean_after']-.15)<1e-6
    inst,public=cases[0]
    for net in [base,model]:
        env=TensorFleetEnv(inst,100,cfg['environment'],device='cpu')
        starts,enc=net.encode_events(inst,public)
        probs=net.probabilities(env.frame(),starts,enc,env)
        assert torch.equal(probs[:,-1],torch.zeros(100))
    # Exercise the actual full train_batch on CPU with AMP disabled.
    train.TensorFleetEnv=lambda inst,count,params:TensorFleetEnv(inst,count,params,device='cpu')
    opt=torch.optim.Adam(model.parameters(),**cfg['optimizer']['optimizer'])
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    before=model.stop_bias.detach().clone()
    summary,_=train.train_batch(model,opt,scaler,inst,public,cfg['environment'],1,100,amp=False,beta=0.)
    assert not summary['optimizer_step_skipped']
    assert summary['entropy_bonus']==0
    assert torch.isfinite(model.stop_bias.grad)
    assert not torch.equal(before,model.stop_bias.detach())
    restored=FleetPolicy(**cfg['model']);restored.load_state_dict(model.state_dict())
    assert torch.equal(restored.stop_bias,model.stop_bias)
    report['smoke_update']=summary
    write(ROOT/'analysis/stop_init_validation/report.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
