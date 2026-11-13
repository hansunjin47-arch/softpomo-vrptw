"""CPU-only adapter, checkpoint and full-trajectory backward checks; no paid calls."""
import json
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.concurrent_fleet_bias_guided.support import PURE,OUT,read,load,cached_env,GuidedBiasPolicy
from experiments.concurrent_fleet_stop_init.policy import FleetPolicy
from experiments.concurrent_fleet_stop_init.train import train_batch
from experiments.concurrent_fleet_v3.run import SOURCE
from experiments.concurrent_fleet_exploration.objective import entropy_beta
from experiments.concurrent_fleet_pair.io import write
from experiments.concurrent_fleet_v9_guided.online import evaluate_instance


def main():
    torch.set_num_threads(1)
    cfg=read(PURE/'protocol.json')['configs']['10']
    split=read(SOURCE/'split.json')
    saved=torch.load(PURE/'late10/checkpoint-500.pt',map_location='cpu',weights_only=False)
    base=FleetPolicy(**cfg['model']).eval()
    model=GuidedBiasPolicy(**cfg['model']).eval()
    base.load_state_dict(saved['model_state_dict'],strict=True)
    model.load_state_dict(saved['model_state_dict'],strict=True)
    assert [n for n,_ in base.named_parameters()]==[n for n,_ in model.named_parameters()]
    opt=torch.optim.Adam(model.parameters(),**cfg['optimizer']['optimizer'])
    opt.load_state_dict(saved['optimizer_state_dict'])
    assert saved['state']['epoch']==501 and entropy_beta(501,1,500)==0
    Env=cached_env(OUT/'kim/artifacts')
    inst,public=load(split['train'][0])
    env=Env(inst,2,cfg['environment'],device='cpu')
    starts,enc=model.encode_events(inst,public)
    base_starts,base_enc=base.encode_events(inst,public)
    dispatch_checks=0;routing_checks=0;nonzero=False
    with torch.no_grad():
        for step in range(300):
            if bool(env.done.all()):break
            frame=env.frame()
            # End-to-end probabilities with zero guidance must match the pure bias model.
            env.guidance.beta=0
            p=model.probabilities(frame,starts,enc,env)
            q=base.probabilities(frame,base_starts,base_enc,env)
            torch.testing.assert_close(p,q,rtol=0,atol=0)
            env.guidance.beta=5
            guided=model.probabilities(frame,starts,enc,env)
            if bool(env.dispatch_open.all()):
                torch.testing.assert_close(p,guided,rtol=0,atol=0);dispatch_checks+=1
            else:
                routing_checks+=1
                nonzero=nonzero or not torch.equal(p,guided)
            env.step(guided.argmax(1),validate=True)
    assert dispatch_checks>0 and routing_checks>0 and nonzero
    calls=[]
    def fake_client(request):
        context=json.loads(request['messages'][1]['content'])
        calls.append(context)
        return {'confidence':{str(n):.5 for n in context['selectable_customers']}}
    accident=next(r for r in split['train'] if '_ACC_' in r['name'])
    acc_inst,acc_public=load(accident)
    online=evaluate_instance(model,acc_inst,cfg['environment'],read(OUT/'kim/artifacts/groups.json'),acc_public,
        template=read(OUT/'kim/artifacts/request_template.json'),client=fake_client,amp=False)
    assert len(calls)==2 and online['metrics']['unserved']==0
    # A fresh disposable model validates the actual shared no-std/entropy loss with guidance.
    model=GuidedBiasPolicy(**cfg['model']).train()
    model.stop_bias.data.fill_(4.6639766693)
    opt=torch.optim.Adam(model.parameters(),**cfg['optimizer']['optimizer'])
    class CPUEnv(Env):
        def __init__(self,inst,count,params):super().__init__(inst,count,params,device='cpu')
    summary,_=train_batch(model,opt,torch.amp.GradScaler('cuda',enabled=False),inst,public,
        cfg['environment'],1,4,amp=False,beta=1.,env_type=CPUEnv)
    assert not summary['optimizer_step_skipped']
    assert abs(summary['loss']-(summary['policy_loss']-summary['entropy_bonus']))<1e-3
    assert model.stop_bias.grad is not None and bool(torch.isfinite(model.stop_bias.grad))
    report=dict(checkpoint_strict_load=True,optimizer_load=True,param_order_identical=True,
        phase2_starts_epoch501_beta0=True,zero_guidance_parity=True,dispatch_unaffected=True,
        routing_guidance_nonzero=True,cpu_backward_finite=True,paid_calls=0,online_mock_calls=len(calls),
        summary=summary)
    write(ROOT/'analysis/bias_guided_validation/report.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
