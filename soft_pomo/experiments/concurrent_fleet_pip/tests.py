"""CPU semantic tests; deferred GPU rollout/backward preflight."""
import argparse
import gc
import json
from types import SimpleNamespace
import torch
from experiments.concurrent_fleet_pip.policy import route_risk, FleetPolicy
from experiments.concurrent_fleet_stop_init.policy import FleetPolicy as BasePolicy
from experiments.concurrent_fleet_v9.env import TensorFleetEnv
from experiments.concurrent_fleet_v3.run import ROOT, SOURCE, load


def toy():
    e = SimpleNamespace(count=1, n=3, v=2, device=torch.device('cpu'), horizon=100.)
    e.time=torch.tensor([0.]); e.dispatch_open=torch.tensor([False]); e.done=torch.tensor([False])
    e.base=torch.tensor([[0.,1.,2.,3.],[1.,0.,1.,2.],[2.,1.,0.,3.],[3.,2.,3.,0.]])
    e.status=torch.tensor([[3,0]]); e.node=torch.tensor([[1,0]]); e.target=e.node.clone()
    e.departure=torch.zeros(1,2); e.available=torch.zeros(1,2)
    e.demand=torch.tensor([0.,.1,.1,.1]); e.load=torch.ones(1,2)
    e.open=torch.zeros(4); e.close=torch.tensor([100.,100.,100.,2.]); e.service=torch.zeros(4)
    e.customer=torch.tensor([[2,2,0,0]])
    e.next_vehicle=lambda: (torch.tensor([0]),torch.tensor([True]),torch.tensor([False]))
    frame=dict(node=torch.tensor([1]), load=torch.tensor([1.]), mask=torch.ones(1,8,dtype=torch.bool))
    return e,frame


def semantics():
    e,f=toy(); risk=route_risk(e,f)
    assert risk[0,2] == 2 and risk[0,3] == 0, risk
    e.status[0,1]=3; e.node[0,1]=3; e.target[0,1]=3
    assert route_risk(e,f)[0,2] == 0, 'Other vehicle must rescue urgent customer'
    e.load[0,1]=0
    assert route_risk(e,f)[0,2] == 2, 'Capacity-ineligible vehicle cannot rescue'
    e.load[0,1]=1; e.status[0,1]=1
    a=route_risk(e,f)
    e.arrival=torch.full((1,2),9999.); e.true_events=[('secret',)]
    e.available[:]=9999.
    assert torch.equal(a,route_risk(e,f)), 'Hidden ETA/event accessed'
    e.dispatch_open[:]=True
    assert route_risk(e,f).sum()==0
    e.dispatch_open[:]=False; e.customer[:]=2
    assert torch.isfinite(route_risk(e,f)).all() and route_risk(e,f).sum()==0
    print('PASS toy: lookahead, rescue, capacity, hidden-state invariance, dispatch, empty')


def rollout(device):
    torch.set_num_threads(1); torch.manual_seed(42)
    cfg=json.loads((ROOT/'result/rc101_learned_first_train20_20261017/late10/config.json').read_text())
    assert cfg['model']['logit_clipping']==10
    split=json.loads((SOURCE/'split.json').read_text())
    inst,public=load(split['train'][0]); ep=cfg['environment']
    model=FleetPolicy(**cfg['model']).to(device)
    # CPU calibration accumulates diagnostics before trainer moves model to CUDA.
    model.risk_stats=torch.zeros(3,device='cpu')
    baseline=BasePolicy(**cfg['model']).to(device); baseline.load_state_dict(model.state_dict())
    env=TensorFleetEnv(inst,4,ep,device=device)
    starts,enc=model.encode_events(inst,public)
    bs,be=baseline.encode_events(inst,public)
    logps=[]; changed=0
    for step in range(300):
        if bool(env.done.all()): break
        frame=env.frame(); before=env.customer.clone()
        p=model.probabilities(frame,starts,enc,env)
        with torch.no_grad():
            q=baseline.probabilities(frame,bs,be,env)
        assert torch.equal(before,env.customer)
        assert torch.isfinite(p).all() and torch.allclose(p.sum(1),torch.ones(4,device=device),atol=1e-5)
        assert (p[~frame['mask']] == 0).all() or bool(env.done.any())
        dispatch=env.dispatch_open & ~env.done
        assert torch.equal(p[dispatch],q[dispatch]), 'Dispatch changed'
        changed+=int(((p-q).abs().sum(1)>1e-6).sum())
        a=p.detach().multinomial(1).squeeze(1)
        logps.append(torch.where(env.done,torch.ones(4,device=device),p[env.ids,a]).log())
        env.step(a,validate=True)
    assert bool(env.done.all()) and (env.metrics()['unserved']==0).all()
    reward=env.metrics()['reward']; advantage=reward-reward.mean()
    loss=-(advantage*torch.stack(logps).sum(0)).mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert changed>0, 'Guidance never affected probabilities'
    print('PASS real rollout and backward:',device,'steps',step,'changed states',changed)
    if device=='cuda':
        from experiments.concurrent_fleet_stop_init.train import train_batch
        opt=torch.optim.Adam(model.parameters(),lr=1e-4)
        scaler=torch.amp.GradScaler('cuda',init_scale=1.)
        summary,_=train_batch(model,opt,scaler,inst,public,ep,2,4,amp=True)
        assert not summary['optimizer_step_skipped']
        print('PASS AMP preflight',summary)
        del env, enc, be, logps, loss, p, q, baseline
        gc.collect(); torch.cuda.empty_cache()
        summary,_=train_batch(model,opt,scaler,inst,public,ep,128,100,amp=True)
        assert not summary['optimizer_step_skipped']
        print('PASS full batch128 x rollout100 AMP preflight',summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--gpu',action='store_true')
    args=parser.parse_args()
    semantics(); rollout('cuda' if args.gpu else 'cpu')
