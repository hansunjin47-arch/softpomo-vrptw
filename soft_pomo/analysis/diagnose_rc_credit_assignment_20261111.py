"""Frozen PG dispatch/routing decomposition preserving original reduction."""
import os
import statistics
import torch
from analysis.diagnose_rc_family_gradients_20261111 import ROOT,PILOT,read,digest,compare
from experiments.concurrent_fleet_pair.policy import FleetPolicy
from experiments.concurrent_fleet_pair.env import TensorFleetEnv
from experiments.concurrent_fleet_pair.io import write
from experiments.concurrent_fleet_v3.run import load
from train_vrptw import _set_seed

STRONG=ROOT/'result/rc_family_entropy_strong_20261111'
OUT=ROOT/'result/rc_credit_assignment_20261111'


def one(model,row,cfg,epoch):
    _set_seed(20261111);batch=8;rollouts=100
    model.new_customer_epsilon=.01+.04*max(0.,1-(epoch-1)/499)
    inst,public=load(row);env=TensorFleetEnv(inst,batch*rollouts,cfg['environment'])
    starts,enc=model.encode_events(inst,public,batch_size=batch)
    dispatch=[];routing=[];counts=[0,0]
    with torch.autocast('cuda',enabled=True):
        for step in range(600):
            if bool(env.done.all()):break
            live=~env.done;opened=env.dispatch_open.clone()
            p=model.probabilities(env.frame(),starts,enc,env)
            with torch.no_grad():action=p.multinomial(1).squeeze(1)
            logp=torch.where(live,p[env.ids,action],1.).float().log().reshape(batch,rollouts)
            gate=(live & opened).reshape(batch,rollouts)
            dispatch.append(torch.where(gate,logp,0.));routing.append(torch.where(~gate,logp,0.))
            counts[0]+=int((live & opened).sum());counts[1]+=int((live & ~opened).sum())
            env.step(action,validate=True)
        else:raise RuntimeError('Step limit')
    reward=env.metrics()['reward'].float().reshape(batch,rollouts)
    shift=reward-reward[:,:1];adv=shift-shift.mean(1,keepdim=True)
    adv=adv/adv.std(1,keepdim=True).clamp(min=1e-6)
    dl=torch.stack(dispatch).sum(0);rl=torch.stack(routing).sum(0)
    full=-(adv*(dl+rl)).mean();d=-(adv*dl).mean();r=-(adv*rl).mean()
    assert torch.allclose(full,d+r,atol=2e-5,rtol=1e-5)
    named=list(model.named_parameters());names=[n for n,p in named];params=[p for n,p in named]
    gf=torch.autograd.grad(full,params,retain_graph=True,allow_unused=True)
    gd=torch.autograd.grad(d,params,retain_graph=True,allow_unused=True)
    gr=torch.autograd.grad(r,params,allow_unused=True)
    error=denom=0.
    for f,a,b in zip(gf,gd,gr):
        if f is None:continue
        assert torch.isfinite(f).all()
        total=(a if a is not None else 0)+(b if b is not None else 0)
        error+=float((f.double()-total.double()).square().sum());denom+=float(f.double().square().sum())
    rel=(error/max(denom,1e-30))**.5
    assert rel<.005,('AMP gradient decomposition residual exceeds0.5%',rel)
    groups={}
    for label,prefix in (('global',None),('encoder','encoder.'),('decoder','decoder.')):
        dr=compare(gd,gr,names,prefix);fd=compare(gf,gd,names,prefix)
        groups[label]=dict(dispatch_norm=dr['pg_norm'],routing_norm=dr['aux_norm'],cosine=dr['cosine'],
            full_norm=fd['pg_norm'],dispatch_over_full=fd['ratio'],routing_over_full=dr['aux_norm']/fd['pg_norm'])
    return dict(loss_full=float(full.detach()),loss_dispatch=float(d.detach()),loss_routing=float(r.detach()),
        loss_residual=float((full-d-r).detach()),gradient_sum_relative_residual=rel,
        gradients=groups,dispatch_steps=counts[0],routing_steps=counts[1],batch=8,rollouts=100,
        reward_mean=float(reward.mean()))


def main():
    torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
    table=[]
    for label,folder in (('control',PILOT/'control'),('weak',PILOT/'treatment'),('strong',STRONG/'treatment')):
        rows=read(folder/'train20_epoch100.json')
        table.append(dict(arm=label,**{k:statistics.mean(r[k] for r in rows) for k in ('reward','vehicles','travel_time','total_late','unserved')}))
    write(OUT/'entropy_three_arm_conclusion.json',dict(rows=table,conclusion='Both preregistered entropy schedules missed control Train20 reward; stop this coefficient family.',scope='One seed100epoch Train20, no Test/generalization claim'))
    cfg=read(PILOT/'protocol.json')['config'];split=read(ROOT/'result/rc_pair_returnmask_v100_l5_s44_20261102/split.json')
    rows=[next(r for r in split['train'] if r['name']==n) for n in ('RC101_RAIN_P018','RC101_ACC_P041')]
    jobs=[(arm,epoch,folder/f'checkpoint-{epoch}.pt') for arm,folder in (('control',PILOT/'control'),('strong',STRONG/'treatment')) for epoch in (40,60)]
    hashes={str(p.relative_to(ROOT)):digest(p) for _,_,p in jobs}
    write(OUT/'protocol.json',dict(pid=os.getpid(),target=8,input_sha256=hashes,seed=20261111,optimizer_steps=0,
        reduction='Both terms retain original trajectory sum and batch/rollout mean. Same full-episode normalized advantage for dispatch and routing.',
        limits='AMP backward gradient sum checked relative tolerance0.5%; parameter sharing and cosine do not establish causal training interference.'))
    results=[]
    for arm,epoch,path in jobs:
        for row in rows:
            model=FleetPolicy(**cfg['model']).cuda().train()
            model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model_state_dict'])
            result=one(model,row,cfg,epoch);result.update(arm=arm,epoch=epoch,name=row['name'])
            write(OUT/f'{arm}_{epoch}_{row["name"]}.json',result);results.append(result)
            print(arm,epoch,row['name'],result['gradients']['global'],'residual',result['gradient_sum_relative_residual'],flush=True)
            del model;torch.cuda.empty_cache()
    assert all(digest(ROOT/p)==h for p,h in hashes.items())
    write(OUT/'summary.json',results);write(OUT/'status.json',dict(stage='complete',count=8,input_hashes_unchanged=True))


if __name__=='__main__':
    try:main()
    except Exception as error:
        write(OUT/'status.json',dict(stage='failed',error=repr(error)))
        raise
