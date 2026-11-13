"""Phase assignment and full-batch AMP regression checks."""
import json
import torch
from experiments.concurrent_fleet_split_reward.train import phase_loss,centered,train_batch
from experiments.concurrent_fleet_stop_init.policy import FleetPolicy
from experiments.concurrent_fleet_v3.run import ROOT,SOURCE,load


def main():
    torch.set_num_threads(1);torch.manual_seed(42)
    reward=torch.tensor([[-3.,-5.,-8.]],dtype=torch.float64)
    logp=torch.randn(1,3,4,dtype=torch.float64,requires_grad=True)
    phases=torch.tensor([[[True,True,False,False]]*3])
    for cost in [torch.ones_like(reward)*2,torch.tensor([[1.,2.,3.]],dtype=torch.float64)]:
        dl,rl,a,ar=phase_loss(reward,cost,logp,phases)
        assert torch.allclose(ar,centered(reward+cost))
        g=torch.autograd.grad(dl+rl,logp,retain_graph=True)[0]
        expected=-torch.where(phases,a[:,:,None],ar[:,:,None])/3
        assert torch.allclose(g,expected)
        if torch.equal(cost,cost[:,:1].expand_as(cost)):
            old=-(centered(reward)*logp.sum(2)).mean()
            assert torch.allclose(dl+rl,old)
            assert torch.allclose(g,torch.autograd.grad(old,logp,retain_graph=True)[0])
    print('PASS loss and gradient: constant K equivalence, mixed K phase attribution',flush=True)
    cfg=json.loads((ROOT/'result/rc101_learned_first_train20_20261017/late10/config.json').read_text())
    assert cfg['model']['logit_clipping']==10 and cfg['trainer']['seed']==42 and cfg['trainer']['epochs']==1000
    split=json.loads((SOURCE/'split.json').read_text())
    inst,public=load(split['train'][0]); ep=cfg['environment']
    ep['vehicle_penalty']=300*ep['time_penalty']*inst['n_customers']/(inst['T']*ep['TT_max'])
    model=FleetPolicy(**cfg['model']).cuda()
    opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    scaler=torch.amp.GradScaler('cuda',init_scale=1.)
    summary,_=train_batch(model,opt,scaler,inst,public,ep,128,100,beta=1.)
    assert not summary['optimizer_step_skipped']
    assert abs(summary['loss']-(summary['dispatch_policy_loss']+summary['route_policy_loss']-summary['entropy_bonus']))<1e-4
    print('PASS full batch128 x rollout100 AMP update',json.dumps(summary),flush=True)


if __name__=='__main__':main()
