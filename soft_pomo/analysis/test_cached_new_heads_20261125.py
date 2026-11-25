"""Regression against pre-cache NEW bias, including updates and gradients."""
import copy,json
import torch
from experiments.concurrent_fleet_pair.policy import mix_new_customers
from experiments.concurrent_fleet_pair_kim.run import GuidedPolicy
from experiments.concurrent_fleet_pair.fleet_context import FleetContextPolicy
from experiments.concurrent_fleet_pair_kim.fleet_context import FleetContextKimPolicy
from experiments.concurrent_fleet_v3 import run as data
from experiments.concurrent_fleet_bias_guided.support import load,read
from analysis.train_kim_fleet_context_20261124 import setup,ROOT


class OriginalHeadPolicy(GuidedPolicy):
    new_head_beta=5.
    def probabilities(self,frame,starts,encodings,env):
        epsilon=self.new_customer_epsilon;self.new_customer_epsilon=0.
        try:base=super().probabilities(frame,starts,encodings,env)
        finally:self.new_customer_epsilon=epsilon
        def select(scores):
            return torch.stack([torch.tensor(g,device=base.device)[scores[k,g].argmax()]-1 for k,g in enumerate(env.guidance.groups)])
        heads=select(env.guidance.initial).expand(base.shape[0],-1)
        for effective,target,scores in env.guidance.updates:
            heads=torch.where((target & (env.time>=effective))[:,None],select(scores)[None],heads)
        active=env.dispatch_open & ~env.done
        bias=torch.zeros_like(base);bias.scatter_(1,heads,active[:,None].to(base.dtype).expand_as(heads)*self.new_head_beta)
        bias=bias.masked_fill(base==0,0.)
        weighted=base*(bias-bias.max(1,keepdim=True).values).exp()
        adjusted=weighted/weighted.sum(1,keepdim=True)
        base=torch.where(active[:,None],adjusted,base)
        return mix_new_customers(base,frame['mask'],env.n,epsilon if self.training else 0.)

class OriginalFleetPolicy(OriginalHeadPolicy,FleetContextPolicy):pass

def main():
    torch.set_num_threads(1);torch.manual_seed(46)
    _,source,_,cfg,Env=setup(5);data.SOURCE=source
    inst,public=load(read(source/'split.json')['train'][0])
    old=OriginalFleetPolicy(**cfg['model']).train();new=FleetContextKimPolicy(**cfg['model']).train()
    torch.nn.init.normal_(old.fleet_out.weight,std=.05);new.load_state_dict(old.state_dict(),strict=True)
    old.new_customer_epsilon=new.new_customer_epsilon=.05
    envs=[Env(inst,4,cfg['environment'],'cpu') for _ in range(2)]
    for ev in envs:
        # Every score tied: the first customer in GROUP order must remain head.
        tied=torch.zeros_like(ev.guidance.initial)
        for k,g in enumerate(ev.guidance.groups):tied[k,g]=.5
        ev.guidance.update(tied,rollout_ids=[0,1],effective_time_min=0.)
        for k,g in enumerate(ev.guidance.groups):assert int(ev.guidance.new_head_slots[0,k])==g[0]-1
        before=ev.guidance.new_head_slots.clone()
        changed=torch.zeros_like(tied)
        for k,g in enumerate(ev.guidance.groups):changed[k,g[-1]]=1.
        # Positive-time confidence affects routing but never recalculates heads.
        ev.guidance._select_new_head_slots=lambda *_: (_ for _ in ()).throw(AssertionError('Unexpected head calculation'))
        ev.guidance.update(changed,effective_time_min=30.)
        assert torch.equal(before,ev.guidance.new_head_slots)
    encoded=[m.encode_events(inst,public,batch_size=2) for m in (old,new)]
    logs=[[],[]];max_error=0.;routing_steps=0
    for step in range(600):
        if bool(envs[0].done.all()):break
        probs=[m.probabilities(ev.frame(),*enc,ev) for m,ev,enc in zip((old,new),envs,encoded)]
        max_error=max(max_error,float((probs[0]-probs[1]).abs().max().detach()))
        assert torch.allclose(probs[0],probs[1],atol=1e-7,rtol=1e-6)
        action=probs[0].detach().multinomial(1).squeeze(1)
        if not bool(envs[0].dispatch_open.any()):routing_steps+=1
        for i,ev in enumerate(envs):
            logs[i].append(probs[i][ev.ids,action].clamp_min(1e-30).log());ev.step(action,validate=True)
    assert bool(envs[0].done.all()) and bool(envs[1].done.all()) and routing_steps>0
    assert envs[1]._new_head_bias_finished
    losses=[]
    for log in logs:
        loss=-torch.stack(log).sum(0).mean();loss.backward();losses.append(float(loss.detach()))
    max_grad=0.
    for (name,p),(_,q) in zip(old.named_parameters(),new.named_parameters()):
        if p.grad is not None:
            assert torch.allclose(p.grad,q.grad,atol=2e-5,rtol=2e-5),name
            max_grad=max(max_grad,float((p.grad-q.grad).abs().max()))
    report=dict(passed=True,decisions=step,routing_steps=routing_steps,max_probability_error=max_error,max_gradient_error=max_grad,losses=losses,partial_t0_update_and_group_order_ties=True,future_update_no_head_recompute=True,policy_loop_no_head_recompute=True)
    (ROOT/'result/cached_new_heads_verification_20261125.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
