"""Read-only FP32 gradient attribution at the greedy dispatch boundary."""
import copy
import hashlib
import torch
from analysis.evaluate_r101_mask_kim_20261109 import BUNDLE, RECEIVED
from experiments.concurrent_fleet_pair_kim.run import GuidedEnv, GuidedPolicy
from experiments.concurrent_fleet_v3 import run as data_module
from experiments.concurrent_fleet_bias_guided.support import ROOT, read, load
from experiments.concurrent_fleet_v9_guided.prompts import parse_response
from experiments.concurrent_fleet_pair.io import write


def main():
    torch.set_num_threads(1)
    data_module.SOURCE=BUNDLE/'dataset'
    run=RECEIVED/'runs/seed44'
    cfg=read(run/'protocol.json')['configs']['5']
    checkpoint=run/'late5/checkpoint-1000.pt'
    original_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    groups=read(BUNDLE/'artifacts/groups.json')
    split=read(run/'split.json')['train']
    results=[]
    for row in [split[0],split[10]]:
        inst,public=load(row)
        cache=read(BUNDLE/'artifacts/train_cache'/(row['name']+'.json'))
        def environment(count):
            env=GuidedEnv(inst,count,cfg['environment'],'cuda',groups=groups,scores=parse_response(cache['initial'],groups))
            if cache['post_disclosure'] is not None:
                t=cache['revealed_at_min'];env.update_event_confidence(parse_response(cache['post_disclosure'],groups),revealed_at_min=t,effective_time_min=t)
            return env
        model=GuidedPolicy(**cfg['model']).cuda().eval()
        model.load_state_dict(saved['model_state_dict'])
        env=environment(1)
        prefix=[]
        with torch.no_grad():
            starts,encoded=model.encode_events(inst,public)
            for _ in range(100):
                probs=model.probabilities(env.frame(),starts,encoded,env)
                action=probs.argmax(1)
                new,_,_=env.decode_action(action)
                if bool(env.dispatch_open[0]) and not bool(new[0]):break
                prefix.append(int(action[0]))
                env.step(action)
            else:raise RuntimeError('No closure')
            new_slot=int(probs[0,:env.n].argmax());stop_slot=int(action[0])
        starts,encoded=model.encode_events(inst,public)
        probs=model.probabilities(env.frame(),starts,encoded,env)
        margin=probs[0,new_slot].log()-probs[0,stop_slot].log()
        parameters=[p for p in model.parameters() if p.requires_grad]
        gm=torch.autograd.grad(margin,parameters,allow_unused=True)
        gm=[None if x is None else x.detach() for x in gm]
        for rng in [20261109,20261110]:
            torch.manual_seed(rng)
            model.train();model.new_customer_epsilon=.01
            env=environment(100)
            starts,encoded=model.encode_events(inst,public,batch_size=1)
            dispatch=[];routing=[]
            for _ in range(600):
                if bool(env.done.all()):break
                live=(~env.done).clone();phase=env.dispatch_open.clone()
                probs=model.probabilities(env.frame(),starts,encoded,env)
                action=probs.multinomial(1).squeeze(1)
                lp=probs[env.ids,action].clamp_min(1e-30).log()
                dispatch.append(torch.where(live & phase,lp,0.))
                routing.append(torch.where(live & ~phase,lp,0.))
                env.step(action)
            else:raise RuntimeError('Action bound')
            metrics=env.metrics();reward=metrics['reward'].float().detach()
            advantage=(reward-reward.mean())/reward.std().clamp_min(1e-6)
            losses={'dispatch':-(advantage*torch.stack(dispatch).sum(0)).mean(),
                    'routing':-(advantage*torch.stack(routing).sum(0)).mean()}
            directions={}
            total_grads=[None for _ in parameters]
            for part,loss in losses.items():
                grads=torch.autograd.grad(loss,parameters,retain_graph=part=='dispatch',allow_unused=True)
                assert all(bool(torch.isfinite(g).all()) for g in grads if g is not None)
                dot=sum(float((a*b).sum()) for a,b in zip(gm,grads) if a is not None and b is not None)
                directions[part]=-dot
                for i,g in enumerate(grads):
                    if g is not None:total_grads[i]=g.detach().clone() if total_grads[i] is None else total_grads[i]+g.detach()
            directions['total']=sum(directions.values())
            optimizer=torch.optim.Adam(model.parameters(),**cfg['optimizer']['optimizer'])
            optimizer.load_state_dict(saved['optimizer_state_dict'])
            norm=torch.sqrt(sum(g.square().sum() for g in total_grads if g is not None))
            scale=min(1.,1./(float(norm)+1e-6))
            adam_delta=0.
            for group in optimizer.param_groups:
                assert group.get('weight_decay',0)==0 and not group.get('amsgrad',False)
                b1,b2=group['betas']
                for p,g,mgrad in zip(group['params'],total_grads,gm):
                    if g is None or mgrad is None:continue
                    state=optimizer.state[p];step=float(state['step'])+1
                    g=g*scale
                    m=state['exp_avg']*b1+g*(1-b1)
                    v=state['exp_avg_sq']*b2+g.square()*(1-b2)
                    delta=-group['lr']*(m/(1-b1**step))/(torch.sqrt(v/(1-b2**step))+group['eps'])
                    adam_delta+=float((mgrad*delta).sum())
            actual_updates=[]
            for variant in ('existing_adam','reset_first_moment','fresh_adam'):
                candidate=GuidedPolicy(**cfg['model']).cuda().eval()
                candidate.load_state_dict(saved['model_state_dict'])
                opt=torch.optim.Adam(candidate.parameters(),**cfg['optimizer']['optimizer'])
                if variant!='fresh_adam':
                    opt.load_state_dict(copy.deepcopy(saved['optimizer_state_dict']))
                    if variant=='reset_first_moment':
                        for state in opt.state.values():state['exp_avg'].zero_()
                for p,g in zip(candidate.parameters(),total_grads):p.grad=None if g is None else g.clone()
                torch.nn.utils.clip_grad_norm_(candidate.parameters(),1.)
                opt.step()
                with torch.no_grad():
                    boundary=environment(1)
                    for action_id in prefix:boundary.step(torch.tensor([action_id],device='cuda'))
                    ss,ee=candidate.encode_events(inst,public)
                    pp=candidate.probabilities(boundary.frame(),ss,ee,boundary)
                    after=float(pp[0,new_slot].log()-pp[0,stop_slot].log())
                    values=dict(variant=variant,margin_after=after,margin_change=after-float(margin.detach()),
                        new_probability=float(pp[0,new_slot]),stop_probability=float(pp[0,stop_slot]),
                        chooses_new=bool(candidate.probabilities(boundary.frame(),ss,ee,boundary).argmax(1)[0]<boundary.n))
                    rollout=environment(1)
                    ss,ee=candidate.encode_events(inst,public)
                    for _ in range(600):
                        if bool(rollout.done.all()):break
                        pp=candidate.probabilities(rollout.frame(),ss,ee,rollout)
                        rollout.step(pp.argmax(1))
                    else:raise RuntimeError('Candidate exceeded action bound')
                    values['greedy']={k:float(v[0]) for k,v in rollout.metrics().items() if k in ('reward','vehicles','travel_time','total_late','unserved')}
                actual_updates.append(values)
                del candidate,opt,boundary,rollout,ss,ee,pp
            bins=[]
            for k in sorted(set(metrics['vehicles'].cpu().tolist())):
                mask=metrics['vehicles']==k
                bins.append(dict(vehicles=k,count=int(mask.sum()),reward=float(reward[mask].mean()),
                    advantage=float(advantage[mask].mean()),late=float(metrics['total_late'][mask].mean())))
            result=dict(name=row['name'],rng=rng,rollouts=100,precision='FP32',epsilon=.01,
                baseline='within 100 rollouts; mean/std',closure_log_new_minus_stop=float(margin.detach()),
                local_sgd_margin_direction=directions,gradient_norm=float(norm),
                adam_first_order_margin_change=adam_delta,by_vehicle=bins,actual_updates=actual_updates)
            results.append(result)
            write(ROOT/'analysis/r101_kim44_adam_step_20261109.json',results)
            print(result,flush=True)
            del dispatch,routing,losses,grads,probs,encoded,env
        del model,gm
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==original_hash
    print('Original checkpoint hash unchanged',flush=True)


if __name__=='__main__':main()
