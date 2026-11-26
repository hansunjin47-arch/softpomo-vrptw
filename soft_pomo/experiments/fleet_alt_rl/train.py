"""RC101 fixed-cache experiments. One epoch covers Train20 once."""
import argparse
from collections import defaultdict, deque
import copy
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from analysis.train_kim_fleet_context_20261124 import setup
from experiments.concurrent_fleet_v3 import run as data_module
from experiments.concurrent_fleet_bias_guided.support import load, read
from experiments.concurrent_fleet_pair.fleet_context import FleetContextPolicy
from experiments.concurrent_fleet_pair_kim.fleet_context import FleetContextKimPolicy
from experiments.concurrent_fleet_pair.io import write
from .models import Critic, ActionValue
from .state import action_bias, observe, pack, unpack, local_index, operating_cost


def active_embedding(starts, encodings, clock):
    e = encodings[0].expand(len(clock),-1,-1)
    for t, candidate in zip(starts,encodings[1:]):
        e = torch.where((clock>=t)[:,None,None],candidate,e)
    return e


def update(loss, model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    return float(norm)


def q_values(model, item, state, encoded=None):
    inst, public, geo = item
    encoded = model.encode_events(inst,public) if encoded is None else encoded
    e = active_embedding(*encoded,state['time'])
    return model.values(e,state,*geo)


def behavior(q, s, args, greedy=False, bias=True):
    scores = q/args.temperature
    if bias:
        scores = scores + action_bias(s,args.beta)
    mask = s['mask'].clone()
    mask[:,0] |= ~mask.any(1)
    scores = scores.masked_fill(~mask,-torch.inf)
    if greedy:
        return scores.argmax(1)
    p = (1-args.epsilon)*scores.softmax(1) + args.epsilon*mask/mask.sum(1,keepdim=True)
    return p.multinomial(1).squeeze(1)


class Replay:
    """Store batched CPU snapshots once; ring entries identify individual rows."""
    def __init__(self, capacity):
        self.rows = deque(maxlen=capacity)

    def add(self, name, s, nxt, actions, rewards, ends, live):
        before, after = pack(s), pack(nxt)
        for i in live.nonzero().flatten().tolist():
            self.rows.append((name,before,after,i,actions[i],rewards[i],ends[i]))

    def grouped(self, size):
        grouped = defaultdict(list)
        for record in random.sample(self.rows,min(size,len(self.rows))):
            grouped[record[0]].append(record)
        return grouped


def replay_update(net, target, opt, replay, items, size, device):
    groups = replay.grouped(size)
    errors = []
    for name, rows in groups.items():
        def states(records, index):
            return unpack([{k:v[r[3]] for k,v in r[index].items()} for r in records],device)
        s = states(rows,1)
        action = torch.tensor([r[4] for r in rows],device=device)
        index = local_index(s,action)
        pred = q_values(net,items[name],s).gather(1,index[:,None]).squeeze(1)
        y = pred.new_tensor([r[5] for r in rows])
        live = [i for i,r in enumerate(rows) if not r[6]]
        if live:
            with torch.no_grad():
                nxt = states([rows[i] for i in live],2)
                if not nxt['mask'].any(1).all():
                    raise ValueError('Live replay row without action')
                choice = q_values(net,items[name],nxt).masked_fill(~nxt['mask'],-torch.inf).argmax(1)
                y[live] += q_values(target,items[name],nxt).gather(1,choice[:,None]).squeeze(1)
        errors.append(F.smooth_l1_loss(pred,y,reduction='none'))
    loss = torch.cat(errors).mean()
    return float(loss.detach()),update(loss,net,opt)


def episode(args, env, name, items, net, opt, aux, auxopt, replay, state, evaluate=False, bias=True):
    inst, public, _ = items[name]
    ac = args.method=='a'
    with torch.set_grad_enabled(ac and not evaluate):
        starts, enc = net.encode_events(inst,public)
    traces, rewards = [], []
    total = torch.zeros(env.count,device=env.device,dtype=torch.float64)
    prev = operating_cost(env.metrics())
    transitions = 0
    diagnostics = {}
    for step in range(600):
        if env.done.all():
            break
        s = observe(env,args.guided)
        live = ~env.done
        if ac:
            with torch.set_grad_enabled(not evaluate):
                probs = net.probabilities(env.frame(),starts,enc,env)
                action = probs.argmax(1) if evaluate else probs.multinomial(1).squeeze(1)
                if not evaluate:
                    logp = probs.gather(1,action[:,None]).squeeze(1).log()
                    e = active_embedding(starts,[x.detach() for x in enc],s['time'])
                    # Backward only needs owned tensors; the environment mutates its fields in place.
                    value = aux(e,{k:v.clone() for k,v in s.items()})
                    traces.append((logp,value,live))
        else:
            with torch.no_grad():
                q = q_values(net,items[name],s,(starts,enc))
                action = behavior(q,s,args,evaluate,bias)
                if not evaluate:
                    if step==0:
                        valid = q[s['mask']]
                        diagnostics.update(q_min=float(valid.min()),q_max=float(valid.max()),
                                           max_prior=float(action_bias(s,args.beta).max()))
                    s = {k:v.clone() for k,v in s.items()}
                    new, veh, dest = env.decode_action(action)
                    canonical = torch.stack((veh.masked_fill(new,-1),dest),1).tolist()
        transitions += int(live.sum())
        env.step(action,validate=True)
        m = env.metrics()
        cost = operating_cost(m)
        ended = live & env.done
        r = prev-cost-ended*m['cost_unserved'].double()
        total += r
        prev = cost
        if not evaluate:
            if ac:
                rewards.append(r.float())
            else:
                replay.add(name,s,observe(env,args.guided),canonical,r.tolist(),ended.tolist(),live)
    if not env.done.all():
        raise RuntimeError('Incomplete trajectory')
    torch.testing.assert_close(total,m['reward'].double(),rtol=2e-6,atol=2e-6)
    if ac and not evaluate:
        G = torch.stack(rewards).flip([0]).cumsum(0).flip([0])
        if args.baseline=='terminal':
            G = m['reward'][None].expand_as(G)
        logp,v,mask = [torch.stack(part) for part in zip(*traces)]
        advantage = G-v.detach()
        actor_loss = -(advantage*logp*mask).sum()/env.count
        critic_loss = F.smooth_l1_loss(v[mask],G[mask])
        diagnostics.update(actor_norm=update(actor_loss,net,opt),critic_norm=update(critic_loss,aux,auxopt),
                           actor_loss=float(actor_loss.detach()),critic_loss=float(critic_loss.detach()))
    elif not ac and not evaluate:
        # Keep collection policy fixed over a trajectory, then learn from replay.
        for _ in range(args.q_updates):
            loss,norm = replay_update(net,aux,opt,replay,items,args.minibatch,env.device)
            state['q_updates'] += 1
            if state['q_updates'] % args.target_interval == 0:
                aux.load_state_dict(net.state_dict())
        diagnostics.update(td_loss=loss,q_norm=norm,replay_rows=len(replay.rows))
    return dict(**{k:float(m[k].float().mean()) for k in
                   ('reward','vehicles','travel_time','total_late','unserved')},
                transitions=transitions,**diagnostics)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method',choices=['a','b'],required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--pure',dest='guided',action='store_false')
    p.add_argument('--baseline',choices=['terminal','remaining'],default='remaining')
    p.add_argument('--epochs',type=int,default=1000)
    p.add_argument('--rollouts',type=int,default=32)
    p.add_argument('--device',default='cuda')
    p.add_argument('--cases',type=int,default=20)
    p.add_argument('--q-updates',type=int,default=8)
    p.add_argument('--minibatch',type=int,default=32)
    p.add_argument('--capacity',type=int,default=32768)
    p.add_argument('--target-interval',type=int,default=100)
    p.add_argument('--temperature',type=float,default=1.)
    p.add_argument('--beta',type=float,default=5.)
    p.add_argument('--epsilon',type=float,default=.05)
    args = p.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    random.seed(46)
    torch.manual_seed(46)
    _,source,_,cfg,Env = setup(5, portable=True)
    data_module.SOURCE = source
    params = cfg['environment']
    items = {}
    for row in read(source/'split.json')['train'][:args.cases]:
        inst,public = load(row)
        probe = Env(inst,1,params,device=args.device)
        geo = tuple(x/probe.horizon for x in (probe.base,probe.open,probe.close))
        items[row['name']] = (inst,public,geo)
    ac = args.method=='a'
    cls = (FleetContextKimPolicy if args.guided else FleetContextPolicy) if ac else ActionValue
    net = cls(**cfg['model']).to(args.device)
    aux = Critic(cfg['model']['embedding_dim']).to(args.device) if ac else copy.deepcopy(net).eval()
    opt = torch.optim.Adam(net.parameters(),lr=1e-4)
    auxopt = torch.optim.Adam(aux.parameters(),lr=1e-4) if ac else None
    replay = None if ac else Replay(args.capacity)
    checkpoint = args.out/'checkpoint.pt'
    protocol = {**vars(args),'out':str(args.out),'seed':46,'objective':[100,1,5]}
    state = dict(epoch=0,q_updates=0)
    if checkpoint.exists():
        saved = torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved['protocol'] != protocol:
            raise ValueError('Checkpoint protocol mismatch')
        net.load_state_dict(saved['model'])
        aux.load_state_dict(saved['aux'])
        opt.load_state_dict(saved['opt'])
        if ac:
            auxopt.load_state_dict(saved['auxopt'])
        state,replay = saved['state'],saved['replay']
        random.setstate(saved['python_rng'])
        torch.set_rng_state(saved['rng'])
        if args.device=='cuda':
            torch.cuda.set_rng_state(saved['cuda_rng'])
        del saved
    for epoch in range(state['epoch']+1,args.epochs+1):
        start = time.monotonic()
        names = list(items)
        random.shuffle(names)
        rows = []
        lr = 1e-4*(1-max(0,epoch-500)*.9/500)
        for optimizer in (opt,auxopt):
            if optimizer:
                for group in optimizer.param_groups:
                    group['lr']=lr
        net.train()
        net.new_customer_epsilon = .05-.04*min(1,(epoch-1)/999)
        for name in names:
            env = Env(items[name][0],args.rollouts,params,device=args.device)
            rows.append(episode(args,env,name,items,net,opt,aux,auxopt,replay,state))
        state['epoch']=epoch
        write(args.out/f'epochs/{epoch:04}.json',dict(epoch=epoch,cases=dict(zip(names,rows)),
              seconds=time.monotonic()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else 0))
        tmp = checkpoint.with_suffix('.tmp')
        torch.save(dict(protocol=protocol,model=net.state_dict(),aux=aux.state_dict(),opt=opt.state_dict(),
                        auxopt=auxopt.state_dict() if ac else None,state=state,replay=replay,
                        python_rng=random.getstate(),rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state() if args.device=='cuda' else None),tmp)
        tmp.replace(checkpoint)
        print(f'{args.method} epoch={epoch} reward={np.mean([r["reward"] for r in rows]):.5f}',flush=True)
    net.eval()
    with torch.no_grad():
        for use_bias in ([True] if ac or not args.guided else [False,True]):
            scores = {}
            for name,item in items.items():
                env = Env(item[0],1,params,device=args.device)
                scores[name] = episode(args,env,name,items,net,opt,aux,auxopt,replay,state,True,use_bias)
            write(args.out/f'train_greedy_bias{int(use_bias)}.json',scores)
    write(args.out/'complete.json',dict(epochs=args.epochs,train_cases=len(items),protocol=protocol))


if __name__=='__main__':
    main()
