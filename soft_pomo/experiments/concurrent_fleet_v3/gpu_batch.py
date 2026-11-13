"""Batched CUDA policy sampling and on-policy, state-buffer gradient replay.

Event simulation remains on CPU. Buffers are used once before the optimizer
step; they are not an off-policy replay buffer.
"""
import numpy as np
import torch


def frame(env):
    actions = env.actions()
    return dict(actions=actions, targets=np.asarray([env.action_target(a) for a in actions]),
                features=env.action_features(actions), fleet=env.observation()['fleet'])


def distribution(model, encoded, frames):
    size=len(frames)
    width=max(len(f['targets']) for f in frames)
    vehicles=max(1, max(len(f['fleet']) for f in frames))
    targets=np.zeros((size,width),dtype=np.int64)
    features=np.zeros((size,width,16),dtype=np.float32)
    fleet=np.zeros((size,vehicles,10),dtype=np.float32)
    valid=np.zeros((size,width),dtype=bool)
    fleet_valid=np.zeros((size,vehicles),dtype=np.float32)
    for i,f in enumerate(frames):
        n=len(f['targets']); v=len(f['fleet'])
        targets[i,:n]=f['targets']; features[i,:n]=f['features']; valid[i,:n]=True
        fleet[i,:v]=f['fleet']; fleet_valid[i,:v]=1
    device=encoded.device
    targets=torch.as_tensor(targets,device=device)
    features=torch.as_tensor(features,device=device)
    fleet=torch.as_tensor(fleet,device=device)
    mask=torch.as_tensor(fleet_valid,device=device)
    pooled=(model.fleet(fleet)*mask[:,:,None]).sum(1)/mask.sum(1).clamp(min=1)[:,None]
    x=torch.cat([encoded[targets],encoded.mean(0).expand(size,width,-1),
                 pooled[:,None,:].expand(-1,width,-1),features],dim=-1)
    logits=10*torch.tanh(model.scorer(x).squeeze(-1))
    logits=logits.masked_fill(~torch.as_tensor(valid,device=device),-torch.inf)
    return torch.distributions.Categorical(logits=logits)


@torch.no_grad()
def sample(model, inst, envs, progress=None):
    encoded=model.encode(inst)
    records=[]
    limit=max((e.vehicle_limit+2)*(4*e.n+len(e.public_events)+1) for e in envs)
    for step in range(limit):
        active=[i for i,e in enumerate(envs) if not e.done]
        if not active:
            return [e.metrics() for e in envs], records
        frames=[frame(envs[i]) for i in active]
        dist=distribution(model,encoded,frames)
        choices=dist.sample()
        logps=dist.log_prob(choices).cpu().tolist()
        choices=choices.cpu().tolist()
        for i,f,choice,logp in zip(active,frames,choices,logps):
            envs[i].step(f['actions'][choice])
            del f['actions']
            records.append(dict(**f, trajectory=i, choice=choice, logp=logp))
        if progress is not None and step%25==0:
            progress(step,len(active))
    raise RuntimeError('Batched trajectory exceeded finite action bound')


def backward(model, inst, records, advantages, batch_size, chunk=256):
    device=next(model.parameters()).device
    advantages=advantages.to(device)
    loss_value=0.
    for offset in range(0,len(records),chunk):
        frames=records[offset:offset+chunk]
        dist=distribution(model,model.encode(inst),frames)
        choices=torch.tensor([f['choice'] for f in frames],device=device)
        ids=torch.tensor([f['trajectory'] for f in frames],device=device)
        logp=dist.log_prob(choices)
        old=torch.tensor([f['logp'] for f in frames],device=device)
        if not torch.allclose(logp.detach(),old,atol=5e-5,rtol=1e-5):
            raise RuntimeError('On-policy replay probability mismatch')
        loss=-(advantages[ids]*logp).sum()/(len(advantages)*batch_size)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        loss.backward()
        loss_value+=float(loss.detach())
    return loss_value
