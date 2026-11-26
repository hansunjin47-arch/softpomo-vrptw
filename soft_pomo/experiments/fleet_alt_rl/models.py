import math
import torch
from torch import nn
from experiments.concurrent_fleet_pair.fleet_context import FleetContextPolicy


def gather(embedding, ids):
    return embedding.gather(1, ids.clamp_min(0)[:,:,None].expand(-1,-1,embedding.size(-1)))


def mean(x, mask):
    return (x * mask[:,:,None]).sum(1) / mask.sum(1,keepdim=True).clamp_min(1)


class Critic(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.customer = nn.Linear(dim+5, 64)
        self.vehicle = nn.Linear(3*dim+8, 64)
        self.value = nn.Sequential(nn.Linear(64*3+4,128),nn.ReLU(),nn.Linear(128,1))

    def forward(self, encoded, s):
        flags = torch.nn.functional.one_hot(s['customer'],3)
        stop = self.customer(torch.cat((encoded, flags, s['confidence'][:,:,None],
                                        s['heads'][:,:,None]),-1)).relu()
        state = s['status']
        tokens = [gather(encoded,s[k]) for k in ('fleet_node','target','pending')]
        tokens += [torch.nn.functional.one_hot(state,5),s['capacity'][:,:,None],
                   (s['pending']>=0)[:,:,None],(s['group']>=0)[:,:,None]]
        fleet = self.vehicle(torch.cat(tokens,-1)).relu()
        global_state = torch.stack((s['time'],s['dispatch'],(state>0).float().mean(1),
                                    s['prebook'][:,0]),1)
        return self.value(torch.cat((mean(stop,s['customer']==0),mean(stop,s['customer']==1),
                                     mean(fleet,state>0),global_state),1)).squeeze(-1)


class ActionValue(FleetContextPolicy):
    """Reuse event encoder and fleet attention. Linear action values in reward units."""
    def __init__(self, **params):
        super().__init__(**params)
        dim = params['embedding_dim']
        del self.decoder
        self.action_query = nn.Linear(dim+4,dim)
        self.action_key = nn.Linear(dim,dim)
        self.state_value = nn.Linear(dim+4,1)
        self.slack_value = nn.Sequential(nn.Linear(3,16),nn.ReLU(),nn.Linear(16,1))

    def values(self, e, s, base, opening, closing):
        b, _, dim = e.shape
        fleet = [gather(e,s[k]) for k in ('fleet_node','target','pending')]
        fleet += [torch.nn.functional.one_hot(s['status'],5),s['capacity'][:,:,None],
                  (s['pending']>=0)[:,:,None]]
        tokens = self.fleet_token(torch.cat(fleet,-1)).tanh()
        live = (s['status']>0)&(s['status']<4)
        safe = live.clone()
        safe[:,0] |= ~live.any(1)
        nodes = torch.cat((torch.zeros_like(s['node'][:,:1]),s['node']),1)
        load = torch.cat((torch.ones_like(s['load'][:,:1]),s['load']),1)
        clock = s['time'][:,None].expand_as(load)
        last = gather(e,nodes)
        query = self.fleet_query(torch.cat((last,load[:,:,None],clock[:,:,None]),-1))
        weights = (query @ self.fleet_key(tokens).transpose(1,2))/math.sqrt(dim)
        fleet_context = weights.masked_fill(~safe[:,None],-torch.inf).softmax(-1) @ self.fleet_value(tokens)
        enriched = last + self.fleet_out(fleet_context)*live.any(1)[:,None,None]
        is_new = torch.zeros_like(load)
        is_new[:,0] = 1
        prebook = torch.cat((torch.zeros_like(s['prebook'][:,:1]),s['prebook']),1)
        query = torch.cat((enriched,torch.stack((load,clock,is_new,prebook),-1)),-1)
        travel = base[nodes]
        arrival = clock[:,:,None]+travel
        slack = torch.stack(((closing-arrival).clamp(-2,2).masked_fill(prebook[:,:,None],0),
                             (opening-arrival).clamp(0,2).masked_fill(prebook[:,:,None],0),travel),-1)
        q = (self.action_query(query) @ self.action_key(e).transpose(1,2))/math.sqrt(dim)
        q = q + self.state_value(query) + self.slack_value(slack).squeeze(-1)
        return torch.cat((q[:,0,1:],q[:,1:].flatten(1)),1)
