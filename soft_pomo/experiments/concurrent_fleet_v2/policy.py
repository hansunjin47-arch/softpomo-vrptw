"""Shared centralized policy; legacy encoder, new fleet-aware action scorer."""
import torch
from torch import nn
from train_vrptw import _Encoder


class FleetPolicy(nn.Module):
    def __init__(self, model_params=None):
        super().__init__()
        params = model_params or dict(embedding_dim=64, encoder_layer_num=2,
                                       head_num=4, qkv_dim=16, ff_hidden_dim=128)
        self.encoder = _Encoder(**params)
        self.fleet = nn.Sequential(nn.Linear(10,32), nn.ReLU(), nn.Linear(32,32))
        self.scorer = nn.Sequential(nn.Linear(params['embedding_dim']*2+32+16,128), nn.ReLU(), nn.Linear(128,1))

    def encode(self, inst):
        features = torch.cat([inst['node_xy'], *[inst['node_'+k][:,None]
                             for k in ('demand','tw_open','tw_close','service')]], dim=-1)
        return self.encoder(inst['depot_xy'][None], features[None])[0]

    def distribution(self, env, encoded, actions):
        obs = env.observation()
        fleet = torch.as_tensor(obs['fleet'])
        pooled = self.fleet(fleet).mean(0) if len(fleet) else encoded.new_zeros(32)
        targets = torch.tensor([n for _,n in actions])
        features = torch.as_tensor(env.action_features(actions))
        x = torch.cat([encoded[targets], encoded.mean(0).expand(len(actions),-1),
                       pooled.expand(len(actions),-1),features],dim=-1)
        return torch.distributions.Categorical(logits=10*torch.tanh(self.scorer(x).squeeze(-1)))


def rollout(policy, env, encoded, greedy=False):
    logprobs = []
    for _ in range((env.vehicle_limit+2)*(4*env.n+len(env.public_events)+1)):
        if env.done:
            zero = encoded.sum()*0
            return env.metrics(), torch.stack(logprobs).sum() if logprobs else zero
        actions = env.actions()
        distribution = policy.distribution(env, encoded, actions)
        choice = distribution.logits.argmax() if greedy else distribution.sample()
        logprobs.append(distribution.log_prob(choice))
        env.step(actions[int(choice)])
    raise RuntimeError('Trajectory exceeded finite event/action bound')
