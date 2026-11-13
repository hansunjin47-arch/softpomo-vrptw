import torch
from experiments.concurrent_fleet_v8.env import TensorFleetEnv as BaseEnv


class TensorFleetEnv(BaseEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dispatch_open = torch.ones(self.count, dtype=torch.bool, device=self.device)

    def mask(self):
        i, ready, active = self.next_vehicle()
        free = self.customer == 0
        live = ~self.done
        demand = (free * self.demand).sum(1)
        operating = (self.status > 0) & (self.status < 4) & (self.target != 0)
        capacity = (operating * self.load).sum(1)
        new = free[:, 1:] & (self.dispatch_open & (self.k < self.v) & live)[:, None]
        old = free & (self.demand[None] <= self.load[self.ids, i, None] + 1e-5)
        fits = old[:, 1:].any(1)
        old[:, 0] = (demand <= capacity - self.load[self.ids, i] + 1e-5) | ~fits
        old &= (ready & ~self.dispatch_open & live)[:, None]
        stop = self.dispatch_open & (self.k > 0) & live & ((demand <= capacity + 1e-5) | (self.k >= self.v))
        return torch.cat([new, old, stop[:, None]], 1)

    def step(self, action, validate=False):
        if validate and not bool((self.mask()[self.ids, action] | self.done).all()):
            raise ValueError('Invalid v9 action')
        self.dispatch_open &= ~((action == 2*self.n+1) & ~self.done)
        super().step(action, validate=False)
