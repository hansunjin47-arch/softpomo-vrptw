"""Opt-in dispatch guidance variants; legacy GuidedPolicy remains unchanged."""
import torch
from experiments.concurrent_fleet_pair.policy import FleetPolicy, mix_new_customers
from experiments.concurrent_fleet_pair_kim.run import GuidedPolicy


class DispatchGuidedPolicy(GuidedPolicy):
    # NEW has no anchor yet. Only anchored existing-vehicle/customer pairs
    # receive scores. Direct mode intentionally changes NEW/OLD family mass.
    dispatch_mode = 'direct'

    def probabilities(self, frame, starts, encodings, env):
        epsilon = self.new_customer_epsilon
        self.new_customer_epsilon = 0.
        try:
            base = FleetPolicy.probabilities(self, frame, starts, encodings, env)
        finally:
            self.new_customer_epsilon = epsilon
        guidance = env.guidance
        group = guidance.vehicle_group.gather(1, frame['vehicles'])
        scores = guidance.initial[group.clamp(min=0)]
        for effective, target, update in guidance.updates:
            eligible = target & (env.time >= effective)
            scores = torch.where(eligible[:, None, None], update[group.clamp(min=0)], scores)
        active = (group >= 0) & ~env.done[:, None]
        bias = guidance.beta * scores * active[:, :, None]
        bias = torch.cat([bias.new_zeros(env.count, env.n), bias.flatten(1)], 1)
        weighted = base * bias.exp()
        guided = weighted / weighted.sum(1, keepdim=True)
        if self.dispatch_mode == 'preserve_family':
            old = weighted[:, env.n:]
            old = old / old.sum(1, keepdim=True).clamp_min(torch.finfo(old.dtype).tiny)
            old = old * base[:, env.n:].sum(1, keepdim=True)
            preserved = torch.cat([base[:, :env.n], old], 1)
            guided = torch.where(env.dispatch_open[:, None], preserved, guided)
        elif self.dispatch_mode != 'direct':
            raise ValueError('Unknown dispatch mode')
        return mix_new_customers(guided, frame['mask'], env.n, epsilon if self.training else 0.)
