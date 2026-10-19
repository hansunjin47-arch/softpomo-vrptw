import torch
from experiments.concurrent_fleet_v9.env import TensorFleetEnv
from experiments.concurrent_fleet_guided.guidance import ConfidenceGuidance


class GuidedFleetEnv(TensorFleetEnv):
    def __init__(self, inst, count, params, device='cuda', *, groups, scores, beta=5.):
        super().__init__(inst, count, params, device)
        self.guidance = ConfidenceGuidance(self, groups, scores, beta)

    def step(self, action, validate=False):
        new = (action < self.n) & ~self.done
        ids = self.ids[new]
        vehicles = self.k[new].clone()
        groups = self.guidance.membership[action[new]+1]
        super().step(action, validate=validate)
        self.guidance.vehicle_group[ids, vehicles] = groups

    def update_confidence(self, scores, rollout_ids=None, effective_time_min=None):
        self.guidance.update(scores, rollout_ids, effective_time_min)

    def update_event_confidence(self, scores, *, revealed_at_min, rollout_ids=None,
                                effective_time_min=None):
        ids = self.ids if rollout_ids is None else torch.as_tensor(rollout_ids, device=self.device, dtype=torch.long)
        disclosure = torch.as_tensor(revealed_at_min, device=self.device, dtype=self.time.dtype)
        if disclosure.numel() != 1 or not bool(torch.isfinite(disclosure)) or disclosure < 0:
            raise ValueError('Disclosure must be a finite nonnegative scalar')
        effective = torch.maximum(self.time[ids], disclosure) if effective_time_min is None else torch.as_tensor(
            effective_time_min, device=self.device, dtype=self.time.dtype).expand(ids.numel())
        if bool((effective < disclosure).any()):
            raise ValueError('Event guidance cannot precede disclosure')
        self.guidance.update(scores, ids, effective)
