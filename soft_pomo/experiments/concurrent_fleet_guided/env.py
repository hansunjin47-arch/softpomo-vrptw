import torch
from experiments.concurrent_fleet_v8.env import TensorFleetEnv as BaseEnv
from experiments.concurrent_fleet_guided.guidance import ConfidenceGuidance


class GuidedFleetEnv(BaseEnv):
    def __init__(self, inst, count, params, device='cuda', *, groups, scores, beta=5.0):
        super().__init__(inst, count, params, device)
        self.guidance = ConfidenceGuidance(self, groups, scores, beta)

    def step(self, action, validate=False):
        new = (action < self.n) & ~self.done
        ids = self.ids[new]
        vehicle = self.k[new].clone()
        group = self.guidance.membership[action[new] + 1]
        super().step(action, validate=validate)
        # Anchoring happens once at actual dispatch, not when another vehicle
        # is selected and not when a vehicle crosses into another group.
        self.guidance.vehicle_group[ids, vehicle] = group

    def update_confidence(self, scores, rollout_ids=None, effective_time_min=None):
        self.guidance.update(scores, rollout_ids, effective_time_min)
