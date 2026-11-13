"""Per-vehicle anchored confidence, supplied externally as scores in [0, 1]."""
from collections.abc import Mapping
import math
import weakref
import torch


class ConfidenceGuidance:
    def __init__(self, env, groups, scores, beta=5.0):
        self.env = weakref.proxy(env)
        self.groups = [list(group) for group in groups]
        customers = [n for group in self.groups for n in group]
        if not self.groups or any(not group for group in self.groups):
            raise ValueError('Groups must be nonempty')
        if sorted(customers) != list(range(1, env.n + 1)):
            raise ValueError('Groups must partition every customer exactly once')
        if not math.isfinite(beta) or beta < 0:
            raise ValueError('beta must be finite and nonnegative')
        self.beta = beta
        self.membership = torch.full((env.n + 1,), -1, dtype=torch.long, device=env.device)
        for k, nodes in enumerate(self.groups):
            self.membership[nodes] = k
        self.vehicle_group = torch.full((env.count, env.v), -1, dtype=torch.long, device=env.device)
        self.initial = self._tensor(scores)
        self.updates = []

    def _tensor(self, scores):
        env = self.env
        allowed = torch.zeros(len(self.groups), env.n + 1, dtype=torch.bool, device=env.device)
        for k, nodes in enumerate(self.groups):
            allowed[k, nodes] = True
        if isinstance(scores, Mapping):
            values = {int(k): v for k, v in scores.items()}
            if len(values) != len(scores) or set(values) != set(range(len(self.groups))):
                raise ValueError('Score group IDs must match the supplied groups (zero-based)')
            tensor = torch.zeros_like(allowed, dtype=torch.float32)
            for k, nodes in enumerate(self.groups):
                row = {int(n): float(v) for n, v in values[k].items()}
                if len(row) != len(values[k]) or set(row) != set(nodes):
                    raise ValueError('Each score row must contain exactly its group customers')
                tensor[k, nodes] = torch.tensor([row[n] for n in nodes], device=env.device)
        else:
            tensor = torch.as_tensor(scores, device=env.device, dtype=torch.float32).detach().clone()
            if tensor.shape != allowed.shape:
                raise ValueError('Expected confidence shape [groups, customers + depot]')
        if not bool(torch.isfinite(tensor).all()) or bool(((tensor < 0) | (tensor > 1)).any()):
            raise ValueError('Confidence must be finite and in [0, 1]')
        if bool((tensor[~allowed] != 0).any()):
            raise ValueError('Depot and out-of-group confidence entries must be zero')
        return tensor

    def update(self, scores, rollout_ids=None, effective_time_min=None):
        """Apply externally generated scores from the current (or later) clock.

        No state is sent externally here. Partial updates target only specified
        rollout IDs, and never alter group membership or committed routes.
        """
        env = self.env
        ids = env.ids if rollout_ids is None else torch.as_tensor(rollout_ids, device=env.device, dtype=torch.long)
        if ids.ndim != 1 or not ids.numel() or bool(((ids < 0) | (ids >= env.count)).any()):
            raise ValueError('Invalid rollout IDs')
        if ids.unique().numel() != ids.numel():
            raise ValueError('Duplicate rollout IDs')
        times = env.time[ids].clone() if effective_time_min is None else torch.as_tensor(
            effective_time_min, device=env.device, dtype=env.time.dtype).expand(ids.numel()).clone()
        if not bool(torch.isfinite(times).all()) or bool((times < env.time[ids]).any()):
            raise ValueError('Confidence cannot be applied retroactively')
        for previous_times, previous_target, _ in self.updates:
            if bool((previous_target[ids] & (times < previous_times[ids])).any()):
                raise ValueError('Updates must be chronological for each rollout')
        target = torch.zeros(env.count, dtype=torch.bool, device=env.device)
        target[ids] = True
        effective = torch.full_like(env.time, torch.inf)
        effective[ids] = times
        self.updates.append((effective, target, self._tensor(scores)))

    def action_bias(self):
        env = self.env
        vehicle, ready, _ = env.next_vehicle()
        group = self.vehicle_group[env.ids, vehicle]
        active = ready & (group >= 0) & ~env.done
        selected = self.initial[group.clamp(min=0)]
        for effective, target, tensor in self.updates:
            eligible = target & (env.time >= effective)
            selected = torch.where(eligible[:, None], tensor[group.clamp(min=0)], selected)
        bias = torch.zeros(env.count, 2 * env.n + 2, device=env.device)
        # NEW, depot and SELECT remain unbiased. Reservation/capacity masks are
        # still applied by the policy after this additive logit adjustment.
        bias[:, env.n:2 * env.n + 1] = self.beta * selected * active[:, None]
        return bias
