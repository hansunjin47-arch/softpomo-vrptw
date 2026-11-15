"""NEW-only bias for group heads during dispatch, then legacy Kim routing."""
import math
import torch
from experiments.concurrent_fleet_pair.policy import mix_new_customers
from experiments.concurrent_fleet_pair_kim.run import GuidedPolicy


class ConfidenceHeadGuidedPolicy(GuidedPolicy):
    # Independent of env.guidance.beta, which controls subsequent routing.
    # Heads are each group's highest currently disclosed confidence customer.
    new_head_beta = 5.0

    def probabilities(self, frame, starts, encodings, env):
        beta = float(self.new_head_beta)
        if not math.isfinite(beta) or beta < 0:
            raise ValueError('new_head_beta must be finite and nonnegative')
        epsilon = self.new_customer_epsilon
        self.new_customer_epsilon = 0.
        try:
            # Legacy Kim: no existing-pair bias during dispatch; normal
            # anchor-group confidence after dispatch has closed.
            base = super().probabilities(frame, starts, encodings, env)
        finally:
            self.new_customer_epsilon = epsilon
        if beta:
            def select(scores):
                return torch.stack([
                    torch.tensor(g, device=base.device)[scores[k, g].argmax()] - 1
                    for k, g in enumerate(env.guidance.groups)])
            heads = select(env.guidance.initial).expand(base.shape[0], -1)
            for effective, target, scores in env.guidance.updates:
                eligible = target & (env.time >= effective)
                heads = torch.where(eligible[:, None], select(scores)[None, :], heads)
            active = env.dispatch_open & ~env.done
            bias = torch.zeros_like(base)
            bias.scatter_(1, heads, active[:, None].to(base.dtype).expand_as(heads) * beta)
            bias = bias.masked_fill(base == 0, 0.)
            # Shift only active rows for numerical stability. Masked zeros
            # stay zero, including heads already assigned to another vehicle.
            weighted = base * (bias-bias.max(1, keepdim=True).values).exp()
            adjusted = weighted / weighted.sum(1, keepdim=True)
            base = torch.where(active[:, None], adjusted, base)
        return mix_new_customers(base, frame['mask'], env.n,
                                 epsilon if self.training else 0.)
