import torch
from experiments.concurrent_fleet_v9.policy import FleetPolicy


class GuidedFleetPolicy(FleetPolicy):
    """Reuse v9 scoring and masks; only routing customer logits receive bias."""

    def _phase_scores(self, encoded, frame, env, dispatch):
        scores = super()._phase_scores(encoded, frame, env, dispatch)
        guidance = getattr(env, 'guidance', None)
        if dispatch or guidance is None or guidance.beta == 0:
            return scores
        bias = guidance.action_bias()[:, env.n:2*env.n+1]
        # Keep guidance in FP32 after clipping and before the inherited hard mask.
        scores = scores.float()
        return scores + torch.cat([bias, bias.new_zeros(env.count, 1)], 1)
