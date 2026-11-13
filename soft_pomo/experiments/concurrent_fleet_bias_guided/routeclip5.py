"""Isolated routing-only C=5 policy; dispatch and guidance amplitudes unchanged."""
import torch
from experiments.concurrent_fleet_stop_init.policy import FleetPolicy


class RouteClip5Policy(FleetPolicy):
    def _phase_scores(self, encoded, frame, env, dispatch):
        scores = super()._phase_scores(encoded, frame, env, dispatch)
        if not dispatch:
            scores = scores * (5. / self.decoder.logit_clipping)
        return scores


class GuidedRouteClip5Policy(RouteClip5Policy):
    def _phase_scores(self, encoded, frame, env, dispatch):
        scores = super()._phase_scores(encoded, frame, env, dispatch)
        guidance = getattr(env, 'guidance', None)
        if dispatch or guidance is None or guidance.beta == 0:
            return scores
        bias = guidance.action_bias()[:, env.n:2*env.n+1]
        return scores.float() + torch.cat([bias, bias.new_zeros(env.count, 1)], 1)
