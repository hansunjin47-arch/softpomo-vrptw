"""Scale the same frozen AMP policy probabilities in float64, preserving ties."""
import torch

from analysis import diagnose_kim_fusion_20261110 as experiment
from experiments.concurrent_fleet_pair.policy import FleetPolicy
from experiments.concurrent_fleet_pair_kim.run import GuidedPolicy


class StableFusionPolicy(GuidedPolicy):
    fusion_alpha = 1.

    def probabilities(self, frame, starts, encodings, env):
        assert not self.training
        assert self.decoder.logit_clipping == 10.
        q = FleetPolicy.probabilities(self, frame, starts, encodings, env)
        guidance = env.guidance
        group = guidance.vehicle_group.gather(1, frame['vehicles'])
        confidence = guidance.initial[group.clamp(min=0)]
        for effective, target, scores in guidance.updates:
            eligible = target & (env.time >= effective)
            confidence = torch.where(eligible[:, None, None], scores[group.clamp(min=0)], confidence)
        active = (group >= 0) & ~env.dispatch_open[:, None] & ~env.done[:, None]
        bias = guidance.beta * confidence * active[:, :, None]
        bias = torch.cat([bias.new_zeros(env.count, env.n), bias.flatten(1)], 1)
        logits = q.double().log() * self.fusion_alpha + bias.double()
        p = logits.softmax(1)
        assert torch.equal(p == 0, q == 0)
        if guidance.beta == 0.:
            assert torch.equal(p.argmax(1), q.argmax(1)), 'Stable off scaling changed argmax'
        return p


def configure_scale(model, alpha):
    model.fusion_alpha = alpha


if __name__ == '__main__':
    experiment.OUT = experiment.suite.ROOT / 'result/kim_fusion_stable_20261110'
    experiment.IMPLEMENTATION = 'Original clipping10 AMP q unchanged; float64 alpha*log(q)+beta*confidence; exact zero mask. Separate from failed FP16 decoder-scaling pilot.'
    experiment.configure_scale = configure_scale
    experiment.VARIANTS.append(('alpha100_reconstruction', 1., 'original', 5.))
    experiment.suite.GuidedPolicy = StableFusionPolicy
    try:
        experiment.main()
    except Exception as error:
        experiment.write(experiment.OUT/'status.json', dict(stage='failed', error=repr(error)))
        raise
