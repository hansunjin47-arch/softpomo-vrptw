"""Remove valid anchor-group membership bonus, retaining signed score residuals."""
import torch

from analysis import diagnose_kim_fusion_20261110 as experiment
from experiments.concurrent_fleet_pair.policy import FleetPolicy
from experiments.concurrent_fleet_pair_kim.run import GuidedPolicy


class ResidualPolicy(GuidedPolicy):
    fusion_alpha = .25

    def probabilities(self, frame, starts, encodings, env):
        assert not self.training and self.decoder.logit_clipping == 10.
        raw = FleetPolicy.probabilities(self, frame, starts, encodings, env)
        guidance = env.guidance
        group = guidance.vehicle_group.gather(1, frame['vehicles'])
        confidence = guidance.initial[group.clamp(min=0)]
        for effective, target, scores in guidance.updates:
            eligible = target & (env.time >= effective)
            confidence = torch.where(eligible[:, None, None], scores[group.clamp(min=0)], confidence)
        active = (group >= 0) & ~env.dispatch_open[:, None] & ~env.done[:, None]
        membership = guidance.membership[None, None, :] == group[:, :, None]
        membership[:, :, 0] = False
        valid = frame['mask'][:, env.n:].reshape_as(confidence)
        selected = membership & active[:, :, None] & valid
        count = selected.sum(-1, keepdim=True)
        mean = (confidence.double()*selected).sum(-1, keepdim=True)/count.clamp_min(1)
        original = guidance.beta*confidence.double()*active[:, :, None]
        bonus = guidance.beta*mean*selected
        residual = original-bonus
        residual = residual.masked_fill(~selected, 0.)
        assert torch.isfinite(residual).all()
        assert (residual.masked_select((count <= 1).expand_as(residual)) == 0).all()
        assert torch.allclose((original-bonus).masked_select(selected), residual.masked_select(selected))
        uniform = mean.expand_as(confidence)*selected
        uniform_residual = (uniform-mean)*selected
        assert (uniform_residual == 0).all()
        def flatten(bias):
            return torch.cat([bias.new_zeros(env.count, env.n), bias.flatten(1)], 1)
        logits = raw.double().log()*self.fusion_alpha
        p = (logits+flatten(residual)).softmax(1)
        assert torch.equal(p == 0, raw == 0)
        if guidance.beta == 0.:
            assert torch.equal(p.argmax(1), raw.argmax(1))
        if bool(env.dispatch_open.all()):
            assert torch.equal(p.argmax(1), raw.argmax(1))
        if guidance.beta:
            choices = {'original':(logits+flatten(original)).argmax(1),
                       'mean_only':(logits+flatten(bonus)).argmax(1),
                       'residual':p.argmax(1), 'off':raw.argmax(1)}
            self.last_diagnostics = {f'{key}_action':int(action[0]) for key, action in choices.items()}
            for key, action in choices.items():
                new, vehicle, customer = env.decode_action(action)
                self.last_diagnostics[f'{key}_depot'] = bool((~new & (customer == 0))[0])
            self.last_diagnostics['valid_group_customers'] = int(count.sum())
        return p


def configure_scale(model, alpha):
    model.fusion_alpha = alpha


if __name__ == '__main__':
    experiment.OUT = experiment.suite.ROOT/'result/kim_group_residual_20261111'
    experiment.IMPLEMENTATION = ('Original AMP q, float64 alpha=.25 beta=5. Subtract current valid anchor-group customer mean only from group members; depot/non-group zero. Signed residual can suppress group customers. Not a general depot-offset removal. Same-state counterfactuals are on residual trajectories.')
    experiment.BASELINE_ROOT = experiment.suite.ROOT/'result/kim_fusion_stable_20261110'
    experiment.BASELINE_GUIDED_VARIANT = 'alpha025'
    experiment.configure_scale = configure_scale
    experiment.VARIANTS = [('group_residual', .25, 'original', 5.)]
    experiment.suite.GuidedPolicy = ResidualPolicy
    try:
        experiment.main()
    except Exception as error:
        experiment.write(experiment.OUT/'status.json', dict(stage='failed', error=repr(error)))
        raise
