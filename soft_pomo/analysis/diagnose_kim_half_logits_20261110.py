"""Inference-only half-logit control using the frozen Train20 ablation harness."""
from analysis import diagnose_kim_guidance_suite_20261110 as suite
from experiments.concurrent_fleet_pair_kim.run import GuidedPolicy
from experiments.concurrent_fleet_bias_guided.support import ROOT
from experiments.concurrent_fleet_pair.io import write

OUT = ROOT/'result/kim_half_logits_20261110'


class HalfLogitPolicy(GuidedPolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.decoder.logit_clipping == 10.
        self.decoder.logit_clipping = 5.


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    write(OUT/'intervention.json', dict(alpha=.5, original_logit_clipping=10.,
        inference_logit_clipping=5., beta_values=[0.,5.], training=False,
        role='Train20 diagnostic, not Test selection; original learned parameters unchanged',
        comparator='result/kim_guidance_suite_20261110 original and off variants'))
    suite.OUT = OUT
    suite.VARIANTS = [('off', 0.), ('original', 5.)]
    suite.GuidedPolicy = HalfLogitPolicy
    try:
        suite.main()
    except Exception as error:
        write(OUT/'status.json', dict(stage='failed', error=repr(error)))
        raise
