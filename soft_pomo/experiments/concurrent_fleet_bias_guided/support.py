import json
from pathlib import Path
import torch
from experiments.concurrent_fleet_v3.run import ROOT, load as original_load
from experiments.concurrent_fleet_stop_init.policy import FleetPolicy as BiasPolicy
from experiments.concurrent_fleet_v9_guided.env import GuidedFleetEnv
from experiments.concurrent_fleet_v9_guided.prompts import parse_response

PURE = ROOT/'result/v9_stopbias_entropy1_700_late10_20261021'
OUT = ROOT/'result/v9_bias_guided700_late10_20261022'
KIM = ROOT/'result/v9_kim700_late10_gpt_20261020'
GUIDANCE_BETA = 5.


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def load(row):
    inst, public = original_load(row)
    inst['name'] = row['name']
    return inst, public


class GuidedBiasPolicy(BiasPolicy):
    def _phase_scores(self, encoded, frame, env, dispatch):
        scores = super()._phase_scores(encoded, frame, env, dispatch)
        guidance = getattr(env, 'guidance', None)
        if dispatch or guidance is None or guidance.beta == 0:
            return scores
        bias = guidance.action_bias()[:, env.n:2*env.n+1]
        return scores.float() + torch.cat([bias, bias.new_zeros(env.count, 1)], 1)


def cached_env(artifact):
    groups = read(artifact/'groups.json')
    caches = {p.stem:read(p) for p in (artifact/'train_cache').glob('*.json')}
    class CachedEnv(GuidedFleetEnv):
        def __init__(self, inst, count, params, device='cuda'):
            cache = caches[inst['name']]
            super().__init__(inst,count,params,device,groups=groups,
                             scores=parse_response(cache['initial'],groups),beta=GUIDANCE_BETA)
            if cache.get('post_disclosure') is not None:
                self.update_event_confidence(parse_response(cache['post_disclosure'],groups),
                    revealed_at_min=cache['revealed_at_min'],effective_time_min=cache['revealed_at_min'])
    return CachedEnv
