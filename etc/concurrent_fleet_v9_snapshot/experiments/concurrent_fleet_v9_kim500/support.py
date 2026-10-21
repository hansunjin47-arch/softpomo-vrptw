"""Vehicle500 Kim artifacts and disclosure-gated training guidance."""
import json
from pathlib import Path
from experiments.concurrent_fleet_v3.run import ROOT, load as original_load
from experiments.concurrent_fleet_v9_guided.env import GuidedFleetEnv
from experiments.concurrent_fleet_v9_guided.prompts import parse_response
from experiments.concurrent_fleet_v9_guided.online import OpenRouterClient, evaluate_instance

OUT = ROOT / 'result/v9_kim500_late10_gpt_20261020'
BETA = 5.


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load(row):
    inst, public = original_load(row)
    inst['name'] = row['name']
    return inst, public


class CachedTrainEnv(GuidedFleetEnv):
    def __init__(self, inst, count, params, device='cuda'):
        cache = read(OUT / 'train_cache' / (inst['name']+'.json'))
        groups = read(OUT / 'groups.json')
        super().__init__(inst, count, params, device, groups=groups,
                         scores=parse_response(cache['initial'], groups), beta=BETA)
        if cache.get('post_disclosure') is not None:
            self.update_event_confidence(parse_response(cache['post_disclosure'], groups),
                revealed_at_min=cache['revealed_at_min'], effective_time_min=cache['revealed_at_min'])


def test_one(model, inst, public, params, amp, folder):
    return evaluate_instance(model, inst, params, read(OUT/'groups.json'), public,
        template=read(OUT/'request_template.json'), client=OpenRouterClient(ROOT/'result/v9kim500_calls_20261020/test', allow_paid=True),
        beta=BETA, group_source='fixed_instance', out=ROOT/'result/v9kim500_test_20261020'/inst['name'], amp=amp)
