"""Separate diagnostic Test80 evaluation of the ten-epoch continuation."""
import hashlib
import json
import os
import statistics
import torch
from analysis import evaluate_r101_mask_kim_20261109 as original
from analysis.pair_kim_rl_example import enrich
from experiments.concurrent_fleet_pair_kim.run import GuidedPolicy, OnlineEnv
from experiments.concurrent_fleet_v9_guided import online
from experiments.concurrent_fleet_v9_guided.prompts import parse_test_response
from experiments.concurrent_fleet_bias_guided.llm_retry import RetryingClient
from experiments.concurrent_fleet_bias_guided.support import ROOT, read, load
from experiments.concurrent_fleet_pair.io import write
from experiments.concurrent_fleet_v3 import run as data_module

OUT = ROOT/'result/r101_kim44_epoch1010_test_20261109'


def main():
    torch.set_num_threads(1)
    original.validate()
    assert os.environ.get('OPENROUTER_API_KEY'), 'Missing API key'
    checkpoint = ROOT/'result/r101_kim44_continuation10_20261109/checkpoint-1010.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert saved['epoch'] == 1010
    run = original.RECEIVED/'runs/seed44'
    cfg = read(run/'protocol.json')['configs']['5']
    OUT.mkdir(exist_ok=True)
    protocol = dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        epoch=1010, seed=44, vehicle_weight=100, late=5, mask=True, test_count=80,
        decode='single greedy; AMP; no augmentation', beta=5,
        initial='reuse exact original initial requests; read-only cache',
        accident='causal live refresh with actual current vehicle state',
        purpose='diagnostic only; extra training budget; original five-seed results not replaced')
    if (OUT/'protocol.json').exists():
        assert read(OUT/'protocol.json') == protocol
    else:
        write(OUT/'protocol.json', protocol)
    data_module.SOURCE = original.BUNDLE/'dataset'
    art = original.BUNDLE/'artifacts'
    groups, template = read(art/'groups.json'), read(art/'request_template.json')
    base, example = read(art/'base_confidence.json'), read(art/'reference.json')
    build = online.build_test_request
    initial = True

    def request(*args, **kwargs):
        nonlocal initial
        initial = kwargs.get('env') is None
        return enrich(build(*args, **kwargs), base, example)

    online.build_test_request = request
    online.OnlineTestEnv = OnlineEnv

    def validate(req, payload):
        parse_test_response(payload, groups, json.loads(req['messages'][1]['content'])['selectable_customers'])

    cached = RetryingClient(original.OUT/'calls/shared_initial', validate, allow_paid=False)
    live = RetryingClient(OUT/'calls/live', validate, allow_paid=True)

    def client(req):
        return cached(req) if initial else live(req)

    model = GuidedPolicy(**cfg['model']).cuda().eval()
    model.load_state_dict(saved['model_state_dict'])
    del saved
    results = []
    for row in read(run/'split.json')['test']:
        folder = OUT/'instances'/row['name']
        write(OUT/'status.json', dict(stage='evaluating', completed=len(results), target=80, name=row['name']))
        if (folder/'result.json').exists():
            result = read(folder/'result.json')
        else:
            inst, reports = load(row)
            result = online.evaluate_instance(model, inst, cfg['environment'], groups, reports,
                template=template, client=client, beta=5., out=folder, amp=True)
        results.append(dict(name=row['name'], **result['metrics']))
        write(OUT/'results.json', results)
        print('TEST', len(results), row['name'], result['metrics']['reward'], flush=True)
    means = {k: statistics.mean(r[k] for r in results)
             for k in ('reward', 'vehicles', 'travel_time', 'total_late', 'unserved')}
    write(OUT/'summary.json', means)
    write(OUT/'status.json', dict(stage='complete', epoch=1010, test_count=80, mean=means))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        write(OUT/'status.json', dict(stage='failed', error=repr(error)))
        raise
