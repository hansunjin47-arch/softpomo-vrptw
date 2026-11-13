"""Causal test caches and isolated checkpoint500-to1000 phase-two runner."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch
from analysis.prepare_pair_phase2_feedback import ART, PURE, read, load
from experiments.concurrent_fleet_pair.io import write
from experiments.concurrent_fleet_pair_kim import run as adapter
from experiments.concurrent_fleet_v9_guided.prompts import build_test_request, parse_test_response
from experiments.concurrent_fleet_bias_guided.llm_retry import RetryingClient

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'result/pair300_phase2_feedback500_20261028'
LATE = 10
VEHICLE_COST = 300


def digest(request):
    return hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()


def test_request(*args, **kwargs):
    req = build_test_request(*args, **kwargs)
    ctx = json.loads(req['messages'][1]['content'])
    assert 'reference_routes' not in ctx and 'target_train_execution' not in ctx
    ctx['prior_base_confidence'] = read(ART/'base_confidence.json')
    text = req['messages'][0]['content'].replace(
        'The environment designates a ready\nvehicle and RL selects its next destination.',
        'RL selects an eligible existing-vehicle/destination pair after initial dispatch.')
    text += ('\nprior_base_confidence is a no-event instance-only LLM assessment, not measured '
             'performance. Revise it using only currently disclosed reports. At refresh, '
             'also use previous_confidence and the actual observed fleet state. '
             'No target or separate RL reference is supplied at test.\n')
    req['messages'] = [dict(role='system', content=text),
                       dict(role='user', content=json.dumps(ctx, allow_nan=False))]
    return req


def validate_artifacts():
    assert read(ART/'complete.json')['train_count'] == 20
    split, groups = read(ART/'split.json'), read(ART/'groups.json')
    assert split == read(adapter.train.SOURCE/'split.json')
    for row in split['train']:
        cache = read(ART/'train_cache'/(row['name']+'.json'))
        parse_test_response(cache['initial'], groups, list(range(1,101)))
        if '_ACC_' in row['name']:
            assert cache['initial'] == read(ART/'base_confidence.json')
            parse_test_response(cache['post_disclosure'], groups, list(range(1,101)))
    return split, groups


def prepare_test():
    split, groups = validate_artifacts()
    params = read(ART/'config.json')['environment']
    template = read(ART/'request_template.json')
    manifest = []
    for row in split['test']:
        inst, reports = load(row)
        req = test_request(inst, params, groups, reports, template=template)
        ctx = json.loads(req['messages'][1]['content'])
        assert len(ctx['selectable_customers']) == 100 and 'fleet' not in ctx
        assert all(e['revealed_at_min'] <= 0 for e in ctx['public_events'])
        key = digest(req)
        path = ART/'test_initial_cache'/(key+'.json')
        if path.exists():
            item = read(path)
            assert item['request'] == req
            payload = item['response']
        else:
            # With no disclosed event, reuse the identical fixed-instance BASE table.
            if not ctx['public_events']:
                payload = read(ART/'base_confidence.json')
                source = 'shared instance-only BASE; no future accident data'
            else:
                client = RetryingClient(ART/'tc'/str(len(manifest)),
                    lambda r,v:parse_test_response(v, groups, list(range(1,101))), allow_paid=True)
                payload = client(req)
                source = 'GPT causal initial RAIN'
            write(path, dict(request=req, response=payload, source=source))
        parse_test_response(payload, groups, list(range(1,101)))
        manifest.append(dict(name=row['name'], request_hash=key))
        print('TEST_INITIAL', row['name'], flush=True)
    write(ART/'test_initial_manifest.json', manifest)
    write(ART/'test_ready.json', dict(instances=len(manifest), initial_only=True,
          accident_refresh='online at disclosure using actual current fleet; never precomputed'))


class TestClient:
    def __init__(self, folder, validate, *, allow_paid=True):
        self.validate = validate
        self.live = RetryingClient(folder, validate, allow_paid=allow_paid)

    def __call__(self, req):
        ctx = json.loads(req['messages'][1]['content'])
        if 'fleet' not in ctx:
            cached = read(ART/'test_initial_cache'/(digest(req)+'.json'))
            assert cached['request'] == req
            payload = cached['response']
        else:
            clock = ctx['fleet']['clock_min']
            assert clock > 0 and 'previous_confidence' in ctx
            assert all(e['revealed_at_min'] <= clock for e in ctx['public_events'])
            payload = self.live(req)
        self.validate(req, payload)
        return payload


def run():
    validate_artifacts()
    assert read(ART/'test_ready.json')['instances'] == 80
    current = read(PURE/'protocol.json')['configs'][str(LATE)]
    current['environment']['vehicle_penalty'] = VEHICLE_COST * current['environment']['problem_size'] / 1696.95
    current['environment']['normalize_advantage_std'] = True
    assert current == read(ART/'config.json'), 'Trainer config differs from reference experiment'
    checkpoint = PURE/f'late{LATE}/checkpoint-500.pt'
    expected = read(ART/'protocol.json')['checkpoint_sha256']
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == expected
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert saved['epoch'] == 500 and saved['state']['epoch'] == 501 and saved['state']['episodes'] == 0
    assert saved['state']['update'] == 10000
    original_train = adapter.train.run

    def training(args):
        args.config = ART/'config.json'
        resume = OUT/'initial_checkpoint'
        (resume/f'late{LATE}').mkdir(parents=True)
        shutil.copy2(checkpoint, resume/f'late{LATE}/checkpoint-last.pt')
        args.resume_from = resume
        assert args.vehicle_cost == VEHICLE_COST and args.weights == [LATE] and not args.no_advantage_std
        protocol = read(OUT/'kim_protocol.json')
        protocol.update(initialization='Pure RL checkpoint500; restore model, optimizer, scaler and RNG; epochs501-1000',
            target_train_execution=True, checkpoint_sha256=expected,
            test='80 cached causal initial tables; online causal ACC refresh; no test RL reference')
        write(OUT/'kim_protocol.json', protocol)
        shutil.copy2(__file__, OUT/'source/pair_phase2_feedback_run.py')
        return original_train(args)

    adapter.ARTIFACT = ART
    adapter.audit = validate_artifacts
    adapter.RetryingClient = TestClient
    adapter.online.build_test_request = test_request
    adapter.train.run = training
    adapter.run(OUT, late=LATE, vehicle_cost=VEHICLE_COST)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare-test', 'run'])
    args = parser.parse_args()
    prepare_test() if args.stage == 'prepare-test' else run()
