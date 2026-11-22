"""Fixed-example Kim cache generation and isolated pair-training adapter."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil

import torch
from experiments.concurrent_fleet_pair_kim import run as adapter
from experiments.concurrent_fleet_bias_guided.support import ROOT, read, load
from experiments.concurrent_fleet_pair.io import write
from experiments.concurrent_fleet_pair.env import TensorFleetEnv
from experiments.concurrent_fleet_pair.policy import FleetPolicy
from experiments.concurrent_fleet_v9_guided.prompts import build_test_request, parse_test_response
from experiments.concurrent_fleet_bias_guided.llm_retry import RetryingClient

PURE = ROOT/'result/pair300_epsilon005_floor001_std_20261026'
OUT = ROOT/'result/pair300_epsilon005_floor001_std_kim_example_20261027'
ART = ROOT/'result/pair_kim_rl_example_cache_20261027'
OLD = adapter.ARTIFACT
EXAMPLE = 'RC101_RAIN_REFERENCE_PAIR_20261027'
LATE = 10
VEHICLE_COST = 300.


def enrich(request, base, example, *, offline=False, new_head_beta=None):
    request = copy.deepcopy(request)
    context = json.loads(request['messages'][1]['content'])
    context['rl_reference_example'] = example
    if base is not None:
        context['prior_base_confidence'] = base
    text = request['messages'][0]['content']
    text = text.replace('The environment designates a ready\nvehicle and RL selects its next destination.',
                        'During initial dispatch, RL chooses NEW(customer) or an eligible '
                        'existing-vehicle/destination pair. NEW opens a new vehicle; selecting '
                        'an existing pair closes initial dispatch and commits that destination '
                        'for the selected vehicle. Dispatch also closes automatically when '
                        'the fleet limit is reached or no unassigned customers remain. '
                        'After dispatch closes, the environment designates one ready vehicle '
                        'and RL selects only its next destination; RL cannot choose a different '
                        'vehicle at that decision.')
    if new_head_beta is not None:
        new_head_beta = float(new_head_beta)
        if not math.isfinite(new_head_beta) or new_head_beta < 0:
            raise ValueError('new_head_beta must be finite and nonnegative')
        old = ('Other-group customers remain selectable with zero bias. Depot, NEW and STOP have no\n'
               'direct bias. No depot score or depot-relative negative-score rule is used.')
        if old not in text:
            raise ValueError('Unrecognized guidance description; cannot safely describe NEW bias')
        text = text.replace(old, (
            'Other-group customers remain selectable with zero routing bias. Depot has no direct bias.\n'
            'During initial dispatch only, each group nominates its highest-confidence customer\n'
            'from the currently disclosed confidence table (ties follow supplied group order).\n'
            f'The NEW(customer) action for that nominee receives a constant +{new_head_beta:g} bias,\n'
            'not new_head_beta times its confidence. Thus your highest score in each group also\n'
            'influences which customer starts a new vehicle. Consider first-visit suitability as\n'
            'well as subsequent within-group preferences; this is not a forced first visit or\n'
            'a requirement to dispatch one vehicle per group. Existing-vehicle actions receive\n'
            'no confidence bias while dispatch is open. Masked nominees remain unavailable;\n'
            'the next-highest customer is not automatically promoted when a nominee is assigned.\n'
            'The NEW bias disappears when dispatch closes. Subsequent choices then use only\n'
            'beta times confidence for the vehicle first customer\'s anchor group, as above.\n'
            'Neither bias overrides feasibility masks or existing commitments. No depot score\n'
            'or depot-relative negative-score rule is used.'))
        context['guidance_application'] = dict(
            version='max_confidence_new_head_v1', new_head_beta=new_head_beta,
            routing_beta=context.get('beta', 5.),
            head_selection='group maximum; ties follow group order; no promotion after masking',
            new_bias_phase='initial_dispatch_only', routing_bias_phase='after_dispatch_only')
    text += ('\nThe rl_reference_example is a separate fixed case, not this target. Its complete '
             'route and measured performance come from one non-optimal Pure RL greedy rollout. '
             'It is neither a correct answer nor a confidence label. Do not copy its event, '
             'customer priorities or route into the target. No target reference execution is supplied.\n')
    if base is None:
        text += 'The target is event-free BASE. Ignore the example event when assessing target scores.\n'
    else:
        text += ('prior_base_confidence is YOUR previous no-event assessment of these fixed groups, '
                 'not measured performance or ground truth. Assess target event exposure and timing '
                 'and revise that baseline where justified; do not force changes. For live refresh, '
                 'also use previous_confidence and the actual fleet snapshot; do not change commitments.\n')
    if offline:
        text += ('This is OFFLINE TRAINING CACHE generation. Public_events here are known training '
                 'reports for the indicated stage, not a claim of disclosure at time zero. '
                 'Post-accident scores are gated until disclosure. No live fleet state is supplied.\n')
        context['stage'] = 'offline_training_event'
    request['messages'] = [dict(role='system', content=text),
                           dict(role='user', content=json.dumps(context, separators=(',', ':'), allow_nan=False))]
    return request


def as_list(payload, groups):
    scores = parse_test_response(payload, groups, list(range(1, 101)))
    return {'confidence': [dict(cluster=k+1, customer=n, score=v) for k, row in scores.items() for n, v in row.items()]}


def reference():
    split = read(OLD/'split.json')
    inst, reports = load(split['train'][0])
    inst, reports = copy.deepcopy(inst), copy.deepcopy(reports)
    inst['name'] = EXAMPLE
    inst['preset_events'][0].update(trigger_time=123.456, duration=44.321)
    reports[0].update(start_min=123.456, end_min=167.777, revealed_at_min=0.)
    # Remove the old textual time interval rather than presenting contradictory facts.
    reports[0].pop('news_text', None)
    for row in split['train'] + split['test']:
        other, _ = load(row)
        assert row['name'] != EXAMPLE and other['preset_events'] != inst['preset_events']
    config = read(PURE/'protocol.json')['configs'][str(LATE)]
    checkpoint = PURE/f'late{LATE}/checkpoint-1000.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = FleetPolicy(**config['model']).cpu().eval()
    model.load_state_dict(saved['model_state_dict'], strict=True)
    env = TensorFleetEnv(inst, 1, config['environment'], 'cpu')
    routes = {}
    with torch.no_grad():
        starts, encodings = model.encode_events(inst, reports)
        for _ in range(600):
            if bool(env.done.all()):
                break
            action = model.probabilities(env.frame(), starts, encodings, env).argmax(1)
            _, vehicle, dest = env.decode_action(action)
            routes.setdefault(int(vehicle[0]), []).append(int(dest[0]))
            env.step(action, validate=True)
        else:
            raise RuntimeError('Reference exceeded action limit')
    assert sorted(n for route in routes.values() for n in route if n) == list(range(1, 101))
    metrics = {k:float(v[0]) for k,v in env.metrics().items()}
    template = read(OLD/'request_template.json')
    req = build_test_request(inst, config['environment'], read(OLD/'groups.json'), reports, template=template)
    context = json.loads(req['messages'][1]['content'])
    fixed = copy.deepcopy(context['fixed_instance'])
    fixed.pop('groups', None)
    value = dict(scenario=EXAMPLE, fixed_instance=fixed, public_events=context['public_events'],
                 routes=list(routes.values()), observed_performance=metrics,
                 objective_coefficients=context['objective_coefficients'], checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                 checkpoint_epoch=1000, decoder='CPU single greedy rollout; no search; fixed event before decoding')
    write(ART/'reference.json', value)
    return value


def prepare():
    ART.mkdir(parents=True, exist_ok=True)
    for name in ('groups.json', 'split.json', 'request_template.json'):
        source = OLD/name
        if (ART/name).exists():
            assert (ART/name).read_bytes() == source.read_bytes()
        else:
            shutil.copy2(source, ART/name)
    config = read(PURE/'protocol.json')['configs'][str(LATE)]
    assert config['environment']['late_penalty'] == LATE
    assert abs(config['environment']['vehicle_penalty'] * 1696.95 / 100 - VEHICLE_COST) < 1e-6
    if (ART/'config.json').exists():
        assert read(ART/'config.json') == config
    else:
        write(ART/'config.json', config)
    example = read(ART/'reference.json') if (ART/'reference.json').exists() else reference()
    groups, split = read(ART/'groups.json'), read(ART/'split.json')
    params = config['environment']
    template = read(ART/'request_template.json')
    client = RetryingClient(ART/'calls', lambda req, payload: parse_test_response(
        payload, groups, json.loads(req['messages'][1]['content'])['selectable_customers']), allow_paid=True)
    inst, _ = load(split['train'][0])
    inst = copy.deepcopy(inst)
    inst['preset_events'] = []
    inst['name'] = 'RC101_BASE_REFERENCE'
    base_path = ART/'base_confidence.json'
    if base_path.exists():
        base = read(base_path)
    else:
        req = enrich(build_test_request(inst, params, groups, [], template=template), None, example)
        base = as_list(client(req), groups)
        write(base_path, base)
    for row in split['train']:
        path = ART/'train_cache'/(row['name']+'.json')
        if path.exists():
            continue
        inst, reports = load(row)
        req = build_test_request(inst, params, groups, reports, template=template)
        context = json.loads(req['messages'][1]['content'])
        from experiments.concurrent_fleet_guided.prompts import PUBLIC_FIELDS
        context['public_events'] = [{k:v for k,v in report.items() if k in PUBLIC_FIELDS} for report in reports]
        req['messages'][1]['content'] = json.dumps(context)
        req = enrich(req, base, example, offline=True)
        event = as_list(client(req), groups)
        accident = reports[0]['event_type'] == 'ACCIDENT'
        write(path, dict(initial=base if accident else event,
                         post_disclosure=event if accident else None,
                         revealed_at_min=reports[0]['revealed_at_min']))
        print('CACHE', row['name'], flush=True)
    write(ART/'reuse_audit.json', dict(rl_reference=False, separate_rl_example=True,
         explanation='No target RL reference; fixed independent Pure RL example only'))
    adapter.ARTIFACT = ART
    adapter.audit()
    write(ART/'complete.json', dict(train_count=20, base_count=1, test='online causal requests'))


def run():
    assert read(ART/'complete.json')['train_count'] == 20
    adapter.ARTIFACT = ART
    example, base = read(ART/'reference.json'), read(ART/'base_confidence.json')
    original_request = adapter.online.build_test_request
    def request(*args, **kwargs):
        return enrich(original_request(*args, **kwargs), base, example)
    adapter.online.build_test_request = request
    original_train = adapter.train.run
    def training(args):
        args.config = ART/'config.json'
        path = OUT/'kim_protocol.json'
        protocol = read(path)
        protocol.update(test='Single greedy trajectory; fixed separate RL example plus BASE; causal live accident refresh',
                        reference=str(ART/'reference.json'),
                        new_method_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        write(path, protocol)
        shutil.copy2(__file__, OUT/'source/pair_kim_rl_example.py')
        return original_train(args)
    adapter.train.run = training
    adapter.run(OUT, late=LATE, vehicle_cost=VEHICLE_COST)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'run'])
    parser.add_argument('--allow-api', action='store_true', required=True)
    parser.add_argument('--late', type=int, default=LATE)
    parser.add_argument('--vehicle-cost', type=float, default=VEHICLE_COST)
    parser.add_argument('--pure', type=Path, default=PURE)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--art', type=Path, default=ART)
    args = parser.parse_args()
    LATE, VEHICLE_COST = args.late, args.vehicle_cost
    PURE, OUT, ART = args.pure.resolve(), args.out.resolve(), args.art.resolve()
    prepare() if args.stage == 'prepare' else run()
