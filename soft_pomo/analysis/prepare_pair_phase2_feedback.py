"""Train-only checkpoint500 execution feedback; no training or test inference."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import torch
from experiments.concurrent_fleet_bias_guided.support import ROOT, read, load
from experiments.concurrent_fleet_bias_guided.llm_retry import RetryingClient
from experiments.concurrent_fleet_pair.env import TensorFleetEnv
from experiments.concurrent_fleet_pair.policy import FleetPolicy
from experiments.concurrent_fleet_pair.io import write
from experiments.concurrent_fleet_v9_guided.prompts import build_test_request, parse_test_response, TEST
from experiments.concurrent_fleet_guided.prompts import PUBLIC_FIELDS
from analysis.pair_kim_rl_example import as_list, PURE, OLD

ART = ROOT / 'result/pair300_phase2_feedback500_cache_20261028'
LATE = 10
OFFLINE = '''
This is OFFLINE TRAIN guidance for the target training instance, not causal test inference.
target_train_execution contains one complete episode from a frozen Pure RL checkpoint500.
It is a non-optimal reference, not a target to copy, and not an execution under these
confidence scores. For ACC it includes both pre- and post-disclosure execution: overall
performance is NOT post-accident-only performance or a no-event counterfactual.
Use route membership, visit order, measured performance and the weighted objective together.
No counterfactual improvement has been measured. Do not invent arrival times or outcomes.
Reference vehicle IDs are not future assignments. Preserve the fixed instance groups;
the reference routes need not match them. No live fleet snapshot is supplied.
prior_base_confidence is the previous no-event LLM assessment, NOT measured ground truth.
For ACC this is also the pre-disclosure confidence. Revise it in light of event exposure
and execution evidence. Post-disclosure confidence activates only at event disclosure.
Training event reports supplied here are offline evidence, not claims they are public at t=0.
Return the same customer-keyed confidence schema. Do not output routes or group changes.
'''


def request(inst, params, groups, reports, template, base=None, reference=None):
    req = build_test_request(inst, params, groups, reports, template=template)
    text = req['messages'][0]['content'].replace(
        'The environment designates a ready\nvehicle and RL selects its next destination.',
        'RL selects an eligible existing-vehicle/destination pair after initial dispatch.')
    ctx = json.loads(req['messages'][1]['content'])
    if reference is not None:
        assert reference['name'] == inst['name']
        text = text.replace(TEST, OFFLINE)
        ctx['stage'] = 'offline_train_post_disclosure' if reports[0]['event_type'] == 'ACCIDENT' else 'offline_train_rain'
        ctx['public_events'] = [{k:v for k,v in r.items() if k in PUBLIC_FIELDS} for r in reports]
        ctx['target_train_execution'] = reference
        ctx['prior_base_confidence'] = base
    else:
        assert not reports and not inst['preset_events']
        ctx['stage'] = 'event_free_base'
    req['messages'] = [dict(role='system', content=text), dict(role='user', content=json.dumps(ctx, allow_nan=False))]
    return req


@torch.no_grad()
def extract(model, inst, reports, params, checkpoint_hash):
    env = TensorFleetEnv(inst, 1, params, 'cpu')
    starts, encoded = model.encode_events(inst, reports)
    routes = {}
    for _ in range(600):
        if bool(env.done.all()):
            break
        a = model.probabilities(env.frame(), starts, encoded, env).argmax(1)
        _, vehicle, dest = env.decode_action(a)
        routes.setdefault(int(vehicle[0]), []).append(int(dest[0]))
        env.step(a, validate=True)
    assert bool(env.done.all()), 'Incomplete reference'
    assert sorted(n for r in routes.values() for n in r if n) == list(range(1, 101))
    metrics = {k:float(v[0]) for k,v in env.metrics().items()}
    assert abs(metrics['reward'] + sum(v for k,v in metrics.items() if k.startswith('cost_'))) < 1e-4
    return dict(name=inst['name'], checkpoint_epoch=500, checkpoint_sha256=checkpoint_hash,
                decoder='CPU FP32 single greedy rollout; no augmentation or search; epsilon=0',
                routes=[dict(vehicle=k, customers=v) for k,v in routes.items()], metrics=metrics,
                scope='complete event episode; not no-event performance')


def main(allow_api=False):
    torch.set_num_threads(2)
    ART.mkdir(parents=True, exist_ok=True)
    for name in ('groups.json', 'split.json', 'request_template.json'):
        if (ART/name).exists():
            assert (ART/name).read_bytes() == (OLD/name).read_bytes()
        else:
            shutil.copy2(OLD/name, ART/name)
    cfg = read(PURE/'protocol.json')['configs'][str(LATE)]
    write(ART/'config.json', cfg)
    checkpoint = PURE/f'late{LATE}/checkpoint-500.pt'
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    protocol = dict(checkpoint=str(checkpoint), checkpoint_sha256=digest, reference_epoch=500,
        groups='fixed instance groups shared with Kim; not target execution groups',
        initial_acc='instance-only shared BASE, no future execution',
        post_acc='previous BASE plus target full-episode RL routes and metrics',
        rain='BASE plus target RL routes and metrics',
        test='no target RL execution; initial instance-only; causal live accident refresh',
        training_started=False, model=read(ART/'request_template.json')['model'])
    if (ART/'protocol.json').exists():
        assert read(ART/'protocol.json') == protocol
    write(ART/'protocol.json', protocol)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = FleetPolicy(**cfg['model']).cpu().eval()
    model.load_state_dict(saved['model_state_dict'], strict=True)
    groups, split, template = (read(ART/n) for n in ('groups.json', 'split.json', 'request_template.json'))
    assert not ({x['name'] for x in split['train']} & {x['name'] for x in split['test']})
    for row in split['train']:
        path = ART/'train_references'/(row['name']+'.json')
        if not path.exists():
            inst, reports = load(row)
            write(path, extract(model, inst, reports, cfg['environment'], digest))
        assert read(path)['checkpoint_sha256'] == digest
        print('REFERENCE', row['name'], flush=True)
    if not allow_api:
        return
    def call(req, name):
        write(ART/'requests'/(name+'.json'), req)
        client = RetryingClient(ART/'calls'/name, lambda r, v: parse_test_response(v, groups, list(range(1,101))), allow_paid=True)
        return as_list(client(req), groups)
    basepath = ART/'base_confidence.json'
    if not basepath.exists():
        inst, _ = load(split['train'][0]);inst = copy.deepcopy(inst)
        inst['preset_events'] = [];inst['name'] = 'RC101_BASE_PHASE2'
        write(basepath, call(request(inst, cfg['environment'], groups, [], template), 'BASE'))
    base = read(basepath)
    parse_test_response(base, groups, list(range(1,101)))
    for row in split['train']:
        path = ART/'train_cache'/(row['name']+'.json')
        inst, reports = load(row)
        acc = reports[0]['event_type'] == 'ACCIDENT'
        if not path.exists():
            req = request(inst, cfg['environment'], groups, reports, template, base,
                          read(ART/'train_references'/(row['name']+'.json')))
            scores = call(req, row['name'])
            write(path, dict(initial=base if acc else scores, post_disclosure=scores if acc else None,
                             revealed_at_min=reports[0]['revealed_at_min']))
        cached = read(path)
        parse_test_response(cached['initial'], groups, list(range(1,101)))
        if acc:
            assert cached['initial'] == base
            parse_test_response(cached['post_disclosure'], groups, list(range(1,101)))
        print('CACHE', row['name'], flush=True)
    write(ART/'complete.json', dict(train_count=len(split['train']), base_count=1, reference_epoch=500,
                                   test_cache_generated=False, training_started=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow-api', action='store_true')
    args = parser.parse_args()
    try:
        main(args.allow_api)
    except Exception as error:
        write(ART/'failure.json', dict(error=repr(error)))
        raise
