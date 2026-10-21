"""Evaluate pure500, prepare train-only route feedback and paid GPT caches."""
import argparse
import copy
import hashlib
import json
import torch
from experiments.concurrent_fleet_v3.run import SOURCE
from experiments.concurrent_fleet_pair.io import write
from experiments.concurrent_fleet_v9.env import TensorFleetEnv
from experiments.concurrent_fleet_v9.policy import FleetPolicy
from experiments.concurrent_fleet_v9_guided.prompts import build_train_request, parse_response
from experiments.concurrent_fleet_v9_guided.online import OpenRouterClient
from experiments.concurrent_fleet_v9_phase2.support import OUT, PURE, ROOT, read, load, BETA


@torch.no_grad()
def reference(model, inst, public, params):
    env = TensorFleetEnv(inst, 1, params)
    starts, encoded = model.encode_events(inst, public)
    routes = {}; steps = []
    for _ in range(600):
        if bool(env.done.all()):
            break
        with torch.autocast('cuda'):
            action = model.probabilities(env.frame(), starts, encoded, env).argmax(1)
        a = int(action[0]); new = a < env.n; stop = a == 2*env.n+1
        vehicle = int(env.k[0]) if new else int(env.next_vehicle()[0][0])
        dest = a+1 if new else a-env.n
        clock = float(env.time[0])
        env.step(action, validate=True)
        if not stop:
            routes.setdefault(vehicle, []).append(dest)
            steps.append(dict(vehicle=vehicle, destination=dest, departure_min=clock,
                arrival_min=float(env.arrival[0, vehicle]), service_end_min=float(env.available[0, vehicle])))
    else:
        raise RuntimeError('Reference exceeded action limit')
    return dict(routes=list(routes.values()), steps=steps,
                metrics={k:float(v[0]) for k,v in env.metrics().items()})


def main(paid=False, limit=None):
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint = PURE/'late10/checkpoint-500.pt'
    saved = torch.load(checkpoint, weights_only=False)
    assert saved['epoch'] == 500 and saved['state']['epoch'] == 501
    config = read(PURE/'protocol.json')['configs']['10']
    model = FleetPolicy(**config['model']).cuda().eval()
    model.load_state_dict(saved['model_state_dict'], strict=True)
    split = read(SOURCE/'split.json')
    write(OUT/'config.json', config); write(OUT/'split.json', split)
    template = read(ROOT/'result/rc101_phase2_online_gpt_train20_v2_2_20261018/request_template.json')
    write(OUT/'request_template.json', template)
    write(OUT/'preparation_protocol.json', dict(checkpoint=str(checkpoint),
        sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), beta=BETA,
        groups='Event-free pure500 greedy routes on first TRAIN instance, shared with test',
        reference='Single greedy train rollout; never test feedback', model=template['model']))
    results = []
    for i,row in enumerate(split['test']):
        path = OUT/'pure500_test80'/ (row['name']+'.json')
        if not path.exists():
            inst, public = load(row)
            write(path, reference(model, inst, public, config['environment']))
        results.append(dict(name=row['name'], **read(path)['metrics']))
        write(OUT/'status.json', dict(stage='pure500_evaluation', completed=i+1, total=80))
    write(OUT/'pure500_test80/results.json', results)
    inst, _ = load(split['train'][0])
    base = copy.deepcopy(inst); base['preset_events'] = []
    path = OUT/'event_free_train_reference.json'
    if not path.exists(): write(path, reference(model, base, [], config['environment']))
    base_ref = read(path)
    groups = [[n for n in route if n] for route in base_ref['routes']]
    assert sorted(n for g in groups for n in g) == list(range(1,101))
    write(OUT/'groups.json', groups)
    client = OpenRouterClient(ROOT/'result/v9p2_calls_20261020/train', allow_paid=paid)
    selected = split['train'] if limit is None else split['train'][:limit]
    for i,row in enumerate(selected):
        inst, public = load(row)
        future = [e for e in public if e['revealed_at_min'] > 0]
        path = OUT/'train_references'/(row['name']+'.json')
        if not path.exists(): write(path, reference(model, inst, public, config['environment']))
        ref = read(path)
        initial_ref = base_ref if future else ref
        requests = {'initial': build_train_request(inst, config['environment'], groups,
            [e for e in public if e['revealed_at_min'] <= 0], template=template,
            training_reference=initial_ref, reference_id='pure500-event-free' if future else row['name'],
            cache_stage='initial', beta=BETA)}
        if future:
            assert len(future) == 1
            requests['post_disclosure'] = build_train_request(inst, config['environment'], groups, public,
                template=template, training_reference=ref, reference_id=row['name'], beta=BETA)
        cache = dict(revealed_at_min=future[0]['revealed_at_min'] if future else None)
        for stage, request in requests.items():
            write(OUT/'requests'/row['name']/(stage+'.json'), request)
            if paid:
                payload = client(request); parse_response(payload, groups)
                cache[stage] = payload
        if paid: write(OUT/'train_cache'/(row['name']+'.json'), cache)
        write(OUT/'status.json', dict(stage='train_cache' if paid else 'requests_prepared', completed=i+1, total=20))
        print(row['name'], 'cached' if paid else 'prepared', flush=True)
    write(OUT/'status.json', dict(stage=('train_cache_complete' if len(selected)==20 else 'sample_complete') if paid else 'requests_ready', completed=len(selected), total=20))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--paid', action='store_true')
    parser.add_argument('--limit', type=int, choices=range(1,21))
    args=parser.parse_args()
    main(args.paid,args.limit)
