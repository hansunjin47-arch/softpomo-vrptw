"""Prepare test-only v5 guidance, call the API, and evaluate frozen weights."""
import argparse
import copy
import json
from pathlib import Path
import statistics
import subprocess
import sys

from run_ontology_b128_train20 import ROOT, FROZEN, OUT as TRAIN, CHECKPOINT
from train_cached import read, write, sha, verify_bundle, validate_cache

OUT = TRAIN / 'eval_1000_test80'
FINAL = TRAIN / 'training_fixed/checkpoint-1000.pt'
TEST_ROOT = FROZEN / 'eval_addon'


def immutable(path, value):
    if path.exists():
        assert read(path) == value, f'Existing evidence differs: {path}'
    else:
        write(path, value)


def inputs():
    verify_bundle(FROZEN)
    split = read(FROZEN / 'data/split.json')
    names = [r['name'] for r in split['test']]
    assert len(names) == len(set(names)) == 80
    assert not set(names) & set(read(TRAIN / 'protocol.json')['train_names'])
    for entry in split['test']:
        assert sha(TEST_ROOT / 'data' / (entry['name'] + '.txt')) == entry['sha256']
    return split, names


def prepare():
    import torch
    from train_cached import build_trainer
    split, names = inputs()
    config = copy.deepcopy(read(TRAIN / 'source_config.json'))
    config['trainer'].update(train_instances=names, use_augmentation=True,
                             n_mc_samples=1, pomo_auto=False, eval_top_k=None)
    config['environment']['pomo_size'] = 100
    config['model']['eval_type'] = 'argmax'
    immutable(OUT / 'protocol.json', dict(test_names=names, augmentation=8, pomo=100,
        per_instance_seed='42 + test index', beta=5, reference_checkpoint_sha256=sha(CHECKPOINT),
        final_checkpoint_sha256=sha(FINAL), train_protocol_sha256=sha(TRAIN / 'protocol.json'),
        split_sha256=sha(FROZEN / 'data/split.json'), version='routing_context_ontology_v5'))
    trainer = build_trainer(config, TEST_ROOT, OUT / 'reference_runtime')
    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    assert state['epoch'] == 500
    trainer.model.load_state_dict(state['model_state_dict'], strict=True)
    trainer.model.eval()
    import train_vrptw as pure
    from prepare_ontology_train20 import INSTRUCTIONS, instance_context
    from routing_ontology import context, specification, attach_reference_groups
    from audit_event_relations import analyze
    assert specification() == read(TRAIN / 'ontology.json')
    train_name = read(TRAIN / 'protocol.json')['train_names'][0]
    template = read(TRAIN / 'requests' / train_name / 'request.json')
    old_payload = read(TRAIN / 'requests' / train_name / 'context.json')
    assert template['messages'][0]['content'] == INSTRUCTIONS + '\n' + json.dumps(old_payload, separators=(',', ':'), allow_nan=False)
    assert [i['name'] for i in trainer.instance_pool] == names
    for index, inst in enumerate(trainer.instance_pool):
        folder = OUT / 'requests' / inst['name']
        ref_path = folder / 'reference.json'
        if ref_path.exists():
            ref = read(ref_path)
            assert ref['checkpoint_sha256'] == sha(CHECKPOINT)
        else:
            pure._set_seed(42 + index)
            with torch.no_grad():
                routes, reward = trainer._best_solution(inst)
            clusters = [[int(n) for n in route if n] for route in routes if any(route)]
            assert sorted(n for group in clusters for n in group) == list(range(1, 101))
            ref = dict(clusters=clusters, reward=float(reward), checkpoint_sha256=sha(CHECKPOINT))
            immutable(ref_path, ref)
        obs = TEST_ROOT / 'data' / (inst['name'] + '.observations.json')
        graph, stats = analyze(inst, [[0] + c + [0] for c in ref['clusters']], read(obs))
        payload = context(graph, stats['coords'])
        attach_reference_groups(payload, instance_context(inst, stats['coords'], ref['clusters']))
        payload['reference_scope'] = dict(scenario=inst['name'], checkpoint_sha256=sha(CHECKPOINT),
            reference_file_sha256=sha(ref_path), instance_sha256=sha(TEST_ROOT / 'data' / (inst['name'] + '.txt')),
            public_observation_sha256=sha(obs), schedules=['event_reference', 'same_order_event_removed'],
            applies_to='Only this fixed route and schedule pair, not a live RL trajectory.')
        request = copy.deepcopy(template)
        request['messages'] = [dict(role='user', content=INSTRUCTIONS + '\n' + json.dumps(payload, separators=(',', ':'), allow_nan=False))]
        schema = request['response_format']['json_schema']['schema']['properties']['confidence']
        schema['minItems'] = schema['maxItems'] = 100 + len(ref['clusters'])
        for filename, value in [('request.json', request), ('context.json', payload), ('relations.json', graph)]:
            immutable(folder / filename, value)
        write(OUT / 'status.json', dict(stage='preparing', completed=index + 1, total=80))
        print('PREPARED', index + 1, inst['name'], flush=True)
    immutable(OUT / 'request_hashes.json', {n: sha(OUT / 'requests' / n / 'request.json') for n in names})
    write(OUT / 'status.json', dict(stage='prepared', completed=80))


def call():
    from run_rc101_kim_conversation import decode
    import VRPTWLLMModule as backend
    names = read(OUT / 'protocol.json')['test_names']
    hashes = read(OUT / 'request_hashes.json')
    for index, name in enumerate(names):
        folder = OUT / 'requests' / name
        assert sha(folder / 'request.json') == hashes[name]
        request = read(folder / 'request.json')
        response_path = folder / 'response.txt'
        write(OUT / 'status.json', dict(stage='calling', completed=index, total=80, name=name))
        if not response_path.exists():
            print('CALL', index + 1, name, flush=True)
            response = backend.query_llm(request['messages'][0]['content'], request['model'], response_format=request['response_format'])
            response_path.write_text(response, encoding='utf-8')
        _, cache = decode(response_path.read_text(encoding='utf-8'), read(folder / 'reference.json')['clusters'])
        immutable(OUT / 'cache' / (name + '_rl_cluster.json'), cache)
        validate_cache(OUT / 'cache' / (name + '_rl_cluster.json'))
        print('VALIDATED', index + 1, name, flush=True)
    write(OUT / 'status.json', dict(stage='cache_complete', completed=80))


def evaluate():
    import torch
    from train_cached import build_trainer
    split, names = inputs()
    protocol = read(OUT / 'protocol.json')
    assert sha(FINAL) == protocol['final_checkpoint_sha256']
    bundle = TRAIN / 'training_bundle'
    verify_bundle(bundle)
    saved = read(TRAIN / 'training_fixed/settings.json')
    assert saved['bundle_sha256'] == sha(bundle / 'bundle_manifest.json')
    config = copy.deepcopy(saved['config'])
    assert config['llm']['bias_strength'] == 5 and config['trainer']['train_batch_size'] == 128
    config['cache_dir'] = str(OUT / 'cache')
    config['trainer'].update(train_instances=names, use_augmentation=True, pomo_size=100, n_mc_samples=1)
    config['model']['eval_type'] = 'argmax'
    cache_hashes = {}
    for name in names:
        path = OUT / 'cache' / (name + '_rl_cluster.json')
        validate_cache(path)
        cache_hashes[name] = sha(path)
    immutable(OUT / 'evaluation_contract.json', dict(config=config, cache_sha256=cache_hashes,
        checkpoint_sha256=sha(FINAL), evaluator_sha256=sha(Path(__file__)),
        frozen_bundle_sha256=sha(FROZEN / 'bundle_manifest.json')))
    trainer = build_trainer(config, TEST_ROOT, OUT / 'guided_runtime')
    state = torch.load(FINAL, map_location='cpu', weights_only=False)
    assert state['epoch'] == 1000 and state['server_contract'] == saved
    trainer.model.load_state_dict(state['model_state_dict'], strict=True)
    trainer.model.eval()
    import train_vrptw as pure
    from vrptw_env import _solution_stats
    results_path = OUT / 'results.json'
    results = read(results_path) if results_path.exists() else []
    assert [r['name'] for r in results] == names[:len(results)]
    assert [i['name'] for i in trainer.instance_pool] == names
    for index, inst in enumerate(trainer.instance_pool):
        if index < len(results):
            continue
        pure._set_seed(42 + index)
        with torch.no_grad():
            routes, reward = trainer._best_solution_aug(inst, n_aug=8)
        assert sorted(int(n) for route in routes for n in route if n) == list(range(1, 101))
        stats = _solution_stats(inst, routes)
        row = dict(name=inst['name'], kind=split['test'][index]['kind'],
            vehicles=len(stats['per_vehicle']), travel_time=stats['total_travel_time'],
            late_count=stats['late_count'], total_late=stats['total_late'],
            served=stats['n_served'], reward=float(reward), routes=routes)
        assert row['served'] == 100
        row['cost'] = .04 * row['vehicles'] + row['travel_time'] / 5760 + .0075 * row['total_late']
        assert abs(row['cost'] + row['reward']) < 1e-4, row
        results.append(row)
        write(results_path, results)
        write(OUT / 'status.json', dict(stage='evaluating', completed=len(results), total=80))
        print('EVALUATED', len(results), row['name'], 'cost', row['cost'], flush=True)
    summary = {}
    for kind in ('RAIN', 'ACCIDENT', 'ALL'):
        rows = [r for r in results if kind == 'ALL' or r['kind'] == kind]
        assert len(rows) == (80 if kind == 'ALL' else 40)
        keys = ('vehicles', 'travel_time', 'late_count', 'total_late', 'served', 'cost')
        summary[kind] = dict(count=len(rows), zero_late=sum(r['late_count'] == 0 for r in rows),
            **{k: statistics.mean(r[k] for r in rows) for k in keys},
            std={k: statistics.stdev(r[k] for r in rows) for k in keys})
    write(OUT / 'summary.json', dict(protocol=protocol, summary=summary))
    write(OUT / 'status.json', dict(stage='complete', completed=80, checkpoint_epoch=1000))
    print(json.dumps(summary, indent=2), flush=True)


def pipeline():
    for stage in ('prepare', 'call', 'evaluate'):
        command = [sys.executable, '-u']
        if stage == 'call':
            command.append(str(ROOT / 'analysis/run_with_openrouter_key.py'))
        command += [str(Path(__file__).resolve()), stage]
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'call', 'evaluate', 'pipeline'])
    args = parser.parse_args()
    try:
        globals()[args.stage]()
    except Exception as exc:
        write(OUT / 'failure.json', dict(stage=args.stage, error=str(exc)))
        raise
