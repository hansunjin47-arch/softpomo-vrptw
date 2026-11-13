"""Validate server Kim checkpoints, create held-out caches and evaluate on CUDA."""
import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

from travel_time_prompt_contract import ROOT, VERSION, prompt, validate_config
from phase2_routes_only import PUBLIC

QUEUE = ROOT / 'result/kim_tt_ac_test80_queue_20261011'


def setup(variant):
    source = ROOT / f'server_export/kim_tt_ac_train20_20261010/{variant}'
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / 'server'))
    import train_cached as io
    training = ROOT / f'server_export/kim_tt_ac_train20_results_20261011/{variant}'
    digest = io.verify_bundle(source)
    assert io.sha(training / 'bundle_manifest.json') == digest
    saved = io.read(training / 'runs/kim/settings.json')
    assert saved['bundle_sha256'] == digest
    assert saved['config'] == io.read(source / 'configs/kim.json')
    validate_config(saved['config'], variant)
    state = io.read(training / 'runs/kim/status.json')
    assert state['status'] == 'complete' and state['epoch'] == 1000
    checkpoint = training / 'runs/kim/checkpoint-1000.pt'
    cache_train = ROOT / f'result/rc101_kim_train20_config{variant}_{VERSION}'
    protocol = io.read(cache_train / 'protocol.json')
    assert protocol['version'] == VERSION and protocol['variant'] == variant
    assert protocol['environment'] == saved['config']['environment']
    assert io.sha(cache_train / 'protocol.json') == io.sha(source / 'provenance/protocol.json')
    pure = ROOT / f'result/rc101_pure_train20_b128_s42_config{variant}_travel_time_1000_20261010_v2'
    split = io.read(pure / 'split.json')
    names = [r['name'] for r in split['test']]
    assert len(set(names)) == 80 and not set(names) & set(protocol['train_names'])
    for row in split['test']:
        assert io.sha(source / 'data' / (row['name'] + '.txt')) == row['sha256']
    out = cache_train / 'eval_1000_test80_gpu'
    return source, checkpoint, saved, protocol, split, out, io


def main(stage, variant):
    source, checkpoint, saved, protocol, split, out, io = setup(variant)
    names = [r['name'] for r in split['test']]

    def immutable(path, value):
        if path.exists():
            assert io.read(path) == value, str(path)
        else:
            io.write(path, value)

    if stage == 'prepare':
        import torch
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert state['epoch'] == 1000 and state['server_contract'] == saved
        template = io.read(out.parent / 'requests' / protocol['train_names'][0] / 'request.json')
        fixed = json.loads(template['messages'][0]['content'].split('\n')[-1])['fixed_instance']
        assert template['messages'][0]['content'].startswith(prompt(variant, 'kim') + '\n')
        obsroot = ROOT / 'server_export/soft_pomo_seed_sweep/train20/eval_addon/data'
        from vrptw_env import load_solomon
        import torch as th
        base = load_solomon(str(source / 'data' / (protocol['train_names'][0] + '.txt')))
        for name in names:
            inst = load_solomon(str(source / 'data' / (name + '.txt')))
            for key in ('depot_xy', 'node_xy', 'node_demand', 'node_tw_open', 'node_tw_close', 'node_service'):
                assert th.equal(base[key], inst[key])
            assert io.sha(obsroot / (name + '.txt')) == io.sha(source / 'data' / (name + '.txt'))
            observation = io.read(obsroot / (name + '.observations.json'))
            assert set(observation) <= PUBLIC
            request = copy.deepcopy(template)
            payload = dict(scenario=name, fixed_instance=fixed, public_event=observation)
            request['messages'] = [dict(role='user', content=prompt(variant, 'kim') + '\n' + json.dumps(payload, separators=(',', ':'), allow_nan=False))]
            immutable(out / 'requests' / name / 'request.json', request)
        immutable(out / 'protocol.json', dict(version=VERSION, variant=variant, test_names=names,
            clusters=protocol['clusters'], checkpoint_sha256=io.sha(checkpoint),
            training_bundle_sha256=saved['bundle_sha256'], feedback=False))
        immutable(out / 'request_hashes.json', {n: io.sha(out / 'requests' / n / 'request.json') for n in names})
        io.write(out / 'status.json', dict(stage='prepared', completed=80))
        print('PREPARED', variant, 80, flush=True)
        return

    assert io.read(out / 'protocol.json')['checkpoint_sha256'] == io.sha(checkpoint)
    hashes = io.read(out / 'request_hashes.json')
    for name in names:
        assert io.sha(out / 'requests' / name / 'request.json') == hashes[name]
    if stage == 'call':
        sys.path.insert(0, str(ROOT))
        import VRPTWLLMModule as backend
        from run_rc101_kim_conversation import decode
        assert backend.OPENROUTER_API_KEY, 'Missing API key'
        for index, name in enumerate(names):
            folder = out / 'requests' / name
            request = io.read(folder / 'request.json')
            response = folder / 'response.txt'
            io.write(out / 'status.json', dict(stage='calling', completed=index, total=80))
            for attempt in range(3):
                if not response.exists():
                    print('CALL', variant, index + 1, name, attempt + 1, flush=True)
                    os.environ['OPENROUTER_RESPONSE_AUDIT_DIR'] = str(folder / 'api_metadata')
                    text = backend.query_llm(request['messages'][0]['content'], request['model'], response_format=request['response_format'])
                    response.write_text(text, encoding='utf-8')
                try:
                    _, cache = decode(response.read_text(encoding='utf-8'), protocol['clusters'])
                    break
                except (AssertionError, ValueError, KeyError, TypeError):
                    response.rename(folder / f'response_rejected_{len(list(folder.glob("response_rejected_*.txt"))) + 1}.txt')
                    if attempt == 2:
                        raise
            path = out / 'cache' / (name + '_cluster.json')
            immutable(path, cache)
            io.validate_cache(path)
            print('VALIDATED', variant, index + 1, flush=True)
        immutable(out / 'cache_manifest.json', dict(request_hashes_sha256=io.sha(out / 'request_hashes.json'),
            files={p.name: io.sha(p) for p in (out / 'cache').glob('*.json')}))
        io.write(out / 'status.json', dict(stage='cache_complete', completed=80))
        return

    import torch
    manifest = io.read(out / 'cache_manifest.json')
    assert manifest['request_hashes_sha256'] == io.sha(out / 'request_hashes.json')
    for name in names:
        path = out / 'cache' / (name + '_cluster.json')
        io.validate_cache(path)
        assert io.sha(path) == manifest['files'][path.name]
    config = copy.deepcopy(saved['config'])
    config['cache_dir'] = str(out / 'cache')
    config['trainer'].update(train_instances=names, use_augmentation=True, pomo_size=100, n_mc_samples=1)
    config['model']['eval_type'] = 'argmax'
    immutable(out / 'evaluation_contract.json', dict(config=config, checkpoint_sha256=io.sha(checkpoint),
        evaluator_sha256=io.sha(Path(__file__)), cache_manifest_sha256=io.sha(out / 'cache_manifest.json'), device='cuda'))
    trainer = io.build_trainer(config, source, out / 'runtime', cpu=False)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert state['epoch'] == 1000 and state['server_contract'] == saved
    trainer.model.load_state_dict(state['model_state_dict'], strict=True)
    trainer.model.eval()
    import train_vrptw as pure
    from vrptw_env import _solution_stats
    results_path = out / 'results.json'
    results = io.read(results_path) if results_path.exists() else []
    assert [r['name'] for r in results] == names[:len(results)]
    assert [i['name'] for i in trainer.instance_pool] == names
    for index, inst in enumerate(trainer.instance_pool):
        if index < len(results):
            continue
        pure._set_seed(42 + index)
        with torch.no_grad():
            routes, reward = trainer._best_solution_aug(inst, n_aug=8)
        routes = [[int(n) for n in route] for route in routes]
        assert sorted(n for route in routes for n in route if n) == list(range(1, 101))
        stats = _solution_stats(inst, routes)
        row = dict(name=inst['name'], kind=split['test'][index]['kind'], vehicles=len(stats['per_vehicle']),
            distance=float(stats['total_dist']), travel_time=float(stats['total_travel_time']),
            late_count=int(stats['late_count']), total_late=float(stats['total_late']),
            served=int(stats['n_served']), reward=float(reward), routes=routes)
        env = config['environment']
        row['cost'] = (env['vehicle_penalty'] * row['vehicles'] / env['problem_size']
            + env['time_penalty'] * row['travel_time'] / (float(inst['T']) * env['TT_max'])
            + env['late_penalty'] * row['total_late'] / (float(inst['T']) * env['Lt_max']))
        assert row['served'] == 100 and abs(row['cost'] + row['reward']) < 1e-4, row
        results.append(row)
        io.write(results_path, results)
        io.write(out / 'status.json', dict(stage='evaluating', completed=len(results), total=80))
        print('EVALUATED', variant, len(results), row['cost'], flush=True)
    keys = ('vehicles', 'travel_time', 'late_count', 'total_late', 'served', 'cost', 'reward', 'distance')
    summary = {}
    for kind in ('RAIN', 'ACCIDENT', 'ALL'):
        rows = [r for r in results if kind == 'ALL' or r['kind'] == kind]
        assert len(rows) == (80 if kind == 'ALL' else 40)
        summary[kind] = dict(count=len(rows), mean={k: statistics.mean(r[k] for r in rows) for k in keys},
            std={k: statistics.stdev(r[k] for r in rows) for k in keys})
    io.write(out / 'summary.json', summary)
    io.write(out / 'status.json', dict(stage='complete', completed=80, checkpoint_epoch=1000))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'call', 'evaluate', 'queue'])
    parser.add_argument('--variant', choices=['A', 'C'])
    parser.add_argument('--allow-api', action='store_true')
    args = parser.parse_args()
    if args.stage in ('call', 'queue') and not args.allow_api:
        parser.error('Paid calls require --allow-api')
    if args.stage != 'queue':
        if args.variant is None:
            parser.error('--variant required')
        main(args.stage, args.variant)
    else:
        QUEUE.mkdir(parents=True, exist_ok=True)
        try:
            for variant in ('A', 'C'):
                for stage in ('prepare', 'call', 'evaluate'):
                    (QUEUE / 'status.json').write_text(json.dumps(dict(variant=variant, stage=stage)), encoding='utf-8')
                    command = [sys.executable, '-u', __file__, stage, '--variant', variant]
                    if stage == 'call':
                        command.append('--allow-api')
                    subprocess.run(command, cwd=ROOT, check=True)
            (QUEUE / 'status.json').write_text(json.dumps(dict(stage='complete')), encoding='utf-8')
        except Exception as exc:
            (QUEUE / 'status.json').write_text(json.dumps(dict(stage='failed', error=repr(exc))), encoding='utf-8')
            raise
