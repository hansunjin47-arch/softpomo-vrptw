"""Objective-matched test80 ontology caches and frozen Phase2 evaluation."""
import argparse
import copy
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

from prepare_phase2_ac_train20 import ROOT, VERSION, setup, objective_prompt


def main(stage, variant, device='cpu', epoch=1000):
    run, source, train, io = setup(variant)
    out = train / 'eval_1000_test80'
    def immutable(path, value):
        if path.exists():
            assert io.read(path) == value, str(path)
        else:
            io.write(path, value)
    split = io.read(run / 'split.json')
    names = [r['name'] for r in split['test']]
    assert len(names) == len(set(names)) == 80
    protocol = io.read(train / 'protocol.json')
    assert not set(names) & set(protocol['train_names'])
    checkpoint = Path(protocol['checkpoint'])
    training_dir = train / ('training' if epoch == 1000 else 'continuation_3000')
    final = training_dir / f'checkpoint-{epoch}.pt'
    for name, digest in io.read(run / 'manifest.json').items():
        assert io.sha(run / name) == digest, name
    for entry in split['test']:
        assert io.sha(source / 'data' / (entry['name'] + '.txt')) == entry['sha256']
    if stage == 'pipeline':
        for child in ('prepare', 'call', 'evaluate'):
            command = [sys.executable, '-u', __file__, child, '--variant', variant]
            if child == 'evaluate':
                command += ['--device', device]
            subprocess.run(command, check=True, cwd=ROOT)
        return
    if stage == 'prepare':
        import torch
        config = io.read(train / 'source_config.json')
        config['trainer'].update(train_instances=names, use_augmentation=True, n_mc_samples=1,
            pomo_auto=False, eval_top_k=None)
        immutable(out / 'protocol.json', dict(test_names=names, variant=variant, version=VERSION,
            reference_checkpoint_sha256=io.sha(checkpoint), final_checkpoint_sha256=io.sha(final),
            prompt_sha256=protocol['prompt_sha256'], train_protocol_sha256=io.sha(train / 'protocol.json'),
            device='cpu', augmentation=8, pomo=100, beta=5, per_instance_seed='42 + test index'))
        trainer = io.build_trainer(config, source, out / 'reference_runtime', cpu=True)
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert state['epoch'] == 500 and io.sha(checkpoint) == protocol['checkpoint_sha256']
        trainer.model.load_state_dict(state['model_state_dict'], strict=True)
        trainer.model.eval()
        import train_vrptw as pure
        from prepare_ontology_train20 import instance_context
        from routing_ontology import context, specification, attach_reference_groups
        from audit_event_relations import analyze
        assert specification() == io.read(train / 'ontology.json')
        template = io.read(train / 'requests' / protocol['train_names'][0] / 'request.json')
        prompt = objective_prompt(variant)
        assert template['messages'][0]['content'].startswith(prompt + '\n')
        assert [i['name'] for i in trainer.instance_pool] == names
        for index, inst in enumerate(trainer.instance_pool):
            name = inst['name']
            folder = out / 'requests' / name
            if (folder / 'reference.json').exists():
                ref = io.read(folder / 'reference.json')
                assert ref['checkpoint_sha256'] == io.sha(checkpoint)
            else:
                pure._set_seed(42 + index)
                with torch.no_grad():
                    routes, reward = trainer._best_solution(inst)
                clusters = [[int(n) for n in route if n] for route in routes if any(route)]
                assert sorted(n for c in clusters for n in c) == list(range(1, 101))
                ref = dict(clusters=clusters, reward=float(reward), checkpoint_sha256=io.sha(checkpoint))
                immutable(folder / 'reference.json', ref)
            obsroot = ROOT / 'server_export/soft_pomo_seed_sweep/train20/eval_addon/data'
            assert io.sha(obsroot / (name + '.txt')) == io.sha(source / 'data' / (name + '.txt'))
            obs = obsroot / (name + '.observations.json')
            graph, stats = analyze(inst, [[0]+c+[0] for c in ref['clusters']], io.read(obs))
            payload = context(graph, stats['coords'])
            attach_reference_groups(payload, instance_context(inst, stats['coords'], ref['clusters']))
            payload['reference_scope'] = dict(scenario=name, checkpoint_sha256=io.sha(checkpoint),
                reference_file_sha256=io.sha(folder / 'reference.json'),
                instance_sha256=io.sha(source / 'data' / (name + '.txt')), public_observation_sha256=io.sha(obs),
                schedules=['event_reference', 'same_order_event_removed'],
                applies_to='Only this reference schedule pair, not a live RL trajectory.')
            request = copy.deepcopy(template)
            request['messages'] = [dict(role='user', content=prompt+'\n'+json.dumps(payload, separators=(',', ':'), allow_nan=False))]
            schema = request['response_format']['json_schema']['schema']['properties']['confidence']
            schema['minItems'] = schema['maxItems'] = 100 + len(ref['clusters'])
            for filename, value in [('request.json', request), ('context.json', payload), ('relations.json', graph)]:
                immutable(folder / filename, value)
            io.write(out / 'status.json', dict(stage='preparing', completed=index+1, total=80))
            print('PREPARED', index+1, name, flush=True)
        immutable(out / 'request_hashes.json', {n: io.sha(out / 'requests' / n / 'request.json') for n in names})
        io.write(out / 'status.json', dict(stage='prepared', completed=80))
    elif stage == 'call':
        from run_rc101_kim_conversation import decode
        import VRPTWLLMModule as backend
        hashes = io.read(out / 'request_hashes.json')
        for index, name in enumerate(names):
            folder = out / 'requests' / name
            assert io.sha(folder / 'request.json') == hashes[name]
            request = io.read(folder / 'request.json')
            response = folder / 'response.txt'
            io.write(out / 'status.json', dict(stage='calling', completed=index, total=80, name=name))
            if not response.exists():
                print('CALL', index+1, name, flush=True)
                text = backend.query_llm(request['messages'][0]['content'], request['model'], response_format=request['response_format'])
                response.write_text(text, encoding='utf-8')
            _, cache = decode(response.read_text(encoding='utf-8'), io.read(folder / 'reference.json')['clusters'])
            immutable(out / 'cache' / (name+'_rl_cluster.json'), cache)
            io.validate_cache(out / 'cache' / (name+'_rl_cluster.json'))
            print('VALIDATED', index+1, name, flush=True)
        io.write(out / 'status.json', dict(stage='cache_complete', completed=80))
    elif stage == 'evaluate':
        import torch
        if epoch == 1000:
            assert io.sha(final) == io.read(out / 'protocol.json')['final_checkpoint_sha256']
        cache_out = out
        if device == 'cuda' or epoch != 1000:
            out = train / (f'eval_{epoch}_test80' + ('_gpu' if device == 'cuda' else ''))
        bundle = train / 'training_bundle'
        io.verify_bundle(bundle)
        saved = io.read(training_dir / 'settings.json')
        assert saved['bundle_sha256'] == io.sha(bundle / 'bundle_manifest.json')
        config = copy.deepcopy(saved['config'])
        config['cache_dir'] = str(cache_out / 'cache')
        config['trainer'].update(train_instances=names, use_augmentation=True, pomo_size=100, n_mc_samples=1)
        config['model']['eval_type'] = 'argmax'
        for name in names:
            io.validate_cache(cache_out / 'cache' / (name+'_rl_cluster.json'))
        immutable(out / 'evaluation_contract.json', dict(config=config, checkpoint_sha256=io.sha(final),
            evaluator_sha256=io.sha(Path(__file__)), device=device,
            cache_sha256={n: io.sha(cache_out / 'cache' / (n+'_rl_cluster.json')) for n in names}))
        trainer = io.build_trainer(config, source, out / 'guided_runtime', cpu=device == 'cpu')
        state = torch.load(final, map_location='cpu', weights_only=False)
        assert state['epoch'] == epoch and state['server_contract'] == saved
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
            pure._set_seed(42+index)
            with torch.no_grad():
                routes, reward = trainer._best_solution_aug(inst, n_aug=8)
            routes = [[int(n) for n in r] for r in routes]
            assert sorted(n for r in routes for n in r if n) == list(range(1, 101))
            stats = _solution_stats(inst, routes)
            row = dict(name=inst['name'], kind=split['test'][index]['kind'], vehicles=len(stats['per_vehicle']),
                distance=float(stats['total_dist']), travel_time=float(stats['total_travel_time']),
                late_count=int(stats['late_count']), total_late=float(stats['total_late']),
                served=int(stats['n_served']), reward=float(reward), routes=routes)
            env = config['environment']
            horizon = float(inst['T'])
            row['cost'] = (env['vehicle_penalty']*row['vehicles']/env['problem_size']
                + env['dist_penalty_weight']*row['distance']/(horizon*env['D_max'])
                + env['time_penalty']*row['travel_time']/(horizon*env['TT_max'])
                + env['late_penalty']*row['total_late']/(horizon*env['Lt_max'])
                + env['late_count_penalty']*row['late_count']/env['problem_size'])
            assert row['served'] == 100 and abs(row['cost']+row['reward']) < 1e-4, row
            results.append(row)
            io.write(results_path, results)
            io.write(out / 'status.json', dict(stage='evaluating', completed=len(results), total=80))
            print('EVALUATED', len(results), row['name'], row['cost'], flush=True)
        summary = {}
        for kind in ('RAIN', 'ACCIDENT', 'ALL'):
            rows = [r for r in results if kind == 'ALL' or r['kind'] == kind]
            assert len(rows) == (80 if kind == 'ALL' else 40)
            keys = ('vehicles', 'distance', 'travel_time', 'late_count', 'total_late', 'served', 'cost', 'reward')
            summary[kind] = dict(count=len(rows), mean={k: statistics.mean(r[k] for r in rows) for k in keys},
                std={k: statistics.stdev(r[k] for r in rows) for k in keys})
        io.write(out / 'summary.json', summary)
        io.write(out / 'status.json', dict(stage='complete', completed=80, checkpoint_epoch=epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'call', 'evaluate', 'pipeline'])
    parser.add_argument('--variant', choices=['A', 'C'], default='A')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--wait-for-c', action='store_true')
    parser.add_argument('--epoch', type=int, choices=[1000, 2000, 3000], default=1000)
    args = parser.parse_args()
    try:
        if args.epoch != 1000 and args.stage != 'evaluate':
            parser.error('Continuation checkpoints reuse existing caches; use evaluate')
        if args.device == 'cuda' and args.stage not in ('evaluate', 'pipeline'):
            parser.error('CUDA option applies to evaluation, directly or after pipeline preparation')
        if args.wait_for_c:
            _, _, train, io = setup(args.variant)
            destination = train / 'eval_1000_test80_gpu'
            dependency = ROOT / f'result/rc101_phase2_train20_b128_s42_configC_{VERSION}/training/status.json'
            io.write(destination / 'status.json', dict(stage='waiting_for_C_training', dependency=str(dependency)))
            deadline = time.monotonic() + 24*3600
            while True:
                status = io.read(dependency)
                if status['status'] == 'complete' and status['epoch'] == 1000:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError('C training did not finish within 24 hours')
                time.sleep(30)
            time.sleep(10)
            print('C training complete; starting cached GPU evaluation', flush=True)
        main(args.stage, args.variant, device=args.device, epoch=args.epoch)
    except Exception as exc:
        _, _, train, io = setup(args.variant)
        folder = f'eval_{args.epoch}_test80' + ('_gpu' if args.device == 'cuda' else '')
        io.write(train / folder / 'failure.json', dict(stage=args.stage, error=str(exc)))
        raise
