"""Versioned A/C ontology requests from each objective's own Pure500 checkpoint."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'ontology_action_semantics_ac_v1'
OBS = ROOT / 'server_export/soft_pomo_seed_sweep/train20/data'


def setup(variant):
    run = ROOT / f'result/rc101_pure_train20_b128_s42_config{variant}_1000_followup'
    source = run / 'source'
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / 'server'))
    import train_cached as io
    out = ROOT / f'result/rc101_phase2_train20_b128_s42_config{variant}_{VERSION}'
    return run, source, out, io


def objective_prompt(variant):
    from prepare_ontology_v5_action_semantics_v1 import INSTRUCTIONS
    old = INSTRUCTIONS[INSTRUCTIONS.index('Minimize J='):INSTRUCTIONS.index('Capacity is hard')]
    if variant == 'A':
        objective = '''Minimize J_A=0.06*K+D/2880+0.0025*Lt.
The intended weighted preference is vehicles, then distance, then lateness, NOT strict lexicographic ordering.
One vehicle costs as much as 172.8 distance units or 24 total lateness minutes.
Saving 100 distance units compensates for 13.8889 additional lateness minutes, with other terms fixed.
'''
    else:
        objective = '''Minimize J_C=0.04*K+D/1920+0.0025*Lt.
The intended weighted preference is distance, then vehicles, then lateness, NOT strict lexicographic ordering.
One vehicle costs as much as 76.8 distance units or 16 total lateness minutes.
Saving 100 distance units compensates for 20.8333 additional lateness minutes, with other terms fixed.
'''
    objective += '''K is the number of nonempty vehicle routes. D is total Euclidean/free-flow route distance,
including depot legs, NOT event-inflated driving time. Lt is total customer AND depot lateness in minutes.
Events affect arrival times and lateness but do not directly increase D. Late-stop count is not in this cost.
The unserved-customer term is 500*(unserved customers/100); full service remains required.
Do not add a vehicle merely to eliminate a small delay, or prefer a longer route merely because it avoids traffic.
Evaluate such trade-offs with the cost above; lateness is still penalized, not ignored.
'''
    return INSTRUCTIONS.replace(old, objective)


def prepare(variant, cpu=False):
    run, source, out, io = setup(variant)
    for name, digest in io.read(run / 'manifest.json').items():
        assert io.sha(run / name) == digest, name
    import torch
    checkpoint = run / 'training/checkpoint-500.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = io.read(run / 'config.json')
    assert state['epoch'] == 500
    assert state['server_contract'] == dict(config=config, manifest_sha256=io.sha(run / 'manifest.json'))
    assert config['trainer']['train_batch_size'] == 128 and config['trainer']['seed'] == 42
    assert config['environment']['vehicle_penalty'] == (6 if variant == 'A' else 4)
    assert config['environment']['D_max'] == (12 if variant == 'A' else 8)
    assert config['environment']['late_penalty'] == 3 and config['environment']['Lt_max'] == 5
    names = config['trainer']['train_instances']
    assert len(names) == 20
    prompt = objective_prompt(variant)
    prompts = ROOT / 'analysis/prompts' / VERSION
    prompts.mkdir(parents=True, exist_ok=True)
    # Expanded prompts are immutable artifacts, one file per objective.
    for v in ('A', 'C'):
        path = prompts / f'config{v}.txt'
        text = objective_prompt(v)
        if path.exists():
            assert path.read_text(encoding='utf-8') == text
        else:
            path.write_text(text, encoding='utf-8')
    protocol = dict(version=VERSION, variant=variant, train_names=names,
        checkpoint=str(checkpoint), checkpoint_sha256=io.sha(checkpoint),
        source_manifest_sha256=io.sha(run / 'manifest.json'),
        prompt_sha256=io.sha(prompts / f'config{variant}.txt'),
        seed=42, batch=128, beta=5, logit_clipping=10, pomo_starts=100,
        reference_augmentation=8, reference_seed='42 + train index',
        **({'reference_device': 'cpu'} if cpu else {}),
        information='Offline event-reference and same-order event-removed feedback; unchanged ontology facts',
        api_scope='20 training scenarios only; no test calls or C calls implied')
    if (out / 'protocol.json').exists():
        assert io.read(out / 'protocol.json') == protocol
    io.write(out / 'protocol.json', protocol)
    extract = copy.deepcopy(config)
    extract['trainer'].update(use_augmentation=True, n_mc_samples=1, pomo_auto=False, eval_top_k=None)
    trainer = io.build_trainer(extract, source, out / 'reference_runtime', cpu=cpu)
    trainer.model.load_state_dict(state['model_state_dict'], strict=True)
    trainer.model.eval()
    import train_vrptw as pure
    from prepare_ontology_train20 import instance_context
    from routing_ontology import context, specification, attach_reference_groups
    from audit_event_relations import analyze
    io.write(out / 'ontology.json', specification())
    template = io.read(ROOT / 'result/rc101_ontology_train20_v5/requests' / names[0] / 'request.json')
    for index, inst in enumerate(trainer.instance_pool):
        name = inst['name']
        assert name == names[index]
        assert io.sha(source / 'data' / (name + '.txt')) == io.sha(OBS / (name + '.txt'))
        folder = out / 'requests' / name
        io.write(out / 'pipeline.json', dict(stage='preparing', completed=index, total=20))
        if (folder / 'reference.json').exists():
            ref = io.read(folder / 'reference.json')
            assert ref['checkpoint_sha256'] == protocol['checkpoint_sha256']
        else:
            pure._set_seed(42 + index)
            with torch.no_grad():
                routes, reward = trainer._best_solution(inst)
            clusters = [[int(n) for n in route if n] for route in routes if any(route)]
            assert sorted(n for group in clusters for n in group) == list(range(1, 101))
            ref = dict(clusters=clusters, reward=float(reward), checkpoint_sha256=protocol['checkpoint_sha256'])
            io.write(folder / 'reference.json', ref)
        obs = OBS / (name + '.observations.json')
        graph, stats = analyze(inst, [[0] + c + [0] for c in ref['clusters']], io.read(obs))
        payload = context(graph, stats['coords'])
        attach_reference_groups(payload, instance_context(inst, stats['coords'], ref['clusters']))
        payload['reference_scope'] = dict(scenario=name, checkpoint_sha256=protocol['checkpoint_sha256'],
            reference_file_sha256=io.sha(folder / 'reference.json'),
            instance_sha256=io.sha(source / 'data' / (name + '.txt')), public_observation_sha256=io.sha(obs),
            schedules=['event_reference', 'same_order_event_removed'],
            applies_to='Only this reference schedule pair, not a live RL trajectory.')
        request = copy.deepcopy(template)
        request['messages'] = [dict(role='user', content=prompt + '\n' + json.dumps(payload, separators=(',', ':'), allow_nan=False))]
        assert '5760' not in prompt and '0.0075' not in prompt
        schema = request['response_format']['json_schema']['schema']['properties']['confidence']
        schema['minItems'] = schema['maxItems'] = 100 + len(ref['clusters'])
        for filename, value in [('request.json', request), ('context.json', payload), ('relations.json', graph)]:
            path = folder / filename
            if path.exists():
                assert io.read(path) == value
            else:
                io.write(path, value)
        print('PREPARED', index + 1, name, flush=True)
    io.write(out / 'source_config.json', config)
    io.write(out / 'request_manifest.json', {str(p.relative_to(out)): io.sha(p)
        for p in (out / 'requests').glob('*/*.json')})
    io.write(out / 'pipeline.json', dict(stage='prepared', completed=20, api_calls=0))


def train(variant):
    run, source, out, io = setup(variant)
    assert io.read(out / 'pipeline.json')['stage'] == 'cache_complete'
    protocol = io.read(out / 'protocol.json')
    checkpoint = Path(protocol['checkpoint'])
    assert io.sha(checkpoint) == protocol['checkpoint_sha256']
    for name, digest in io.read(run / 'manifest.json').items():
        assert io.sha(run / name) == digest, name
    for name in protocol['train_names']:
        io.validate_cache(out / 'cache' / (name + '_rl_cluster.json'))
    bundle = out / 'training_bundle'
    if bundle.exists():
        raise FileExistsError('Training bundle exists; inspect before resuming')
    shutil.copytree(source, bundle, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(out / 'cache', bundle / 'cache')
    (bundle / 'checkpoints').mkdir()
    shutil.copy2(checkpoint, bundle / 'checkpoints/pure500.pt')
    config = io.read(out / 'source_config.json')
    llm = io.read(ROOT / 'result/rc101_relation_train20_cache_v1/settings.json')['llm']
    llm.update(reward_config=variant, bias_strength=5.0, use_ontology=True)
    config.update(method='phase2', cache_dir='cache', initial_checkpoint='checkpoints/pure500.pt', llm=llm)
    config['trainer'].update(epochs=1000, pomo_size=100, cluster_mode='rl', bias_mode='cluster',
        start_mode='random', benchmark='rc1', experience_refresh_epoch=-1,
        rl_recluster_epoch=None, require_reference_start=False)
    assert config['trainer']['train_batch_size'] == 128 and config['trainer']['seed'] == 42
    assert config['environment'] == io.read(run / 'config.json')['environment']
    io.write(bundle / 'configs/phase2.json', config)
    io.write(bundle / 'phase2_provenance.json', dict(protocol=protocol,
        branch='Historical Phase2: transfer model and optimizer; reset RNG to seed42, not an exact Pure RL resume',
        prompt_sha256=protocol['prompt_sha256']))
    io.write(bundle / 'bundle_manifest.json', dict(format=1, sha256={p.relative_to(bundle).as_posix(): io.sha(p)
        for p in bundle.rglob('*') if p.is_file() and p.name != 'bundle_manifest.json'}))
    io.write(out / 'pipeline.json', dict(stage='training', start_epoch=501, end_epoch=1000,
        batch=128, seed=42, variant=variant, api_calls_during_training=False))
    subprocess.run([sys.executable, '-u', str(bundle / 'server/train_cached.py'), '--method', 'phase2',
        '--seed', '42', '--out', str(out / 'training'), '--max-hours', '11.5'], check=True, cwd=bundle)
    status = io.read(out / 'training/status.json')
    io.write(out / 'pipeline.json', dict(stage=status['status'], epoch=status['epoch'], test_evaluation=False))


def call(variant):
    _, _, out, io = setup(variant)
    from run_rc101_kim_conversation import decode
    import VRPTWLLMModule as backend
    assert backend.OPENROUTER_API_KEY, 'OPENROUTER_API_KEY is not available'
    for name, digest in io.read(out / 'request_manifest.json').items():
        assert io.sha(out / name) == digest, name
    names = io.read(out / 'protocol.json')['train_names']
    for index, name in enumerate(names):
        folder = out / 'requests' / name
        request = io.read(folder / 'request.json')
        path = folder / 'response.txt'
        io.write(out / 'pipeline.json', dict(stage='calling', completed=index, total=20, name=name))
        if not path.exists():
            print('CALL', index + 1, name, flush=True)
            response = backend.query_llm(request['messages'][0]['content'], request['model'], response_format=request['response_format'])
            path.write_text(response, encoding='utf-8')
        _, cache = decode(path.read_text(encoding='utf-8'), io.read(folder / 'reference.json')['clusters'])
        target = out / 'cache' / (name + '_rl_cluster.json')
        io.write(target, cache)
        io.validate_cache(target)
        print('VALIDATED', index + 1, name, flush=True)
    io.write(out / 'pipeline.json', dict(stage='cache_complete', completed=20, training_started=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'call', 'prepare-call', 'train'])
    parser.add_argument('--variant', choices=['A', 'C'], default='A')
    parser.add_argument('--cpu', action='store_true', help='Prepare reference routes on CPU without interrupting GPU training')
    parser.add_argument('--allow-legacy-distance', action='store_true', help='Explicitly authorize the obsolete distance experiment')
    args = parser.parse_args()
    if not args.allow_legacy_distance:
        parser.error('Legacy distance experiment. Use prepare_phase2_tt_train20.py for corrected request preparation.')
    if args.stage == 'prepare-call':
        # Preparation disables networking; API calls must use a fresh process.
        for stage in ('prepare', 'call'):
            command = [sys.executable, '-u', __file__, stage, '--variant', args.variant]
            if args.cpu and stage == 'prepare':
                command.append('--cpu')
            subprocess.run(command, check=True, cwd=ROOT)
    elif args.stage == 'prepare':
        prepare(args.variant, cpu=args.cpu)
    elif args.stage == 'call':
        call(args.variant)
    else:
        train(args.variant)
