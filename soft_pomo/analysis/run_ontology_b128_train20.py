"""New seed43/batch128 v5 experiment, using the frozen server backbone."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'server_export/soft_pomo_seed_sweep/train20'
OUT = ROOT / 'result/rc101_ontology_v5_train20_b128_seed43'
CHECKPOINT = ROOT / 'server_export/train20_seed43_checkpoint-500.pt'
sys.path.insert(0, str(FROZEN))
sys.path.insert(0, str(FROZEN / 'server'))
from train_cached import read, write, sha, verify_bundle


def prepare():
    import torch
    from train_cached import build_trainer
    verify_bundle(FROZEN)
    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    with tarfile.open(ROOT / 'server_export/train20_seed43_results.tar.gz') as archive:
        saved = json.load(archive.extractfile('train20/runs/pure128_4000_seed43/settings.json'))
    assert state['epoch'] == 500 and state['server_contract'] == saved
    assert saved['bundle_sha256'] == sha(FROZEN / 'bundle_manifest.json')
    config = copy.deepcopy(saved['config'])
    assert config['trainer']['train_batch_size'] == 128 and config['trainer']['seed'] == 43
    names = config['trainer']['train_instances']
    assert len(names) == 20
    split = read(FROZEN / 'data/split.json')
    assert names == [r['name'] for r in split['train']]
    contract = dict(checkpoint_sha256=sha(CHECKPOINT), bundle_sha256=sha(FROZEN / 'bundle_manifest.json'),
                    version='routing_context_ontology_v5', train_names=names, seed=43, batch=128,
                    reference_augmentation=8, reference_pomo=100, reference_seed='42 + train index')
    if (OUT / 'protocol.json').exists():
        assert read(OUT / 'protocol.json') == contract
    write(OUT / 'protocol.json', contract)
    extractor_config = copy.deepcopy(config)
    extractor_config['trainer'].update(use_augmentation=True, n_mc_samples=1, pomo_auto=False, eval_top_k=None)
    extractor_config['environment']['pomo_size'] = 100
    extractor_config['model']['eval_type'] = 'argmax'
    trainer = build_trainer(extractor_config, FROZEN, OUT / 'reference_runtime')
    trainer.model.load_state_dict(state['model_state_dict'], strict=True)
    trainer.model.eval()
    import train_vrptw as pure
    # Import ontology helpers only after the frozen backbone has been loaded.
    from prepare_ontology_train20 import INSTRUCTIONS, instance_context
    from routing_ontology import context, specification, attach_reference_groups
    from audit_event_relations import analyze
    write(OUT / 'ontology.json', specification())
    template = read(ROOT / 'result/rc101_ontology_train20_v5/requests' / names[0] / 'request.json')
    assert template['model'] == 'anthropic/claude-sonnet-4.6'
    for index, inst in enumerate(trainer.instance_pool):
        name = inst['name']
        assert name == names[index]
        folder = OUT / 'requests' / name
        write(OUT / 'pipeline.json', dict(stage='preparing', completed=index, total=20, name=name))
        if (folder / 'reference.json').exists():
            ref = read(folder / 'reference.json')
            assert ref['checkpoint_sha256'] == sha(CHECKPOINT)
        else:
            pure._set_seed(42 + index)
            with torch.no_grad():
                routes, reward = trainer._best_solution(inst)
            clusters = [[int(n) for n in route if n] for route in routes if any(route)]
            assert sorted(n for group in clusters for n in group) == list(range(1, 101))
            ref = dict(clusters=clusters, reward=float(reward), checkpoint_sha256=sha(CHECKPOINT))
            write(folder / 'reference.json', ref)
        obs = FROZEN / 'data' / (name + '.observations.json')
        graph, stats = analyze(inst, [[0] + c + [0] for c in ref['clusters']], read(obs))
        payload = context(graph, stats['coords'])
        attach_reference_groups(payload, instance_context(inst, stats['coords'], ref['clusters']))
        payload['reference_scope'] = dict(scenario=name, checkpoint_sha256=sha(CHECKPOINT),
            reference_file_sha256=sha(folder / 'reference.json'),
            instance_sha256=sha(FROZEN / 'data' / (name + '.txt')), public_observation_sha256=sha(obs),
            schedules=['event_reference', 'same_order_event_removed'],
            applies_to='Only this fixed route and schedule pair, not a live RL trajectory.')
        request = copy.deepcopy(template)
        request['messages'] = [dict(role='user', content=INSTRUCTIONS + '\n' + json.dumps(payload, separators=(',', ':'), allow_nan=False))]
        schema = request['response_format']['json_schema']['schema']['properties']['confidence']
        schema['minItems'] = schema['maxItems'] = 100 + len(ref['clusters'])
        for filename, value in [('request.json', request), ('context.json', payload), ('relations.json', graph)]:
            if (folder / filename).exists():
                assert read(folder / filename) == value
            else:
                write(folder / filename, value)
        print('PREPARED', index + 1, name, flush=True)
    write(OUT / 'source_config.json', config)
    write(OUT / 'pipeline.json', dict(stage='prepared', completed=20, api_calls=0))


def call():
    from run_rc101_kim_conversation import decode
    import VRPTWLLMModule as backend
    names = read(OUT / 'protocol.json')['train_names']
    for index, name in enumerate(names):
        folder = OUT / 'requests' / name
        request = read(folder / 'request.json')
        response_path = folder / 'response.txt'
        write(OUT / 'pipeline.json', dict(stage='calling', completed=index, total=20, name=name))
        if not response_path.exists():
            print('CALL', index + 1, name, flush=True)
            response = backend.query_llm(request['messages'][0]['content'], request['model'], response_format=request['response_format'])
            response_path.write_text(response, encoding='utf-8')
        _, cache = decode(response_path.read_text(encoding='utf-8'), read(folder / 'reference.json')['clusters'])
        write(OUT / 'cache' / (name + '_rl_cluster.json'), cache)
        print('VALIDATED', index + 1, name, flush=True)
    write(OUT / 'pipeline.json', dict(stage='cache_complete', completed=20))


def train():
    assert read(OUT / 'pipeline.json')['stage'] == 'cache_complete'
    bundle = OUT / 'training_bundle'
    if bundle.exists():
        raise FileExistsError('Preserve existing training bundle; inspect before resuming')
    shutil.copytree(FROZEN, bundle)
    shutil.copytree(OUT / 'cache', bundle / 'cache')
    (bundle / 'checkpoints').mkdir()
    shutil.copy2(CHECKPOINT, bundle / 'checkpoints/pure500.pt')
    config = read(OUT / 'source_config.json')
    old = read(ROOT / 'result/rc101_relation_train20_cache_v1/settings.json')
    config.update(method='phase2', cache_dir='cache', initial_checkpoint='checkpoints/pure500.pt', llm=old['llm'])
    config['trainer'].update(epochs=1000, pomo_size=100, cluster_mode='rl', bias_mode='cluster', start_mode='random',
        benchmark='rc1', experience_refresh_epoch=-1, rl_recluster_epoch=None, require_reference_start=False)
    assert config['trainer']['train_batch_size'] == 128 and config['llm']['bias_strength'] == 5
    write(bundle / 'configs/phase2.json', config)
    write(bundle / 'bundle_manifest.json', dict(format=1, sha256={p.relative_to(bundle).as_posix():sha(p)
        for p in bundle.rglob('*') if p.is_file() and p.name != 'bundle_manifest.json' and '__pycache__' not in p.parts}))
    write(OUT / 'pipeline.json', dict(stage='training', start_epoch=501, end_epoch=1000, batch=128, seed=43))
    subprocess.run([sys.executable, '-u', str(bundle / 'server/train_cached.py'), '--method', 'phase2',
                    '--seed', '43', '--out', str(OUT / 'training'), '--max-hours', '11.5'], check=True, cwd=bundle)
    status = read(OUT / 'training/status.json')
    write(OUT / 'pipeline.json', dict(stage=status['status'], epoch=status['epoch'], test_evaluation=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'call', 'train'])
    args = parser.parse_args()
    {'prepare': prepare, 'call': call, 'train': train}[args.stage]()
