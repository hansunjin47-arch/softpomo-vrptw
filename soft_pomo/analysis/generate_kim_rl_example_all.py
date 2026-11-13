"""Finish all train caches before generating shared test caches."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'result/kim_rl_example_all_20261027'
CASES = ('rc101', 'r101', 'c101')


def run(script, *args):
    subprocess.run([sys.executable, str(ROOT / 'analysis' / script), *args], cwd=ROOT, check=True)


def generate(case, split):
    run('generate_kim_rl_example_train.py', 'all', '--case', case, '--split', split, '--allow-api')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow-api', action='store_true', required=True)
    parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        short = case.replace('101', '')
        base = ROOT / f'result/kim_rl_example_base_pilot_{short}_20261027'
        if not (base / 'reference.json').exists():
            run('pilot_kim_rl_example_base.py', 'reference', '--case', case)
        if not (base / 'cache_manifest.json').exists():
            run('pilot_kim_rl_example_base.py', 'prepare', '--case', case)
            run('pilot_kim_rl_example_base.py', 'call', '--case', case, '--allow-api')
    for split in ('train', 'test'):
        print('STAGE', split, flush=True)
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(lambda case: generate(case, split), CASES))
        audit(split)
    print('ALL COMPLETE: 3 BASE + 60 TRAIN + 240 TEST', flush=True)


def audit(split):
    sys.path.insert(0, str(ROOT / 'server'))
    import train_cached as io
    report = {}
    for case in CASES:
        short = case.replace('101', '')
        base = ROOT / f'result/kim_rl_example_base_pilot_{short}_20261027'
        folder = ROOT / f'result/kim_rl_example_{short}_{split}_20261027'
        protocol = io.read(folder / 'protocol.json')
        expected = 20 if split == 'train' else 80
        assert len(protocol['names']) == expected
        manifest = io.read(folder / 'cache_manifest.json')
        hashes = io.read(folder / 'request_hashes.json')
        assert io.sha(folder / 'request_hashes.json') == manifest['request_hashes_sha256']
        assert len(manifest['files']) == expected
        prior = io.validate_cache(base / 'cache/BASE_cluster.json')
        reference = io.read(base / 'requests/BASE/context.json')['rl_reference_example']
        source = ROOT / f'result/kim_l5_gpt_rc_r_c_20261026/{case}'
        train_names = set(io.read(source / 'train/protocol.json')['names'])
        test_names = set(io.read(source / 'test/protocol.json')['names'])
        assert not train_names & test_names
        assert reference['scenario'] not in train_names | test_names
        cost = 0
        for name in protocol['names']:
            path = folder / 'cache' / (name + '_cluster.json')
            assert io.sha(path) == manifest['files'][path.name]
            cache = io.validate_cache(path)
            assert all(v == 0 for v in cache['_depot'].values())
            assert {k: set(v) for k, v in cache.items()} == {k: set(v) for k, v in prior.items()}
            request_dir = folder / 'requests' / name
            assert io.sha(request_dir / 'request.json') == hashes[name]
            context = io.read(request_dir / 'context.json')
            assert set(context) == {'scenario', 'fixed_instance', 'public_event', 'prior_base_confidence', 'rl_reference_example'}
            assert context['rl_reference_example'] == reference
            assert context['prior_base_confidence'] == prior
            old_context = io.read(source / split / 'requests' / name / 'context.json')
            assert all(context[k] == v for k, v in old_context.items())
            metadata = io.read(request_dir / 'api_metadata/response.json')
            assert metadata['choices'][0]['finish_reason'] == 'stop'
            cost += metadata['usage'].get('cost', 0)
        report[case] = dict(count=expected, cost_usd=cost, verified=True)
    (OUT / f'{split}_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('AUDIT', split, json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
