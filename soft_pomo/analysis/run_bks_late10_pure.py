"""Matched fresh C101/R101 runs changing only the lateness coefficient to ten."""
import argparse
import copy
import shutil
import subprocess
import sys

import run_bks_normalized_pure as baseline
from train_cached import read, write, sha, verify_bundle

ROOT = baseline.ROOT
SOURCE = baseline.OUT
OUT = ROOT / 'result/bks_normalized_late10_train20_20261013'


def prepare():
    OUT.mkdir(exist_ok=False)
    for name, *_ in baseline.CASES:
        source = SOURCE / name
        verify_bundle(source)
        target = OUT / name
        target.mkdir()
        for path in source.glob('*.py'):
            shutil.copy2(path, target / path.name)
        shutil.copytree(source / 'data', target / 'data')
        (target / 'server').mkdir()
        shutil.copy2(source / 'server/train_cached.py', target / 'server/train_cached.py')
        shutil.copy2(ROOT / 'server/eval_pure_tt_bundle.py', target / 'server/eval_pure_tt_bundle.py')
        config = read(source / 'configs/pure.json')
        expected = copy.deepcopy(config)
        assert config['environment']['late_penalty'] == 1
        assert config['trainer']['epochs'] == 1000 and config['trainer']['train_batch_size'] == 128
        assert config['trainer']['seed'] == 42
        config['environment']['late_penalty'] = 10.0
        check = copy.deepcopy(config)
        check['environment']['late_penalty'] = 1
        assert check == expected
        write(target / 'configs/pure.json', config)
        split = read(source / 'split.json')
        for row in split['train'] + split['test']:
            assert sha(target / 'data' / (row['name'] + '.txt')) == row['sha256']
        write(target / 'split.json', split)
        provenance = read(source / 'provenance.json')
        provenance.update(lateness_weight=10, objective='K/K0 + (travel_time_minutes + 10*lateness_minutes)/T0',
                          matched_baseline=str(source), baseline_manifest_sha256=sha(source / 'bundle_manifest.json'),
                          fresh_training=True)
        write(target / 'provenance.json', provenance)
        write(target / 'bundle_manifest.json', {'sha256': {
            p.relative_to(target).as_posix(): sha(p) for p in target.rglob('*') if p.is_file()}})
        verify_bundle(target)
    write(OUT / 'status.json', dict(stage='prepared', lateness_weight=10))


def run():
    for name, *_ in baseline.CASES:
        target = OUT / name
        write(OUT / 'status.json', dict(stage='training', instance=name, target_epoch=1000, lateness_weight=10))
        subprocess.run([sys.executable, '-u', str(target / 'server/train_cached.py'),
                        '--method', 'pure', '--max-hours', '168', '--save-every', '10'], cwd=target, check=True)
        assert read(target / 'runs/pure/status.json')['status'] == 'complete'
        write(OUT / 'status.json', dict(stage='evaluating', instance=name, lateness_weight=10))
        subprocess.run([sys.executable, '-u', str(target / 'server/eval_pure_tt_bundle.py'),
                        '--bundle', str(target), '--epoch', '1000'], cwd=target, check=True)
    write(OUT / 'status.json', dict(stage='complete', lateness_weight=10))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    try:
        prepare() if args.prepare else run()
    except Exception as error:
        if OUT.exists():
            write(OUT / 'status.json', dict(stage='failed', error=repr(error)))
        raise
