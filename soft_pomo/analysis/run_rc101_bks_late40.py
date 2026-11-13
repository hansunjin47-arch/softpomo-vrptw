"""Matched fresh RC101 late40 training followed by test80 GPU evaluation."""
import argparse
import copy
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'server'))
from train_cached import read, write, sha, verify_bundle

SOURCE = ROOT / 'result/bks_normalized_late10_train20_20261013/rc101'
OUT = ROOT / 'result/rc101_bks_late40_train20_20261015'


def prepare():
    verify_bundle(SOURCE)
    OUT.mkdir(exist_ok=False)
    for path in SOURCE.glob('*.py'):
        shutil.copy2(path, OUT / path.name)
    shutil.copytree(SOURCE / 'data', OUT / 'data')
    (OUT / 'server').mkdir()
    for name in ('train_cached.py', 'eval_pure_tt_bundle.py'):
        shutil.copy2(SOURCE / 'server' / name, OUT / 'server' / name)
    config = read(SOURCE / 'configs/pure.json')
    original = copy.deepcopy(config)
    assert config['environment']['late_penalty'] == 10
    assert config['environment']['dist_penalty_weight'] == 0
    assert config['environment']['TT_max'] == 1696.95 / 240
    assert config['trainer']['epochs'] == 1000
    assert config['trainer']['train_batch_size'] == 128
    assert config['trainer']['seed'] == 42
    assert 'initial_checkpoint' not in config
    config['environment']['late_penalty'] = 40.0
    check = copy.deepcopy(config)
    check['environment']['late_penalty'] = 10
    assert check == original
    write(OUT / 'configs/pure.json', config)
    shutil.copy2(SOURCE / 'split.json', OUT / 'split.json')
    split = read(OUT / 'split.json')
    assert len(split['train']) == 20 and len(split['test']) == 80
    for row in split['train'] + split['test']:
        assert sha(OUT / 'data' / (row['name'] + '.txt')) == row['sha256']
    write(OUT / 'provenance.json', dict(objective='K/14+(T+40*L)/1696.95',
        lateness_weight=40, fresh_training=True, matched_baseline=str(SOURCE),
        baseline_manifest_sha256=sha(SOURCE / 'bundle_manifest.json')))
    write(OUT / 'bundle_manifest.json', {'sha256': {
        p.relative_to(OUT).as_posix(): sha(p) for p in OUT.rglob('*') if p.is_file()}})
    verify_bundle(OUT)
    write(OUT / 'pipeline.json', dict(stage='prepared'))


def run():
    verify_bundle(OUT)
    write(OUT / 'pipeline.json', dict(stage='training', target_epoch=1000))
    command = [sys.executable, '-u', str(OUT / 'server/train_cached.py'),
        '--method', 'pure', '--max-hours', '168', '--save-every', '10']
    if (OUT / 'runs/pure/checkpoint-last.pt').exists():
        command.append('--resume')
    subprocess.run(command, cwd=OUT, check=True)
    assert read(OUT / 'runs/pure/status.json')['status'] == 'complete'
    write(OUT / 'pipeline.json', dict(stage='evaluating', total=80))
    subprocess.run([sys.executable, '-u', str(OUT / 'server/eval_pure_tt_bundle.py'),
        '--bundle', str(OUT), '--epoch', '1000'], cwd=OUT, check=True)
    state = read(OUT / 'eval_1000_test80_gpu/status.json')
    assert state['stage'] == 'complete' and state['completed'] == 80
    write(OUT / 'pipeline.json', dict(stage='complete', epoch=1000, evaluated=80))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    try:
        prepare() if args.prepare else run()
    except Exception as error:
        if OUT.exists():
            write(OUT / 'pipeline.json', dict(stage='failed', error=repr(error)))
        raise
