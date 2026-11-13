"""Run RC101 late10 after the existing C101/R101 training and evaluation queue."""
import argparse
import copy
import shutil
import subprocess
import sys
import time

import run_bks_late10_pure as previous
from train_cached import read, write, sha, verify_bundle

ROOT = previous.ROOT
OUT = previous.OUT / 'rc101'
SOURCE = ROOT / 'server_export/rc101_bks_train20_80_e1000_20261013/train20_pure'


def prepare():
    verify_bundle(SOURCE)
    OUT.mkdir(exist_ok=False)
    for path in SOURCE.glob('*.py'):
        shutil.copy2(path, OUT / path.name)
    shutil.copytree(SOURCE / 'data', OUT / 'data')
    (OUT / 'server').mkdir()
    shutil.copy2(SOURCE / 'server/train_cached.py', OUT / 'server/train_cached.py')
    shutil.copy2(ROOT / 'server/eval_pure_tt_bundle.py', OUT / 'server/eval_pure_tt_bundle.py')
    config = read(SOURCE / 'configs/pure.json')
    original = copy.deepcopy(config)
    assert config['environment']['late_penalty'] == 1
    assert config['environment']['dist_penalty_weight'] == 0
    assert config['environment']['TT_max'] == 1696.95 / 240
    assert config['environment']['Lt_max'] == 1696.95 / 240
    assert config['trainer']['epochs'] == 1000 and config['trainer']['train_batch_size'] == 128
    assert config['trainer']['seed'] == 42
    config['environment']['late_penalty'] = 10.0
    check = copy.deepcopy(config)
    check['environment']['late_penalty'] = 1
    assert check == original
    write(OUT / 'configs/pure.json', config)
    split = read(SOURCE / 'split.json')
    assert len(split['train']) == 20 and len(split['test']) == 80
    for row in split['train'] + split['test']:
        assert sha(OUT / 'data' / (row['name'] + '.txt')) == row['sha256']
    write(OUT / 'split.json', split)
    write(OUT / 'provenance.json', dict(objective='K/14+(T+10*L)/1696.95',
        lateness_weight=10, horizon=240, fresh_training=True,
        matched_baseline=str(SOURCE), baseline_manifest_sha256=sha(SOURCE / 'bundle_manifest.json')))
    write(OUT / 'bundle_manifest.json', {'sha256': {
        p.relative_to(OUT).as_posix(): sha(p) for p in OUT.rglob('*') if p.is_file()}})
    verify_bundle(OUT)
    write(OUT / 'queue_status.json', dict(stage='prepared'))


def run():
    write(OUT / 'queue_status.json', dict(stage='waiting_for_c101_r101', dependency=str(previous.OUT / 'status.json')))
    while True:
        state = read(previous.OUT / 'status.json')
        if state['stage'] == 'complete':
            for name in ('c101', 'r101'):
                assert read(previous.OUT / name / 'runs/pure/status.json')['status'] == 'complete'
                assert read(previous.OUT / name / 'eval_1000_test80_gpu/status.json')['stage'] == 'complete'
            break
        if state['stage'] == 'failed':
            raise RuntimeError('Prior queue failed; RC101 not started: ' + str(state))
        time.sleep(60)
    verify_bundle(OUT)
    write(OUT / 'queue_status.json', dict(stage='training', target_epoch=1000))
    subprocess.run([sys.executable, '-u', str(OUT / 'server/train_cached.py'),
        '--method', 'pure', '--max-hours', '168', '--save-every', '10'], cwd=OUT, check=True)
    assert read(OUT / 'runs/pure/status.json')['status'] == 'complete'
    write(OUT / 'queue_status.json', dict(stage='evaluating', total=80))
    subprocess.run([sys.executable, '-u', str(OUT / 'server/eval_pure_tt_bundle.py'),
        '--bundle', str(OUT), '--epoch', '1000'], cwd=OUT, check=True)
    write(OUT / 'queue_status.json', dict(stage='complete', epoch=1000, evaluated=80))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    try:
        prepare() if args.prepare else run()
    except Exception as error:
        if OUT.exists():
            write(OUT / 'queue_status.json', dict(stage='failed', error=repr(error)))
        raise
