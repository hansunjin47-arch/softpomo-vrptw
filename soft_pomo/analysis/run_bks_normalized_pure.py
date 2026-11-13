"""Frozen C101/R101 train20 pilots with BKS-normalized time and lateness."""
import argparse
import copy
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'server'))
from train_cached import read, write, sha
from enrich_rc101_observations import enrich

OUT = ROOT / 'result/bks_normalized_train20_20261013'
CASES = [
    ('c101', 10, 828.94, 1236, 'c101_history_strength200_20261013_141106',
     'c101_event_pilot_20260923_203633'),
    ('r101', 19, 1650.80, 230, 'r101_history_strength050_20260925_233437',
     'r101_event_pilot_20260923_202358'),
]


def prepare():
    OUT.mkdir(exist_ok=False)
    prior = ROOT / 'result/rc101_pure_train20_b128_s42_configA_travel_time_1000_20261010_v2'
    for name, k0, t0, horizon, history, original in CASES:
        target = OUT / name
        target.mkdir()
        split_path = ROOT / 'analysis' / history / 'split.json'
        split = read(split_path)
        raw = target / 'raw_audit'
        shutil.copytree(Path(split['data']), raw)
        shutil.copy2(ROOT / 'data' / original / 'base_reference.json', raw / 'base_reference.json')
        # Keep the existing public-observation thresholds, including their saturation.
        enrich(raw, split_path, target / 'data')
        split = read(target / 'data/split.json')
        assert len(split['train']) == 20 and len(split['test']) == 80
        for p in (prior / 'source').glob('*.py'):
            shutil.copy2(p, target / p.name)
        (target / 'server').mkdir()
        shutil.copy2(ROOT / 'server/train_cached.py', target / 'server/train_cached.py')
        config = copy.deepcopy(read(prior / 'config.json'))
        config['trainer']['train_instances'] = [r['name'] for r in split['train']]
        config['environment'].update(vehicle_penalty=100 / k0,
            time_penalty=1.0, TT_max=t0 / horizon, late_penalty=1.0,
            Lt_max=t0 / horizon, dist_penalty_weight=0.0, late_count_penalty=0.0)
        assert config['trainer']['epochs'] == 1000
        write(target / 'configs/pure.json', config)
        write(target / 'split.json', split)
        write(target / 'provenance.json', dict(source=str(prior), bks_vehicles=k0,
            bks_distance=t0, horizon=horizon, lateness_weight=1,
            objective='K/K0 + (travel_time_minutes + lateness_minutes)/T0',
            unserved_penalty=config['environment']['unserved_penalty'],
            reference='https://www.sintef.no/projectweb/top/vrptw/100-customers/',
            observation_rule='Existing absolute severity bins, saturated above highest threshold',
            history=str(ROOT / 'analysis' / history), fresh_training=True))
        write(target / 'bundle_manifest.json', {'sha256': {
            p.relative_to(target).as_posix(): sha(p) for p in target.rglob('*') if p.is_file()}})
    write(OUT / 'status.json', dict(stage='prepared'))


def run():
    for name, *_ in CASES:
        target = OUT / name
        write(OUT / 'status.json', dict(stage='training', instance=name, target_epoch=1000))
        command = [sys.executable, '-u', str(target / 'server/train_cached.py'),
                   '--method', 'pure', '--max-hours', '168', '--save-every', '10']
        subprocess.run(command, cwd=target, check=True)
        assert read(target / 'runs/pure/status.json')['status'] == 'complete'
        write(OUT / 'status.json', dict(stage='evaluating', instance=name))
        subprocess.run([sys.executable, '-u', str(ROOT / 'server/eval_pure_tt_bundle.py'),
                        '--bundle', str(target), '--epoch', '1000'], cwd=ROOT, check=True)
    write(OUT / 'status.json', dict(stage='complete'))


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
