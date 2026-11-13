"""Frozen five-seed Test80 reporting after equal-budget curriculum pilots."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
import math

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'result/rc_curriculum_test80_20261112'
DATA = ROOT / 'result/rc101_phase2_online_gpt_train20_v2_2_20261018/source'
KEYS = ('reward', 'vehicles', 'travel_time', 'total_late', 'unserved')
RUNS = {42: 'rc_pure42_k17_curriculum_20261112', 44: 'rc_pure_k17_curriculum_20261112',
        45: 'rc_pure45_k15_curriculum_20261112',
        43: 'rc_curriculum_confirmatory_20261112/seed43',
        46: 'rc_curriculum_confirmatory_20261112/seed46'}
SOURCES = ('analysis/evaluate_rc_curriculum_test80_20261112.py',
    'experiments/concurrent_fleet_pair/train.py', 'experiments/concurrent_fleet_pair/env.py',
    'experiments/concurrent_fleet_pair/policy.py', 'experiments/concurrent_fleet_v8/env.py',
    'experiments/concurrent_fleet_v8/policy.py', 'experiments/concurrent_fleet_v5/env.py',
    'experiments/concurrent_fleet_v7/policy.py', 'train_vrptw.py', 'vrptw_env.py')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    from experiments.concurrent_fleet_pair.io import write as atomic
    atomic(path, value)


def aggregate(rows):
    return {k: mean(r[k] for r in rows) for k in KEYS}


def prepare():
    import torch
    from experiments.concurrent_fleet_pair.policy import FleetPolicy
    if OUT.exists():
        raise RuntimeError('Fresh output required')
    torch.set_num_threads(1)
    split = read(DATA / 'split.json')
    rows = split['test']
    assert len(rows) == len({r['name'] for r in rows}) == 80
    assert not ({r['name'] for r in rows} & {r['name'] for r in split['train']})
    hashes = {p: sha(ROOT / p) for p in SOURCES}
    for row in rows:
        path = DATA / 'data' / (row['name'] + '.txt')
        assert sha(path) == row['sha256']
        hashes[str(path.relative_to(ROOT))] = sha(path)
        observation = path.with_suffix('.observations.json')
        hashes[str(observation.relative_to(ROOT))] = sha(observation)
        public = read(observation)
        assert 'multiplier' not in public
        if public['event_type'] == 'RAIN':
            assert public['revealed_at_min'] == 0
        else:
            assert public['revealed_at_min'] == public['start_min']
    checkpoints = {}
    for seed, folder in sorted(RUNS.items()):
        config_path = ROOT / 'result' / folder / 'protocol.json'
        cfg = read(config_path)['config']
        assert cfg['trainer']['seed'] == seed
        assert cfg['environment']['late_penalty'] == 5
        assert cfg['environment']['protect_last_service_vehicle'] and cfg['environment']['normalize_advantage_std']
        assert abs(cfg['environment']['vehicle_penalty'] - 10000 / 1696.95) < 1e-8
        hashes[str(config_path.relative_to(ROOT))] = sha(config_path)
        baseline = ROOT / f'result/rc_pair_returnmask_v100_l5_s{seed}_{"20261031" if seed == 42 else "20261102"}'
        assert read(baseline / 'split.json')['test'] == rows
        baseline_result = baseline / 'late5/results.json'
        hashes[str(baseline_result.relative_to(ROOT))] = sha(baseline_result)
        for arm in ('control', 'curriculum'):
            path = ROOT / 'result' / folder / arm / 'checkpoint-40.pt'
            saved = torch.load(path, map_location='cpu', weights_only=False)
            assert saved['epoch'] == 40
            model = FleetPolicy(**cfg['model']).cpu().eval()
            model.load_state_dict(saved['model_state_dict'], strict=True)
            assert model.new_customer_epsilon == 0
            checkpoints[f'{seed}/{arm}'] = dict(path=str(path.relative_to(ROOT)), sha256=sha(path))
    lkh = ROOT / 'analysis/rc101_pilot_history_20_80_20260923_104517/results.json'
    assert {r['name'] for r in read(lkh)} == {r['name'] for r in rows}
    hashes[str(lkh.relative_to(ROOT))] = sha(lkh)
    OUT.mkdir()
    write(OUT / 'protocol.json', dict(setting='RC101 simultaneous v100 lambda5 maskON stdON',
        evaluation='Fixed official existing Test80; single free greedy AMP rollout, epsilon0',
        budget='control1040 vs curriculum1040 primary; originalPure1000 separately labeled',
        seeds=[42, 43, 44, 45, 46], exploratory_seeds=[42, 44, 45], confirmatory_rule_seeds=[43, 46],
        targets={'42': 17, '43': 15, '44': 17, '45': 15, '46': 16},
        no_reselection='No target, schedule or epoch changes after these Test results; all results reported',
        caveat='Existing historical Test has previously been inspected; not a newly untouched final test.',
        source_data_config_hashes=hashes, checkpoints=checkpoints, test_rows=rows,
        api_calls=0, further_training=False, cpu_strict_checkpoint_loads=10,
        source_preparation='Frozen before new Test inference; no new results existed.'))
    write(OUT / 'status.json', dict(stage='prepared_cpu_only', gpu_started=False))
    print('Prepared 10 frozen checkpoints x80 cases; CPU validation passed. GPU not started.')


def evaluate():
    import torch
    from vrptw_env import load_solomon
    from experiments.concurrent_fleet_pair import train
    from experiments.concurrent_fleet_pair.policy import FleetPolicy
    torch.set_num_threads(1)
    assert torch.cuda.is_available()
    protocol = read(OUT / 'protocol.json')
    assert all(sha(ROOT / p) == digest for p, digest in protocol['source_data_config_hashes'].items())
    rows = protocol['test_rows']
    for seed, folder in sorted(RUNS.items()):
        cfg = read(ROOT / 'result' / folder / 'protocol.json')['config']
        for arm in ('control', 'curriculum'):
            destination = OUT / f'seed{seed}' / f'{arm}.json'
            if destination.exists():
                previous = read(destination)
                assert len(previous) == 80 and {r['name'] for r in previous} == {r['name'] for r in rows}
                continue
            checkpoint = protocol['checkpoints'][f'{seed}/{arm}']
            path = ROOT / checkpoint['path']
            assert sha(path) == checkpoint['sha256']
            model = FleetPolicy(**cfg['model']).cuda().eval()
            model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model_state_dict'], strict=True)
            results = []
            for row in rows:
                path = DATA / 'data' / (row['name'] + '.txt')
                inst = load_solomon(str(path))
                public = [read(path.with_suffix('.observations.json'))]
                env, _, _ = train.collect(model, inst, public, cfg['environment'], 1, True, greedy=True)
                results.append(dict(name=row['name'], **{k: float(v[0]) for k, v in env.metrics().items()}))
                write(OUT / 'status.json', dict(stage='evaluating', seed=seed, arm=arm, completed=len(results), total=80))
            write(destination, results)
            print('COMPLETE', seed, arm, aggregate(results), flush=True)
            del model, env
            torch.cuda.empty_cache()
    report = {}
    for seed in sorted(RUNS):
        arms = {arm: read(OUT / f'seed{seed}' / f'{arm}.json') for arm in ('control', 'curriculum')}
        control = {r['name']: r for r in arms['control']}
        paired = [dict(name=r['name'], delta={k: r[k] - control[r['name']][k] for k in KEYS})
                  for r in arms['curriculum']]
        groups = {}
        for kind in ('all', 'RAIN', 'ACC'):
            selected = [r for r in paired if kind == 'all' or f'_{kind}_' in r['name']]
            groups[kind] = dict(delta={k: mean(r['delta'][k] for r in selected) for k in KEYS},
                wins=sum(r['delta']['reward'] > 1e-6 for r in selected),
                losses=sum(r['delta']['reward'] < -1e-6 for r in selected),
                late_increase_cases=sum(r['delta']['total_late'] > 1e-6 for r in selected))
        baseline = ROOT / f'result/rc_pair_returnmask_v100_l5_s{seed}_{"20261031" if seed == 42 else "20261102"}/late5/results.json'
        report[str(seed)] = dict(means={**{a: aggregate(v) for a, v in arms.items()},
                                       'original_pure_1000': aggregate(read(baseline))}, paired=paired, groups=groups)
    overall = {}
    for arm in ('control', 'curriculum', 'original_pure_1000'):
        overall[arm] = {k: dict(mean=mean(report[str(s)]['means'][arm][k] for s in RUNS),
            ci95_halfwidth=1.96 * stdev(report[str(s)]['means'][arm][k] for s in RUNS) / math.sqrt(5)) for k in KEYS}
    lkh = read(ROOT / 'analysis/rc101_pilot_history_20_80_20260923_104517/results.json')
    for r in lkh:
        assert r['served'] == 100
        r['unserved'] = 0
        r['reward'] = -(100 * r['vehicles'] + r['travel_time'] + 5 * r['total_late']) / 1696.95
    write(OUT / 'summary.json', dict(seed_results=report, five_seed_summary=overall, lkh=aggregate(lkh),
        ci='Normal approximation 1.96*sampleSD/sqrt(5), over seed means; not instance-level CI'))
    write(OUT / 'status.json', dict(stage='complete', seeds=sorted(RUNS), evaluations=800, api_calls=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate-gpu', action='store_true')
    args = parser.parse_args()
    evaluate() if args.evaluate_gpu else prepare()
