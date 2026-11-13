"""Prepare unseen RC event validation; GPU evaluation requires an explicit flag."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'result/rc_curriculum_validation_20261112'
DATA = OUT / 'events'
SOURCE = ROOT / 'result/rc101_phase2_online_gpt_train20_v2_2_20261018/source'
RUNS = {42: 'rc_pure42_k17_curriculum_20261112', 44: 'rc_pure_k17_curriculum_20261112',
        45: 'rc_pure45_k15_curriculum_20261112'}
KEYS = ('reward', 'vehicles', 'travel_time', 'total_late', 'unserved')
SOURCE_FILES = ('analysis/validate_rc_curriculum_events_20261112.py',
    'experiments/concurrent_fleet_pair/train.py', 'experiments/concurrent_fleet_pair/env.py',
    'experiments/concurrent_fleet_pair/policy.py', 'experiments/concurrent_fleet_v8/env.py',
    'experiments/concurrent_fleet_v8/policy.py', 'experiments/concurrent_fleet_v5/env.py',
    'experiments/concurrent_fleet_v7/policy.py', 'train_vrptw.py', 'vrptw_env.py')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    from experiments.concurrent_fleet_pair.io import write as atomic_write
    atomic_write(path, value)


def signature(event):
    return (event['type'], tuple(sorted(event['nodes'])), float(event['trigger_time']),
            float(event['duration']), float(event['multiplier']))


def prepare():
    sys.path.insert(0, str(ROOT / 'analysis'))
    from analysis.generate_rc101_event_pilot import main as generate
    from analysis.enrich_rc101_observations import enrich
    from vrptw_env import load_solomon
    if OUT.exists():
        raise RuntimeError('Fresh output required')
    OUT.mkdir()
    raw = OUT / 'raw_events'
    generate(output=raw, seed=20261112, per_kind=10, case_prefix='V20261112', diagnostics=False)
    old_manifest = read(ROOT / 'data/rc101_event_pilot_discrete_v1/manifest.json')
    new_manifest = read(raw / 'manifest.json')
    for key in ('source_sha256', 'horizon', 'regions', 'duration_fraction', 'multiplier',
                'start', 'spatial', 'acceptance'):
        assert new_manifest[key] == old_manifest[key], key
    split = read(SOURCE / 'split.json')
    # Existing Test inputs are read for identity exclusion only, never performance.
    excluded = {signature(load_solomon(str(SOURCE / 'data' / (r['name'] + '.txt')))['preset_events'][0])
                for r in split['train'] + split['test']}
    reference_event = copy.deepcopy(load_solomon(str(SOURCE / 'data' / (split['train'][0]['name'] + '.txt')))['preset_events'][0])
    reference_event.update(trigger_time=123.456, duration=44.321)
    excluded.add(signature(reference_event))
    assert not ({signature(c['event']) for c in new_manifest['cases']} & excluded)
    baseline_routes = Path(old_manifest['source_pilot']) / 'base_reference.json'
    shutil.copy2(baseline_routes, raw / 'base_reference.json')
    temporary_split = dict(train=[dict(name=c['name'], kind=c['kind'], sha256=c['file_sha256'])
                                 for c in new_manifest['cases']], test=[])
    write(raw / 'split_for_enrichment.json', temporary_split)
    enrich(raw, raw / 'split_for_enrichment.json', DATA)
    manifest = read(DATA / 'manifest.json')
    public_hashes = {}
    for case in manifest['cases']:
        before = load_solomon(str(raw / (case['name'] + '.txt')))
        after = load_solomon(str(DATA / (case['name'] + '.txt')))
        assert signature(before['preset_events'][0]) == signature(after['preset_events'][0])
        public_path = DATA / (case['name'] + '.observations.json')
        public = read(public_path)
        assert 'multiplier' not in public
        key = 'rainfall_mm_per_hour' if case['kind'] == 'RAIN' else 'vehicles_involved'
        assert public[key] > 0
        assert public['revealed_at_min'] == (0 if case['kind'] == 'RAIN' else case['event']['trigger_time'])
        public_hashes[case['name']] = digest(public_path)
    checkpoints = {}
    for seed, folder in RUNS.items():
        for arm in ('control', 'curriculum'):
            path = ROOT / 'result' / folder / arm / 'checkpoint-40.pt'
            checkpoints[f'{seed}/{arm}'] = dict(path=str(path), sha256=digest(path))
    protocol = dict(role='New event validation, not untouched final Test and not new geometry',
        seed=20261112, count=20, family_counts=dict(RAIN=10, ACCIDENT=10),
        targets={'42': 17, '44': 17, '45': 15}, schedule='20 constrained + 20 free vs 40 free',
        models=checkpoints, data_sha256=digest(DATA / 'manifest.json'), observation_hashes=public_hashes,
        generation_code_sha256=digest(ROOT / 'analysis/generate_rc101_event_pilot.py'),
        enrichment_code_sha256=digest(ROOT / 'analysis/enrich_rc101_observations.py'),
        observation_mapping='Existing discrete_v1 deterministic mapping; no extra mapping RNG',
        identity_exclusion_count=len(excluded), geometry_and_event_distribution_match=True,
        dynamics_unchanged_by_enrichment=True, no_test_performance_access=True,
        success='Each seed final free greedy reward delta > 0 and unserved == 0; report all lateness tradeoffs.',
        tuning='No target or schedule changes based on this validation; failures retained.',
        evaluation='Existing pair trainer.collect, single greedy rollout, AMP, epsilon0', api_calls=0)
    write(OUT / 'protocol.json', protocol)
    write(OUT / 'status.json', dict(stage='prepared_cpu_only', gpu_started=False))
    print('Prepared and CPU-validated 20 unseen event scenarios; GPU not started.')


def freeze_sources():
    assert read(OUT / 'status.json')['stage'] == 'prepared_cpu_only'
    assert not any(OUT.glob('seed*/*.json'))
    protocol = read(OUT / 'protocol.json')
    protocol['source_hashes'] = {p: digest(ROOT / p) for p in SOURCE_FILES}
    protocol['config_protocol_hashes'] = {str(seed): digest(ROOT / 'result' / folder / 'protocol.json')
                                          for seed, folder in RUNS.items()}
    protocol['source_freeze_note'] = 'Added before evaluation; no validation outcomes existed.'
    write(OUT / 'protocol.json', protocol)
    print('Evaluator, model/environment/trainer and config sources frozen before evaluation.')


def evaluate():
    import torch
    from experiments.concurrent_fleet_pair import train
    from experiments.concurrent_fleet_pair.policy import FleetPolicy
    from vrptw_env import load_solomon
    torch.set_num_threads(1)
    assert torch.cuda.is_available()
    protocol = read(OUT / 'protocol.json')
    assert all(digest(ROOT / p) == value for p, value in protocol['source_hashes'].items())
    assert all(digest(ROOT / 'result' / folder / 'protocol.json') == protocol['config_protocol_hashes'][str(seed)]
               for seed, folder in RUNS.items())
    assert digest(DATA / 'manifest.json') == protocol['data_sha256']
    cases = read(DATA / 'manifest.json')['cases']
    for seed, folder in RUNS.items():
        cfg = read(ROOT / 'result' / folder / 'protocol.json')['config']
        for arm in ('control', 'curriculum'):
            destination = OUT / f'seed{seed}' / f'{arm}.json'
            if destination.exists():
                continue
            ref = protocol['models'][f'{seed}/{arm}']
            assert digest(Path(ref['path'])) == ref['sha256']
            saved = torch.load(ref['path'], map_location='cpu', weights_only=False)
            assert saved['epoch'] == 40
            model = FleetPolicy(**cfg['model']).cuda().eval()
            model.load_state_dict(saved['model_state_dict'], strict=True)
            rows = []
            for case in cases:
                path = DATA / (case['name'] + '.txt')
                assert digest(path) == case['file_sha256']
                public_path = path.with_suffix('.observations.json')
                assert digest(public_path) == protocol['observation_hashes'][case['name']]
                inst = load_solomon(str(path))
                env, _, _ = train.collect(model, inst, [read(public_path)], cfg['environment'], 1, True, greedy=True)
                rows.append(dict(name=case['name'], **{k: float(v[0]) for k, v in env.metrics().items()}))
                write(OUT / 'status.json', dict(stage='evaluating', seed=seed, arm=arm, completed=len(rows)))
            write(destination, rows)
            del model, saved, env
            torch.cuda.empty_cache()
    summary = {}
    for seed in RUNS:
        arms = {arm: read(OUT / f'seed{seed}' / f'{arm}.json') for arm in ('control', 'curriculum')}
        means = {arm: {k: mean(r[k] for r in rows) for k in KEYS} for arm, rows in arms.items()}
        delta = {k: means['curriculum'][k] - means['control'][k] for k in KEYS}
        control = {r['name']: r for r in arms['control']}
        paired = [dict(name=r['name'], delta={k: r[k] - control[r['name']][k] for k in KEYS})
                  for r in arms['curriculum']]
        groups = {}
        for group in ('all', 'RAIN', 'ACC'):
            rows = [r for r in paired if group == 'all' or f'_{group}_' in r['name']]
            groups[group] = dict(n=len(rows), delta={k: mean(r['delta'][k] for r in rows) for k in KEYS},
                wins=sum(r['delta']['reward'] > 1e-6 for r in rows),
                losses=sum(r['delta']['reward'] < -1e-6 for r in rows),
                ties=sum(abs(r['delta']['reward']) <= 1e-6 for r in rows),
                late_increase_cases=sum(r['delta']['total_late'] > 1e-6 for r in rows))
        summary[str(seed)] = dict(means=means, delta=delta,
            paired=paired, groups=groups,
            pass_criterion=delta['reward'] > 0 and means['curriculum']['unserved'] == 0)
    write(OUT / 'summary.json', summary)
    write(OUT / 'status.json', dict(stage='complete', test_performance_accessed=False, api_calls=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate-gpu', action='store_true')
    parser.add_argument('--freeze-sources', action='store_true')
    args = parser.parse_args()
    if args.freeze_sources:
        freeze_sources()
    elif args.evaluate_gpu:
        evaluate()
    else:
        prepare()
