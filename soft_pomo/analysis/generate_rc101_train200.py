"""Extend frozen train80 by 120 draws; preserve old files and test80 exactly."""
import copy
from pathlib import Path
import shutil
import numpy as np

from run_rc101_train40_extension import read, signature
from eval_fixed_v2_lkh import ROOT, write, sha
from generate_rc101_event_pilot import main as generate
from enrich_rc101_observations import enrich
from vrptw_env import load_solomon


def main():
    old = ROOT / 'data/rc101_train80_test80_20260929'
    raw = ROOT / 'data/rc101_extra_train120_raw_20261002'
    observed = ROOT / 'data/rc101_extra_train120_discrete_20261002'
    out = ROOT / 'data/rc101_train200_test80_20261002'
    assert not any(p.exists() for p in (raw, observed, out)), 'Never overwrite generated datasets'
    original = read(old / 'manifest.json')
    prior = read(old / 'split.json')
    assert len(prior['train']) == len(prior['test']) == 80
    generate('rc101', raw, seed=20261002, per_kind=60, case_prefix='Z', diagnostics=False)
    added = read(raw / 'manifest.json')
    assert not {signature(c) for c in original['cases']} & {signature(c) for c in added['cases']}
    write(raw / 'split.json', dict(train=[dict(name=c['name'], kind=c['kind'], sha256=c['file_sha256'])
                                         for c in added['cases']], test=[]))
    # Reuse a fixed route solely to check that observation enrichment preserves dynamics.
    reference_path = ROOT / 'result/rc101_pure500_train80_phase2_20260930/requests/RC101_RAIN_P018/reference.json'
    reference = read(reference_path)
    write(raw / 'base_reference.json', dict(routes=[[0]+c+[0] for c in reference['clusters']],
        purpose='Enrichment-equivalence check only; not a BASE optimum or acceptance criterion',
        source=str(reference_path), source_sha256=sha(reference_path)))
    enrich(raw, raw / 'split.json', observed)
    extra = read(observed / 'manifest.json')
    assert read(old / 'observation_rules.audit.json') == read(observed / 'observation_rules.audit.json')
    out.mkdir()
    for source, manifest in ((old, original), (observed, extra)):
        for case in manifest['cases']:
            for suffix in ('.txt', '.observations.json'):
                path = source / (case['name'] + suffix)
                shutil.copy2(path, out / path.name)
                assert sha(path) == sha(out / path.name)
    for name in ('RC101_BASE.txt', 'observation_rules.audit.json'):
        shutil.copy2(old / name, out / name)
    manifest = dict(original, cases=original['cases'] + extra['cases'], added_seed=20261002,
                    added_source=str(observed), original_source=str(old),
                    extension_code_sha256=sha(Path(__file__)), added_count=120)
    assert len({signature(c) for c in manifest['cases']}) == 280
    write(out / 'manifest.json', manifest)
    split = copy.deepcopy(prior)
    split['train'] += read(observed / 'split.json')['train']
    split.update(data=str(out), generator_manifest_hash=sha(out / 'manifest.json'),
                 role='Nested train200; legacy exploratory test80 preserved, not a fresh final holdout')
    assert split['train'][:80] == prior['train'] and split['test'] == prior['test']
    assert len({x['name'] for group in ('train','test') for x in split[group]}) == 280
    fields = ['n_customers','vehicle_limit','vehicle_capacity','T','depot_xy','node_xy','node_demand',
              'node_tw_open','node_tw_close','node_service','depot_demand','depot_tw_open','depot_tw_close','depot_service']
    base = load_solomon(str(out / 'RC101_BASE.txt'))
    for group, count in (('train', 100), ('test', 40)):
        for kind in ('RAIN','ACCIDENT'):
            assert sum(e['kind'] == kind for e in split[group]) == count
        for entry in split[group]:
            path = out / (entry['name'] + '.txt')
            assert sha(path) == entry['sha256']
            inst = load_solomon(str(path))
            for field in fields:
                assert np.array_equal(np.asarray(inst[field]), np.asarray(base[field])), field
            observation = read(out / (entry['name'] + '.observations.json'))
            assert 'multiplier' not in observation and len(inst['preset_events']) == 1
    write(out / 'split.json', split)
    write(out / 'validation.json', dict(train=200, test=80, added=120, added_rain=60, added_accident=60,
        original_instances_and_observations_byte_identical=True, original_test_split_unchanged=True,
        no_duplicate_events=True, static_attributes_identical=True,
        added_validation=read(observed / 'validation.json'),
        training_started=False, llm_calls=0, fresh_validation_and_test_generated=False))
    write(out / 'file_hashes.json', {p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    print(f'PASS: train200 (100 RAIN + 100 ACC), test80 unchanged; {out}')


if __name__ == '__main__':
    main()
