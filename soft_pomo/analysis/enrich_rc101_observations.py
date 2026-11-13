"""Create a versioned pilot with synthetic news; preserve event dynamics/split."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / 'original_POMO'))
import generate_events as ge
from vrptw_env import load_solomon, _solution_stats


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def enrich(source, split_path, output):
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    split = json.loads(split_path.read_text(encoding='utf-8'))
    output.mkdir(parents=True, exist_ok=False)
    rules = dict(version='synthetic_observations_discrete_v1',
        assumption='Synthetic ordinal association, NOT an empirical traffic model or physical inverse.',
        rainfall_units='mm/h; legacy field rainfall_mm stores intensity, not accumulated rainfall',
        rain_multiplier_edges=[1.1, 1.4, 1.7, 2.0],
        rainfall_representatives=[5, 10, 20],
        accident_multiplier_edges=[1.2, 1.65, 2.1, 2.55, 3.0],
        vehicles_by_bin=[2, 3, 4, 5],
        access='Mapping and simulator files are audit-only; do not include in RL/LLM inputs.')
    if manifest.get('strength_alpha') is not None:
        rules['strength_alpha'] = manifest['strength_alpha']
        rules['rain_multiplier_edges'][0] = min(1.1, manifest['multiplier']['RAIN'][0])
        rules['accident_multiplier_edges'][0] = min(1.2, manifest['multiplier']['ACCIDENT'][0])
        rules['mapping_note'] = ('Classify the reduced multiplier with the existing observable '
                                 'thresholds; extend the lowest bin for weaker events.')
    write(output / 'observation_rules.audit.json', rules)
    raw = ge.load_raw(manifest['source'])
    routes = json.loads((source / 'base_reference.json').read_text())['routes']
    cases = []
    for case in manifest['cases']:
        old_path = source / (case['name'] + '.txt')
        assert digest(old_path) == case['file_sha256']
        ev = copy.deepcopy(case['event'])
        mu = ev['multiplier']
        if ev['type'] == 'RAIN':
            tier = int(np.searchsorted([1.4, 1.7], mu, side='right'))
            ev['rainfall_mm'] = rules['rainfall_representatives'][tier]
            detail = f"Rainfall intensity of approximately {ev['rainfall_mm']} mm/h is forecast"
            revealed = 0.0
        else:
            tier = int(np.searchsorted([1.65, 2.1, 2.55], mu, side='right'))
            ev['vehicles_involved'] = rules['vehicles_by_bin'][tier]
            detail = f"A collision involving {ev['vehicles_involved']} vehicles is reported"
            revealed = ev['trigger_time']
        public = dict(event_type=ev['type'], affected_nodes=ev['nodes'],
            regions=case['sector_names'], start_min=ev['trigger_time'],
            end_min=round(ev['trigger_time'] + ev['duration'], 4),
            revealed_at_min=revealed, synthetic=True)
        if ev['type'] == 'RAIN':
            public['rainfall_mm_per_hour'] = ev['rainfall_mm']
        else:
            public['vehicles_involved'] = ev['vehicles_involved']
        public['news_text'] = (f"{detail} in {', '.join(case['sector_names'])}. "
            f"The announced disruption window is minute {public['start_min']:.4f} "
            f"to {public['end_min']:.4f} after dispatch.")
        path = output / (case['name'] + '.txt')
        ge.write_scenario(raw, [ev], case['name'], str(path))
        write(output / (case['name'] + '.observations.json'), public)
        old, new = load_solomon(str(old_path)), load_solomon(str(path))
        for key in ['nodes', 'trigger_time', 'duration', 'multiplier', 'type']:
            assert old['preset_events'][0][key] == new['preset_events'][0][key]
        np.testing.assert_equal(_solution_stats(old, routes), _solution_stats(new, routes))
        updated = dict(case, event=ev, file_sha256=digest(path), source_sha256=digest(old_path))
        cases.append(updated)
    manifest.update(cases=cases, source_pilot=str(source),
        observations='Populated synthetic observations; public sidecars exclude multipliers.',
        observation_rules='observation_rules.audit.json',
        observation_code_sha256=digest(Path(__file__)))
    write(output / 'manifest.json', manifest)
    split.update(data=str(output), source_split=str(split_path),
                 generator_manifest_hash=digest(output / 'manifest.json'))
    for group in ['train', 'test']:
        for entry in split[group]:
            entry['source_sha256'] = entry['sha256']
            entry['sha256'] = digest(output / (entry['name'] + '.txt'))
    write(output / 'split.json', split)
    write(output / 'validation.json', dict(count=len(cases),
        train=len(split['train']), test=len(split['test']),
        fixed_route_replay_identical=True, event_dynamics_unchanged=True,
        public_news_count=len(cases), original_data_untouched=True))
    print(output)
    print(f'{len(cases)} enriched scenarios; unchanged dynamics and '
          f'{len(split["train"])}/{len(split["test"])} membership; replay matched.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    enrich(args.source.resolve(), args.split.resolve(), args.output.resolve())
