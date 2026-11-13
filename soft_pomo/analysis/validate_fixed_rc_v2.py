"""Check serialized events, source preservation and coverage of the fixed RC pool."""
from collections import Counter
import json
from pathlib import Path
import sys

from generate_fixed_rc_v2 import ROOT, digest
from vrptw_env import load_solomon

out = Path(sys.argv[1])
manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
assert not manifest['failures'], manifest['failures']
prefix = {'RC': 'RCH', 'C': 'CH', 'R': 'RH'}[manifest.get('family', 'RC')]
assert sorted(r['name'] for r in manifest['instances']) == [f'{prefix}{i:02d}' for i in range(1, 16)]
rain = Counter()
acc = Counter()
waves = Counter()
for row in manifest['instances']:
    source = ROOT.parent / 'data/Solomon' / (row['name'] + '.txt')
    assert digest(source) == row['source_hash'] == digest(out / source.name)
    for suffix, record in row['scenarios'].items():
        path = out / record['file']
        assert digest(path) == record['sha256']
        parsed = load_solomon(str(path))
        events = parsed['preset_events']
        assert len(events) == len(record['events'])
        if suffix == 'ACC2':
            waves[len(events)] += 1
        for event, original in zip(events, record['events']):
            for key in ['type', 'nodes', 'trigger_time', 'duration', 'multiplier']:
                assert event[key] == original[key], (path, key)
            assert event['nodes'] and set(event['nodes']) <= set(range(1, 101))
            assert event['duration'] > 0
            assert event['trigger_time'] < parsed['T'] and event['trigger_time'] + event['duration'] > 0
            if suffix == 'RAIN2':
                assert event['multiplier'] in (1.25, 1.5, 2.0)
                assert event['rainfall_mm'] == original['rainfall_mm']
                rain[event['multiplier']] += 1
            else:
                assert event['multiplier'] in (1.2, 2.04, 5.88)
                assert event['vehicles_involved'] == original['vehicles_involved']
                acc[event['multiplier']] += 1
assert len(list(out.glob('*.txt'))) == 45
report = dict(validated=True, instances=15, files=45, rain_severity=dict(rain),
              accident_severity=dict(acc), accident_wave_sizes=dict(waves),
              source_base_preserved=True, serialized_events_match=True)
(out / 'validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
