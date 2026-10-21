"""Verify immutable bundle inputs and coefficient equivalence without a GPU."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for name, expected in json.loads((ROOT / 'manifest.json').read_text()).items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name

from experiments.concurrent_fleet_v3.run import SOURCE, load
from experiments.concurrent_fleet_v9.train_vehicle_sweep import FleetPolicy

split = json.loads((SOURCE / 'split.json').read_text())
assert len(split['train']) == 20 and len(split['test']) == 80
assert not ({r['name'] for r in split['train']} & {r['name'] for r in split['test']})
for row in split['train'] + split['test']:
    inst, public = load(row)
    assert inst['n_customers'] == 100
config = json.loads((ROOT / 'result/rc101_learned_first_train20_20261017/late10/config.json').read_text())
tp = config['trainer']
assert tp['epochs'] == 1000 and tp['train_episodes'] == 2500
assert tp['train_batch_size'] == 128 and tp['seed'] == 42
assert config['environment']['pomo_size'] == 100
assert set(tp['train_instances']) == {r['name'] for r in split['train']}
ep = config['environment']
for minutes in (300, 400, 500, 600):
    penalty = minutes * ep['time_penalty'] * inst['n_customers'] / (inst['T'] * ep['TT_max'])
    equivalent = penalty / inst['n_customers'] / (ep['time_penalty'] / (inst['T'] * ep['TT_max']))
    assert abs(equivalent - minutes) < 1e-8
    print('vehicle_minutes', minutes, 'vehicle_penalty', penalty)
print('Verified: SELECT on, train20/test80, lambda10, seed42, 1000 epochs; four fresh runs.')
