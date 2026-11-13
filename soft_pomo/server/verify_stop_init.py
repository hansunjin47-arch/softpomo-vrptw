"""Verify packaged inputs, experiment settings and imports before GPU training."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
for name,expected in json.loads((ROOT/'manifest.json').read_text()).items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
from experiments.concurrent_fleet_v3.run import SOURCE,load
from experiments.concurrent_fleet_stop_init.train import FleetPolicy,BiasedPolicy
split=json.loads((SOURCE/'split.json').read_text())
assert len(split['train'])==20 and len(split['test'])==80
assert not ({r['name'] for r in split['train']} & {r['name'] for r in split['test']})
for row in split['train']+split['test']:
    inst,public=load(row)
    assert inst['n_customers']==100
cfg=json.loads((ROOT/'result/rc101_learned_first_train20_20261017/late10/config.json').read_text())
t=cfg['trainer'];e=cfg['environment']
assert (t['epochs'],t['train_episodes'],t['train_batch_size'],t['seed'],e['pomo_size'])==(1000,2500,128,42,100)
assert e['late_penalty']==10
assert set(t['train_instances'])=={r['name'] for r in split['train']}
print('Verified: train20/test80, seed42, 1000 epochs, two fresh no-std/no-entropy arms.')
