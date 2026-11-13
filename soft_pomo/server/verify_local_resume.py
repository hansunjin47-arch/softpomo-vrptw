"""Validate portable source and paused-state imports without modifying checkpoints."""
import hashlib
import json
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parent
for name,digest in json.loads((ROOT/'manifest.json').read_text()).items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
run=ROOT/'runs/resumed'
assert not (run/'STOP').exists()
provenance=json.loads((ROOT/'migration.json').read_text())
checkpoint=run/'late10/checkpoint-last.pt'
saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
for k in ['model_state_dict','optimizer_state_dict','scaler','state','python_rng','cpu_rng','cuda_rng','reward_log','loss_log']:
    assert k in saved,k
state=saved['state']
assert state['update']>=provenance['state']['update']
if state['update']==provenance['state']['update']:
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==provenance['checkpoint_sha256']
    assert state==provenance['state']
from experiments.concurrent_fleet_v3.run import SOURCE,load
from experiments.concurrent_fleet_v9.policy import FleetPolicy
protocol=json.loads((run/'protocol.json').read_text())
config=protocol['configs']['10']
model=FleetPolicy(**config['model'])
model.load_state_dict(saved['model_state_dict'],strict=True)
opt=torch.optim.Adam(model.parameters(),**config['optimizer']['optimizer'])
opt.load_state_dict(saved['optimizer_state_dict'])
split=json.loads((SOURCE/'split.json').read_text())
assert len(split['train'])==20 and len(split['test'])==80
for row in split['train']+split['test']:load(row)
assert config['trainer']['epochs']==1000
print('Verified resume:',provenance['arm'],'epoch',state['epoch'],'episodes',state['episodes'],'update',state['update'])
