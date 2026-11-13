"""Observed replay discrepancy, not a repeated-seed variance estimate."""
import json
import statistics
from pathlib import Path
from experiments.concurrent_fleet_pair.io import write

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'result/rc_family_control_repeat_20261111'
BASE=ROOT/'result/rc_family_entropy_pilot_v3_20261111'
def read(p):return json.loads(p.read_text())
metrics=('reward','vehicles','travel_time','total_late','unserved')
old=[json.loads(l) for l in (BASE/'control/training.jsonl').read_text().splitlines()]
new=[json.loads(l) for l in (OUT/'control/training.jsonl').read_text().splitlines()]
assert [r['name'] for r in new]==[r['name'] for r in old[:len(new)]]
first=next((dict(update=a['update'],old_reward=a['reward'],repeat_reward=b['reward']) for a,b in zip(old,new) if a['reward']!=b['reward']),None)
results=[]
for epoch in (20,40,60,80,100):
    path=OUT/'control'/f'train20_epoch{epoch:03d}.json'
    if not path.exists():continue
    a=read(BASE/'control'/path.name);b=read(path);lookup={r['name']:r for r in a}
    pair=[dict(name=r['name'],delta={k:r[k]-lookup[r['name']][k] for k in metrics}) for r in b]
    rs=[r for r in new if r['epoch']==epoch]
    results.append(dict(epoch=epoch,original={k:statistics.mean(r[k] for r in a) for k in metrics},
        repeat={k:statistics.mean(r[k] for r in b) for k in metrics},paired=pair,
        sampling={k:sum(r[k]*r['batch'] for r in rs)/2500 for k in (*metrics,'family_entropy')},
        delta={k:statistics.mean(r['delta'][k] for r in pair) for k in metrics},
        cumulative_skips=sum(r['optimizer_step_skipped'] for r in new if r['epoch']<=epoch),
        train_seconds=sum(r['seconds'] for r in new if r['epoch']<=epoch)))
write(OUT/'comparison.json',dict(first_training_reward_difference=first,actual_instance_order_identical=True,
    epoch0_identical=read(OUT/'control/train20_epoch000.json')==read(BASE/'control/train20_epoch000.json'),
    milestones=results,limits=['One repeat quantifies one observed discrepancy, not a stable variance estimate or significance.',
        'No Test results; do not generalize100epoch Train performance.',
        'Initial state_dict SHA and epoch0 metrics verified; historical initial RNG was not separately persisted.']))
print('first',first)
for r in results:print(r['epoch'],r['repeat'],'delta',r['delta'])
