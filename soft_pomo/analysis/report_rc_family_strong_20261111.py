"""CPU-only strong-arm matched milestone summary."""
import json
import statistics
from pathlib import Path
from experiments.concurrent_fleet_pair.io import write

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'result/rc_family_entropy_strong_20261111'
BASE=ROOT/'result/rc_family_entropy_pilot_v3_20261111'
metrics=('reward','vehicles','travel_time','total_late','unserved')
def read(p):return json.loads(p.read_text())

logs=[json.loads(line) for line in (OUT/'treatment/training.jsonl').read_text().splitlines()]
assert [r['name'] for r in logs]==read(BASE/'protocol.json')['data_order'][:len(logs)]
assert read(OUT/'treatment/initial.json')==read(BASE/'control/initial.json')
results=[]
for epoch in (20,40,60,80,100):
    path=OUT/'treatment'/f'train20_epoch{epoch:03d}.json'
    if not path.exists():continue
    rows=read(path);control=read(BASE/'control'/path.name);weak=read(BASE/'treatment'/path.name)
    us=[r for r in logs if r['epoch']==epoch]
    value=dict(epoch=epoch,sampling={k:sum(r[k]*r['batch'] for r in us)/2500 for k in (*metrics,'family_entropy','aux_loss')},
        train20={k:statistics.mean(r[k] for r in rows) for k in metrics},
        control={k:statistics.mean(r[k] for r in control) for k in metrics},
        weak={k:statistics.mean(r[k] for r in weak) for k in metrics},
        fixed2={k:statistics.mean(r[k] for r in read(OUT/'treatment/probes'/f'epoch{epoch:03d}.json')) for k in metrics},
        skips=sum(r['optimizer_step_skipped'] for r in logs if r['epoch']<=epoch),
        train_seconds=sum(r['seconds'] for r in logs if r['epoch']<=epoch))
    c={r['name']:r for r in control}
    value['paired']=[dict(name=r['name'],delta={k:r[k]-c[r['name']][k] for k in metrics}) for r in rows]
    results.append(value)
write(OUT/'milestone_summary.json',dict(results=results,initial_hash_matches_control=True,data_order_matches_control=True))
for r in results:print(r['epoch'],r['sampling'],r['train20'],'control',r['control']['reward'])
if results and results[-1]['epoch']==100:
    r=results[-1];deltas=[p['delta']['reward'] for p in r['paired']]
    write(OUT/'final_comparison.json',dict(final=r,
        delta_control={k:r['train20'][k]-r['control'][k] for k in metrics},
        wins=sum(d>1e-5 for d in deltas),ties=sum(abs(d)<=1e-5 for d in deltas),losses=sum(d < -1e-5 for d in deltas),
        preregistered_success=r['train20']['reward']>r['control']['reward'] and r['train20']['unserved']==0,
        conclusion='Stronger-signal pilot failed the preregistered Train20 reward criterion; no further coefficient sweep.',
        limitations=['Single seed100epochs; not a general rejection of entropy methods.',
                     'No Test/API evaluation; no generalization claim.',
                     'Reused fresh control, not historical seed44 baseline.']))
