"""Compare saved K18 control to identically initialized K17 and free arms."""
import json
import statistics
from pathlib import Path
from experiments.concurrent_fleet_pair.io import write

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'result/rc_pure_k17_curriculum_20261112'
OUT = ROOT/'result/rc_pure_k18_curriculum_20261112'
read = lambda p: json.loads(p.read_text())
metrics = ('reward','vehicles','travel_time','total_late','unserved')
paths = {'control': OLD/'control', 'k17': OLD/'curriculum', 'k18': OUT/'curriculum'}
a, b = read(OLD/'protocol.json'), read(OUT/'protocol.json')
for key in ('config','data_order','input_hashes','checkpoint'):
    assert a[key] == b[key], key
cases = {arm: read(path/'free_epoch040.json') for arm,path in paths.items()}
means = {arm:{k:statistics.mean(r[k] for r in rows) for k in metrics} for arm,rows in cases.items()}
comparisons = {}
for baseline in ('control','k17'):
    lookup = {r['name']:r for r in cases[baseline]}
    paired = [dict(name=r['name'],delta={k:r[k]-lookup[r['name']][k] for k in metrics}) for r in cases['k18']]
    comparisons[baseline] = dict(delta={k:statistics.mean(r['delta'][k] for r in paired) for k in metrics},
        wins=sum(r['delta']['reward']>1e-5 for r in paired),ties=sum(abs(r['delta']['reward'])<=1e-5 for r in paired),
        losses=sum(r['delta']['reward'] < -1e-5 for r in paired),paired=paired)
logs = [json.loads(line) for line in (OUT/'curriculum/training.jsonl').read_text().splitlines()]
report = dict(means=means,k18_minus=comparisons,matched_config_order_hash=True,
    criterion_pass=means['k18']['reward']>means['control']['reward'] and means['k18']['unserved']==0,
    training=dict(updates=len(logs),skips=sum(r['optimizer_step_skipped'] for r in logs),fallback=sum(r['fallback_count'] for r in logs),seconds=sum(r['seconds'] for r in logs)),
    limitation='Single seed44 Train20 mechanism contrast, not Test generalization or final training recommendation.')
write(OUT/'mechanism_comparison.json',report)
print(json.dumps({k:v for k,v in report.items() if k!='k18_minus'},indent=2))
print(json.dumps({k:{q:v for q,v in r.items() if q!='paired'} for k,r in comparisons.items()},indent=2))
