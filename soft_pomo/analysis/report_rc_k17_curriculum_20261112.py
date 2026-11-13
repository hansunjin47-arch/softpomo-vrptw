"""Saved free versus forced evaluation summaries for the K17 curriculum."""
import json
import statistics
import sys
from pathlib import Path
from experiments.concurrent_fleet_pair.io import write

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'result'/(sys.argv[1] if len(sys.argv)>1 else 'rc_k17_curriculum_20261112')
metrics=('reward','vehicles','travel_time','total_late','unserved')
results=[]
for arm in ('control','curriculum'):
    folder=OUT/arm
    for path in sorted(folder.glob('*_epoch*.json')):
        data=json.loads(path.read_text())
        results.append(dict(arm=arm,mode=path.name.split('_epoch')[0],epoch=int(path.stem.split('epoch')[-1]),
            mean={k:statistics.mean(r[k] for r in data) for k in metrics}))
logs={}
for arm in ('control','curriculum'):
    path=OUT/arm/'training.jsonl'
    if not path.exists():continue
    rows=[json.loads(l) for l in path.read_text().splitlines()]
    logs[arm]=dict(updates=len(rows),fallback_count=sum(r['fallback_count'] for r in rows),
        constrained_states=sum(r['constrained_states'] for r in rows),skips=sum(r['optimizer_step_skipped'] for r in rows),
        seconds=sum(r['seconds'] for r in rows))
write(OUT/'summary.json',dict(evaluations=results,training=logs))
for r in results:print(r['arm'],r['mode'],r['epoch'],r['mean'])
if all((OUT/arm/'free_epoch040.json').exists() for arm in ('control','curriculum')):
    control={r['name']:r for r in json.loads((OUT/'control/free_epoch040.json').read_text())}
    curriculum=json.loads((OUT/'curriculum/free_epoch040.json').read_text())
    paired=[dict(name=r['name'],delta={k:r[k]-control[r['name']][k] for k in metrics}) for r in curriculum]
    means={arm:next(r['mean'] for r in results if (r['arm'],r['mode'],r['epoch'])==(arm,'free',40)) for arm in ('control','curriculum')}
    write(OUT/'final_comparison.json',dict(means=means,delta={k:statistics.mean(r['delta'][k] for r in paired) for k in metrics},paired=paired,
        wins=sum(r['delta']['reward']>1e-5 for r in paired),ties=sum(abs(r['delta']['reward'])<=1e-5 for r in paired),losses=sum(r['delta']['reward'] < -1e-5 for r in paired),
        criterion_pass=means['curriculum']['reward']>means['control']['reward'] and means['curriculum']['unserved']==0,
        training=logs,limits=['Single starting checkpoint and Train20; no Test or independent seed replication.',
            'Passing this pilot criterion does not establish final1000epoch superiority.',
            'Constrained epoch20 improved forced routing but impaired free dispatch; subsequent free training was essential.']))
