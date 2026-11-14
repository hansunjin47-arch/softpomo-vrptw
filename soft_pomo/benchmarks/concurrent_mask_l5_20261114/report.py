"""Print archived matched-setting Test80 comparisons; no ML dependencies."""
import json, statistics, math
from pathlib import Path
from collections import defaultdict
runs=json.loads(Path(__file__).with_name('runs.json').read_text(encoding='utf-8'))
groups=defaultdict(list)
for r in runs: groups[(r['case'],r['vehicle_weight'],r['method'])].append(r)
print('| Case | Vehicle weight | Method | Seeds | Reward | Vehicles | Travel time | Late |')
print('|---|---:|---|---|---:|---:|---:|---:|')
for (case,w,method),rs in sorted(groups.items()):
    assert all({x['name'] for x in r['results']}=={x['name'] for x in rs[0]['results']} for r in rs)
    cells=[]
    for k in ('reward','vehicles','travel_time','total_late'):
        vals=[r['mean'][k] for r in rs];m=statistics.mean(vals)
        cells.append(f'{m:.2f} \u00b1 {1.96*statistics.stdev(vals)/math.sqrt(len(vals)):.2f}' if len(vals)>1 else f'{m:.2f}')
    seeds=','.join(str(r['seed']) for r in rs) if method!='LKH-3' else 'single run'
    print('| '+' | '.join([case,str(w),method,seeds]+cells)+' |')
print('All RL runs: simultaneous routing, lambda=5, return mask ON, epoch1000, Test80. Mean \u00b1 95% CI half-width (1.96 sample SD / sqrt(n)); single runs have no CI.')
