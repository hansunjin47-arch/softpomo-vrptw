"""Extract per-probe dispatch plateaus from saved epoch checks, CPU-only."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import statistics

from analysis.diagnose_rc_pure_training_20261111 import mean,FIELDS

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'result/rc_pure_training_audit_20261111'


def read(path):
    return json.loads(path.read_text())


def main():
    runs=[]
    for weight in (100,200):
        for seed in range(42,47):
            path=next((ROOT/'result').glob(f'rc_pair_returnmask_v{weight}_l5_s{seed}_*'))
            folder=path/'late5'
            by_name=defaultdict(list)
            for epoch in range(1,1001):
                rows=read(folder/'checks'/f'update_{epoch*20:06d}.json')['results']
                assert len(rows)==2
                for row in rows:
                    by_name[row['name']].append(dict(epoch=epoch,**{k:row[k] for k in FIELDS[:-1]}))
            probes={}
            for name,series in by_name.items():
                final_k=series[-1]['vehicles']
                first50=next((start+1 for start in range(951)
                    if all(r['vehicles']==final_k for r in series[start:start+50])),None)
                blocks=[]
                for start in range(0,1000,200):
                    block=series[start:start+200]
                    counts=Counter(r['vehicles'] for r in block)
                    max_count=max(counts.values())
                    blocks.append(dict(start=start+1,end=start+200,mean_k=statistics.mean(r['vehicles'] for r in block),
                        mode_k=sorted(k for k,n in counts.items() if n==max_count),k_counts=dict(counts),
                        final_k_fraction=sum(r['vehicles']==final_k for r in block)/200))
                changes=[dict(epoch=series[i]['epoch'],before=series[i-1]['vehicles'],after=series[i]['vehicles'])
                    for i in range(1,1000) if series[i]['vehicles']!=series[i-1]['vehicles']]
                probes[name]=dict(final_k=final_k,first_final_k_50epoch_start=first50,
                    blocks=blocks,snapshots={str(e):series[e-1] for e in (500,800,1000)},
                    transitions=changes,last200_transitions=[r for r in changes if r['epoch']>=801],
                    vehicle_series=[r['vehicles'] for r in series])
            test=read(folder/'results.json')
            updates=[json.loads(line) for line in (folder/'training.jsonl').read_text().splitlines()]
            by_period={key:defaultdict(list) for key in ('601_800','801_1000')}
            for row in updates:
                if row['epoch']>=601:
                    by_period['601_800' if row['epoch']<=800 else '801_1000'][row['name']].append(row)
            sampling=[]
            for name,rows in by_period['801_1000'].items():
                previous=mean(by_period['601_800'][name]);current=mean(rows)
                sampling.append(dict(name=name,is_fixed_probe=name in by_name,updates=len(rows),
                    mean=current,delta_previous_block={k:current[k]-previous[k] for k in FIELDS}))
            value=dict(weight=weight,seed=seed,probes=probes,
                test_k_distribution=dict(Counter(r['vehicles'] for r in test)),
                test_mean_k=statistics.mean(r['vehicles'] for r in test),
                last200_sampling_per_instance=sampling,
                other18_worst_lateness_changes=sorted((r for r in sampling if not r['is_fixed_probe']),
                    key=lambda r:r['delta_previous_block']['total_late'],reverse=True)[:3])
            runs.append(value)
            print(weight,seed,[(name, p['final_k'],p['first_final_k_50epoch_start'],
                p['blocks'][-1]['final_k_fraction'],len(p['last200_transitions'])) for name,p in probes.items()],
                value['test_k_distribution'],flush=True)
    result=dict(definitions=dict(plateau='Earliest start of 50 consecutive epochs matching final K; does not imply permanence after that window',
        probe='Two fixed TRAIN cases, not representative Train20 evaluation',
        sampling='Per-instance weighted rollout means, not greedy policy; random batches and trajectory tails may differ',
        comparability='Weights change objective; K may be compared descriptively but rewards across weights not interpreted as improvement'),runs=runs)
    (OUT/'dispatch_plateau.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
