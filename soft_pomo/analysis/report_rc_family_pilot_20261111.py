"""Summarize observed pilot milestones without evaluating models."""
import json
import statistics
from pathlib import Path
from experiments.concurrent_fleet_pair.io import write

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'result/rc_family_entropy_pilot_v3_20261111'
metrics=('reward','vehicles','travel_time','total_late','unserved')
results=[]
for arm in ('control','treatment'):
    folder=OUT/arm
    if not (folder/'training.jsonl').exists():continue
    updates=[json.loads(line) for line in (folder/'training.jsonl').read_text().splitlines()]
    for epoch in (20,40,60,80,100):
        allpath=folder/f'train20_epoch{epoch:03d}.json'
        if not allpath.exists():continue
        rs=[r for r in updates if r['epoch']==epoch]
        assert len(rs)==20 and sum(r['batch'] for r in rs)==2500
        value=dict(arm=arm,epoch=epoch,
            sampling={k:sum(r[k]*r['batch'] for r in rs)/2500 for k in (*metrics,'family_entropy','new_mass','aux_loss','policy_loss')},
            epoch_skips=sum(r['optimizer_step_skipped'] for r in rs),
            cumulative_skips=sum(r['optimizer_step_skipped'] for r in updates if r['epoch']<=epoch),
            epoch_train_seconds=sum(r['seconds'] for r in rs),
            cumulative_train_seconds=sum(r['seconds'] for r in updates if r['epoch']<=epoch),
            peak_gpu_mb=max(r['peak_gpu_mb'] for r in updates if r['epoch']<=epoch))
        for label,path in (('fixed2',folder/'probes'/f'epoch{epoch:03d}.json'),('train20',allpath)):
            data=json.loads(path.read_text())
            value[label]={k:statistics.mean(r[k] for r in data) for k in metrics}
        results.append(value)
oldpath=ROOT/'result/rc_pair_returnmask_v100_l5_s44_20261102/late5/training.jsonl'
oldorder=[json.loads(line)['name'] for line in oldpath.read_text().splitlines()][:2000]
neworder=json.loads((OUT/'protocol.json').read_text())['data_order']
write(OUT/'milestone_summary.json',dict(results=results,original_data_order_comparison=dict(
    updates=2000,matching=sum(a==b for a,b in zip(oldorder,neworder)),identical=oldorder==neworder),limitations=[
    'Original historical run and fresh control need not reproduce bitwise; actual first2000 data order is identical. Cause of changed plateau not isolated.',
    'Sampling entropy is recorded; saturation fraction and auxiliary/PG gradient ratio were not instrumented.',
    'Train2/20 greedy metrics are not Test performance; single seed100epochs is only a pilot.']))
for row in results:
    print(row['arm'],row['epoch'],'sample',row['sampling'],'Train20',row['train20'],'skips',row['epoch_skips'])
if all((OUT/arm/'train20_epoch100.json').exists() for arm in ('control','treatment')):
    base={r['name']:r for r in json.loads((OUT/'control/train20_epoch100.json').read_text())}
    treated=json.loads((OUT/'treatment/train20_epoch100.json').read_text())
    paired=[dict(name=r['name'],delta={k:r[k]-base[r['name']][k] for k in metrics}) for r in treated]
    final={r['arm']:r for r in results if r['epoch']==100}
    write(OUT/'final_comparison.json',dict(scope='RC101 v100 lambda5 maskON single seed44 fresh paired100epoch pilot; no Test',
        control=final['control'],treatment=final['treatment'],
        train20_delta={k:statistics.mean(r['delta'][k] for r in paired) for k in metrics},
        wins=sum(r['delta']['reward']>1e-5 for r in paired),ties=sum(abs(r['delta']['reward'])<=1e-5 for r in paired),
        losses=sum(r['delta']['reward'] < -1e-5 for r in paired),paired=sorted(paired,key=lambda r:r['delta']['reward']),
        conclusion='This fixed coefficient/schedule pilot did not improve the matched fresh control. No evidence of improved final Train20 performance.',
        limitations=['One seed and100epochs only, not general proof against entropy methods.',
                     'Fresh control historical-seed44 plateau differs despite identical data order; historical reproduction not established.',
                     'Trajectory-dependent RNG consumption and CUDA arithmetic need not be identical after policies differ.',
                     'Saturation ratio and auxiliary gradient norm ratio not logged; entropy and loss ratios do not substitute for them.',
                     'Control epoch20 checkpoint missed; control40/60/80/100 and all treatment milestones preserved.']))
