"""Saved-result only examination of late-pilot degradation."""
import json
import statistics
from analysis.report_rc_family_pilot_20261111 import OUT,metrics
from experiments.concurrent_fleet_pair.io import write


def read(path):return json.loads(path.read_text())


def stats(rows):
    return {k:dict(mean=statistics.mean(r[k] for r in rows),minimum=min(r[k] for r in rows),maximum=max(r[k] for r in rows)) for k in metrics}


report=[]
for arm in ('control','treatment'):
    folder=OUT/arm
    old={r['name']:r for r in read(folder/'train20_epoch080.json')}
    new=read(folder/'train20_epoch100.json')
    pairs=[dict(name=r['name'],kind='ACC' if '_ACC_' in r['name'] else 'RAIN',
                before=old[r['name']],after=r,delta={k:r[k]-old[r['name']][k] for k in metrics}) for r in new]
    groups={kind:dict(count=len(rs),delta={k:statistics.mean(r['delta'][k] for r in rs) for k in metrics},
        reward_wins=sum(r['delta']['reward']>1e-5 for r in rs),reward_losses=sum(r['delta']['reward'] < -1e-5 for r in rs),
        changed_k=sum(r['delta']['vehicles']!=0 for r in rs)) for kind in ('RAIN','ACC') if (rs:=[r for r in pairs if r['kind']==kind])}
    logs=[json.loads(line) for line in (folder/'training.jsonl').read_text().splitlines()]
    curve=[]
    for epoch in range(80,101):
        probes=read(folder/'probes'/f'epoch{epoch:03d}.json')
        updates=[r for r in logs if r['epoch']==epoch]
        curve.append(dict(epoch=epoch,probes=probes,mean={k:statistics.mean(r[k] for r in probes) for k in metrics},
            max_gradient_norm=max(r['gradient_norm'] for r in updates if r['gradient_norm'] is not None),
            skips=sum(r['optimizer_step_skipped'] for r in updates),eta=updates[0]['eta']))
    report.append(dict(arm=arm,groups=groups,paired=sorted(pairs,key=lambda r:r['delta']['reward']),fixed2_curve=curve,
        milestones={epoch:stats(read(folder/f'train20_epoch{epoch:03d}.json')) for epoch in (0,20,40,60,80,100)}))
write(OUT/'late_diagnosis.json',dict(arms=report,limits=[
    'Train20 available only e80/e100, cannot identify intermediate full-set onset.',
    'Fixed2 is a separate longitudinal sample, not a representative substitute for all20.',
    'Sampling batches change instances; use fixed-case comparison for degradation evidence.',
    'Preclip gradient and skips are observational, not causal attribution to auxiliary term.',
    'Auxiliary gradient norm was not logged. A matched frozen minibatch gradient decomposition could quantify direct signal, not prove training benefit.']))
for arm in report:
    print(arm['arm'],arm['groups'])
    print([(r['epoch'],round(r['mean']['reward'],4),round(r['mean']['total_late'],2),r['mean']['vehicles'],round(r['max_gradient_norm'],2),r['skips']) for r in arm['fixed2_curve']])
