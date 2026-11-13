"""CPU-only audit of completed RC101 Pure RL training logs, no evaluation."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'result/rc_pure_training_audit_20261111'
FIELDS=('reward','vehicles','travel_time','total_late','unserved','best_rollout_reward')


def read(path):
    return json.loads(path.read_text())


def mean(rows,fields=FIELDS):
    total=sum(r.get('batch_size',1) for r in rows)
    return {k:sum(r[k]*r.get('batch_size',1) for r in rows)/total for k in fields}


def main():
    OUT.mkdir(exist_ok=True)
    runs=[]
    for weight in (100,200):
        for seed in range(42,47):
            paths=list((ROOT/'result').glob(f'rc_pair_returnmask_v{weight}_l5_s{seed}_*'))
            assert len(paths)==1
            path=paths[0]; folder=path/'late5'
            protocol=read(path/'protocol.json');cfg=protocol['configs']['5']
            assert cfg['trainer']['seed']==seed and cfg['trainer']['epochs']==1000
            assert cfg['environment']['late_penalty']==5
            assert cfg['environment']['protect_last_service_vehicle'] and cfg['environment']['normalize_advantage_std']
            assert abs(cfg['environment']['vehicle_penalty']-weight*100/1696.95)<1e-8
            assert read(folder/'complete.json')['epoch']==1000 and read(folder/'complete.json')['test_count']==80
            source=path/'source/experiments/concurrent_fleet_pair/train.py'
            code=source.read_text()
            assert "state['epoch_reward_sum']+=summary['best_rollout_reward']*batch" in code
            assert "(tp['train_instances'][0],tp['train_instances'][10])" in code
            updates=[json.loads(line) for line in (folder/'training.jsonl').read_text().splitlines()]
            logged_epochs=[json.loads(line) for line in (folder/'epochs.jsonl').read_text().splitlines()]
            assert len(updates)==20000 and len(logged_epochs)==1000
            by_epoch=defaultdict(list)
            for row in updates:by_epoch[row['epoch']].append(row)
            epochs=[]
            for epoch,rows in sorted(by_epoch.items()):
                assert sum(r['batch_size'] for r in rows)==2500
                value=dict(epoch=epoch,**mean(rows))
                assert abs(value['best_rollout_reward']-logged_epochs[epoch-1]['train_reward'])<1e-6
                norms=[r['gradient_norm'] for r in rows if r['gradient_norm'] is not None]
                value.update(max_gradient_norm=max(norms) if norms else None,
                    amp_skips=sum(r['optimizer_step_skipped'] for r in rows),
                    nonfinite_gradient_records=sum(r['gradient_norm'] is None for r in rows))
                epochs.append(value)
            blocks=[]
            for start in (1,201,401,601,801):
                rows=[r for r in updates if start<=r['epoch']<=start+199]
                norms=[r['gradient_norm'] for r in rows if r['gradient_norm'] is not None]
                blocks.append(dict(start=start,end=start+199,**mean(rows),amp_skips=sum(r['optimizer_step_skipped'] for r in rows),max_gradient_norm=max(norms)))
            changes=[dict(epoch=e['epoch'],reward_delta=e['reward']-epochs[e['epoch']-2]['reward'],
                lateness_delta=e['total_late']-epochs[e['epoch']-2]['total_late'],
                vehicle_delta=e['vehicles']-epochs[e['epoch']-2]['vehicles'],
                gradient=e['max_gradient_norm'],amp_skips=e['amp_skips']) for e in epochs if e['epoch']>=801]
            worst=sorted(changes,key=lambda r:r['reward_delta'])[:5]
            probes=[]
            for epoch in range(800,1001):
                probe=read(folder/'checks'/f'update_{epoch*20:06d}.json')
                assert len(probe['results'])==2
                probes.append(dict(epoch=epoch,**mean(probe['results'],FIELDS[:-1])))
            probe_changes=[dict(epoch=p['epoch'],reward_delta=p['reward']-probes[i-1]['reward'],
                lateness_delta=p['total_late']-probes[i-1]['total_late'],vehicle_delta=p['vehicles']-probes[i-1]['vehicles'])
                for i,p in enumerate(probes) if i>0]
            results=read(folder/'results.json'); assert len(results)==80
            run=dict(weight=weight,late=5,seed=seed,folder=str(path.relative_to(ROOT)),
                settings_verified=True,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                definitions=dict(sampling='training.jsonl reward is all sampled rollouts mean before update; batch-size weighted',
                    epoch_train_reward='epochs.jsonl is mean best-of100 per episode, not sampling or greedy',
                    probe='two fixed TRAIN instances index0 and10, greedy after update, not Train20 average',
                    test='80 held-out instance greedy results from final saved run; no new evaluation'),
                blocks=blocks,epochs=epochs,last200_worst_sampling_drops=worst,
                last200_probe=probes,last200_worst_probe_drops=sorted(probe_changes,key=lambda r:r['reward_delta'])[:5],
                last200_probe_best=max(probes[1:],key=lambda r:r['reward']),
                last200_sampling_worst=min(epochs[800:],key=lambda r:r['reward']),
                final_sampling=epochs[-1],final_probe=probes[-1],final_test=mean(results,FIELDS[:-1]),
                last200_sampling_drops_over_point2=sum(r['reward_delta'] < -.2 for r in changes),
                last200_probe_drops_over_point2=sum(r['reward_delta'] < -.2 for r in probe_changes),
                last200_amp_skips=blocks[-1]['amp_skips'])
            runs.append(run)
            print(weight,seed,'lastsampling',round(run['final_sampling']['reward'],4),'probe',round(run['final_probe']['reward'],4),
                'test',round(run['final_test']['reward'],4),'worstdrop',worst[0],
                'probe_worst',run['last200_worst_probe_drops'][0],'skips',run['last200_amp_skips'],flush=True)
            (OUT/f'v{weight}_seed{seed}.json').write_text(json.dumps(run,indent=2),encoding='utf-8')
    summary=dict(settings='RC101 Pure RL concurrent maskON stdON lambda5 vehicle weight100/200 seeds42-46 completed1000',
        warning='Reward objectives differ between weight100 and200; never interpret their raw reward difference as performance improvement. Sampling/best rollout/two-probe/Test metrics are distinct.',
        runs=[{k:v for k,v in r.items() if k not in ('epochs','last200_probe')} for r in runs])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
