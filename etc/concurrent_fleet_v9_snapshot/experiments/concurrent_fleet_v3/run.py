"""CPU pilot: python -m experiments.concurrent_fleet_v1.run --updates 2.

This is a new centralized REINFORCE prototype, not a checkpoint-compatible
POMO reproduction. Each rollout owns an independent concurrent fleet.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import torch
from vrptw_env import load_solomon
from experiments.concurrent_fleet_v3.env import FleetEnv
from experiments.concurrent_fleet_v3.policy import FleetPolicy, rollout

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'result/rc101_phase2_online_gpt_train20_v2_2_20261018/source'


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2),encoding='utf-8');tmp.replace(path)


def load(row):
    path=SOURCE/'data'/(row['name']+'.txt')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
    return load_solomon(str(path)),[json.loads(path.with_suffix('.observations.json').read_text())]


def diagnostics(env, metrics):
    moving=set();peak=0
    for row in env.trace:
        if row['event']=='depart':
            moving.add(row['vehicle'])
        elif row['event']=='arrival':
            moving.discard(row['vehicle'])
        peak=max(peak,len(moving))
    assert metrics['vehicles']<=env.vehicle_limit
    assert all(a['time']<=b['time'] for a,b in zip(env.trace,env.trace[1:]))
    visited=[n for route in metrics['routes'] for n in route if n]
    assert len(visited)==len(set(visited))==metrics['served']
    assert all(v['departures'][0]==0 for v in env.vehicles)
    streak=longest=idle=0
    for row in env.trace:
        if row['time']>0:
            break
        if row['event']=='depart':
            streak+=1;longest=max(longest,streak)
        elif row['event']=='select_existing':
            streak=0;idle+=1
    return dict(peak_moving=peak,initial_dispatches=sum(v['departures'][0]==0 for v in env.vehicles),
                initial_new_streak=longest,initial_idle_decisions=idle,
                **{k:metrics[k] for k in ('vehicles','served','unserved','feasible','travel_time','total_late','late_count','reward','finish_time')})


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--updates',type=int,default=2)
    ap.add_argument('--rollouts',type=int,default=2)
    ap.add_argument('--eval-limit',type=int,default=4)
    ap.add_argument('--check-every',type=int,default=20)
    ap.add_argument('--weights',type=int,nargs='+',default=[5,10])
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.rollouts<2 or args.updates<1 or args.eval_limit<1 or args.check_every<1:
        ap.error('Need >=2 rollouts and positive updates/eval-limit')
    if args.out.exists():
        ap.error('Use a new output directory to preserve previous versions')
    args.out.mkdir(parents=True)
    torch.set_num_threads(1)
    split=json.loads((SOURCE/'split.json').read_text())
    # Alternate event families rather than using only the first rain instances.
    train=[x for pair in zip(split['train'][:10],split['train'][10:]) for x in pair]
    rain=sorted([r for r in split['test'] if r['kind']=='RAIN'],key=lambda r:r['name'])
    accident=sorted([r for r in split['test'] if r['kind']!='RAIN'],key=lambda r:r['name'])
    test=[r for pair in zip(rain,accident) for r in pair][:args.eval_limit]
    snapshot=args.out/'source';snapshot.mkdir()
    files=list(Path(__file__).parent.glob('*.py'))+[ROOT/'vrptw_env.py',ROOT/'train_vrptw.py',ROOT/'advantage.py',ROOT/'event_observation.py']
    for f in files:
        shutil.copy2(f,snapshot/f.name)
    configs={str(w):json.loads((ROOT/f'result/rc101_learned_first_train20_20261017/late{w}/config.json').read_text()) for w in args.weights}
    manifest=dict(version='concurrent_fleet_v3_t0_dispatch',device='cpu',seed=42,
                  updates=args.updates,rollouts=args.rollouts,weights=args.weights,
                  train_names=[x['name'] for x in train],test_names=[x['name'] for x in test],
                  objective='Exact environment reward coefficients from existing per-lambda config, including unserved penalty',
                  original_configs=configs,
                  constraints='Solomon vehicle_limit; single trip per vehicle; capacity; soft TW including depot return; no forced initial fleet',
                  traffic='arc entry; ongoing arcs are not rescheduled',
                  dispatch='NEW(customer) only at t=0; select earliest existing vehicle before choosing its destination; no idle/advance action',
                  clock='after t>0 environment assigns earliest service completion, ties by vehicle id; immediate departure',
                  learning='REINFORCE shared-rollout mean baseline and std-normalized advantage matching legacy; original encoder dimensions; new decoder; diagnostic budget',
                  evaluation='one greedy rollout; pilot only, not full test80',
                  source_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    write(args.out/'protocol.json',manifest)
    for weight in args.weights:
        torch.manual_seed(42)
        config=configs[str(weight)]
        model=FleetPolicy(config['model'])
        optimizer=torch.optim.Adam(model.parameters(),**config['optimizer']['optimizer'])
        def environment(inst,public):
            return FleetEnv(inst,public,weight,environment_params=config['environment'])
        folder=args.out/f'late{weight}'
        log=[]
        def probe_at(update):
            model.eval();probe=[]
            with torch.no_grad():
                for entry in train[:2]:
                    pi,po=load(entry);pe=environment(pi,po)
                    pm,_=rollout(model,pe,model.encode(pi),greedy=True)
                    prefix=[r for r in pe.trace if r['event'] in ('depart','select_existing')][:30]
                    probe.append(dict(name=entry['name'],**diagnostics(pe,pm),decision_prefix=prefix))
            write(folder/'checks'/f'update_{update:05d}.json',dict(update=update,
                  role='fixed TRAIN probes, not held-out test',results=probe))
            print('CHECK '+json.dumps(dict(weight=weight,update=update,
                  results=[{k:v for k,v in r.items() if k!='decision_prefix'} for r in probe])),flush=True)
        probe_at(0)
        def checkpoint(update):
            folder.mkdir(parents=True,exist_ok=True)
            temp=folder/'checkpoint.tmp'
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                            protocol=manifest,weight=weight,update=update,
                            rng_state=torch.get_rng_state()),temp)
            temp.replace(folder/'checkpoint-last.pt')
        for update in range(args.updates):
            if (args.out/'STOP').exists():
                checkpoint(update)
                write(args.out/'status.json',dict(stage='paused',weight=weight,update=update))
                return
            inst,public=load(train[update%len(train)])
            model.train()
            rewards=[];samples=[];states=[];traces=[]
            # Sample first, then replay each trajectory with gradients. The
            # shared baseline is unchanged; only one rollout graph stays live.
            with torch.no_grad():
                encoded=model.encode(inst)
                for _ in range(args.rollouts):
                    states.append(torch.get_rng_state())
                    env=environment(inst,public)
                    metrics,_=rollout(model,env,encoded)
                    samples.append(diagnostics(env,metrics))
                    rewards.append(metrics['reward']);traces.append(env.trace)
            next_rng=torch.get_rng_state()
            r=torch.tensor(rewards)
            advantage=r-r.mean()
            advantage=advantage/advantage.std().clamp(min=1e-6)
            optimizer.zero_grad();loss_value=0.
            for index,rng in enumerate(states):
                torch.set_rng_state(rng)
                env=environment(inst,public)
                metrics,lp=rollout(model,env,model.encode(inst))
                assert metrics['reward']==rewards[index] and env.trace==traces[index]
                loss=-advantage[index]*lp/args.rollouts
                loss.backward();loss_value+=float(loss.detach())
                del loss,lp
            torch.set_rng_state(next_rng)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            assert torch.isfinite(norm) and torch.isfinite(torch.tensor(loss_value))
            optimizer.step()
            log.append(dict(update=update+1,name=inst['name'],mean_reward=float(r.mean()),
                            loss=loss_value,gradient_norm=float(norm),
                            **{'mean_'+k:statistics.mean(s[k] for s in samples) for k in samples[0] if k!='reward'}))
            write(folder/'training_log.json',log)
            checkpoint(update+1)
            write(args.out/'status.json',dict(stage='training',weight=weight,update=update+1,target=args.updates))
            print(f'lambda={weight} update={update+1} reward={r.mean():.5f} gradient={norm:.5f}',flush=True)
            if (update+1)%args.check_every==0 or update+1==args.updates:
                checkpoint(update+1)
                probe_at(update+1)
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),protocol=manifest,weight=weight),folder/'checkpoint.pt')
        model.eval();results=[]
        with torch.no_grad():
            for row in test:
                inst,public=load(row);env=environment(inst,public)
                metrics,_=rollout(model,env,model.encode(inst),greedy=True)
                results.append(dict(name=inst['name'],**metrics))
                write(folder/'traces'/(inst['name']+'.json'),env.trace)
        write(folder/'results.json',results)
        write(folder/'summary.json',{key:statistics.mean(x[key] for x in results)
              for key in ('reward','vehicles','travel_time','total_late','late_count','served','unserved','feasible','finish_time')})
    write(args.out/'status.json',dict(stage='complete',weights=args.weights,full_training=False))


if __name__=='__main__':
    main()
