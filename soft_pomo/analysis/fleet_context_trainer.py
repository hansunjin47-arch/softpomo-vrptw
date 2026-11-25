"""Legacy-budget training with GPU environment and legacy model parameters."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
import traceback
import torch
from experiments.concurrent_fleet_v3.run import ROOT, SOURCE, load
from experiments.concurrent_fleet_pair.io import write, replace, reconcile_logs
from experiments.concurrent_fleet_pair.env import TensorFleetEnv
from experiments.concurrent_fleet_pair.policy import FleetPolicy
from train_vrptw import _set_seed, plot_training_curves


def append(path,row):
    with path.open('a',encoding='utf-8') as f: f.write(json.dumps(row)+'\n')


def exploration_epsilon(initial, epoch, zero_epoch, final=0.):
    return final+(initial-final)*max(0., 1.-(epoch-1)/max(1, zero_epoch-1))


@torch.no_grad()
def collect(model,inst,public,params,count,amp=True,greedy=False,progress=None):
    env=TensorFleetEnv(inst,count,params)
    starts,encodings=model.encode_events(inst,public)
    records=[]; history=[]
    for step in range(2*env.n+2*env.v+4):
        if bool(env.done.all()):
            return env,records,history
        frame=env.frame(); live=~env.done
        with torch.autocast('cuda',enabled=amp):
            dist=model.distribution(frame,starts,encodings,env)
            action=dist.logits.argmax(1) if greedy else dist.sample()
            logp=dist.log_prob(action)
        records.append(dict(**frame,action=action,logp=logp,live=live))
        new,vehicle,dest=env.decode_action(action,frame)
        history.append(torch.stack([env.time[:4],vehicle[:4],dest[:4],new[:4],live[:4]],1).clone())
        env.step(action)
        if progress is not None and step%20==0: progress(step)
    raise RuntimeError('Exceeded routing action bound')


def train_batch(model,opt,scaler,inst,public,params,batch,rollouts,amp=True,progress=None,max_steps=600):
    start=time.monotonic()
    env=TensorFleetEnv(inst,batch*rollouts,params)
    starts,encoded=model.encode_events(inst,public,batch_size=batch)
    probabilities=[]; history=[]
    # Match the legacy trainer: retain the complete trajectory graph, then
    # one scaled backward and one optimizer step. No replay or detach.
    with torch.autocast('cuda',enabled=amp):
        for step in range(max_steps):
            if bool(env.done.all()): break
            frame=env.frame(); live=~env.done
            probs=model.probabilities(frame,starts,encoded,env)
            with torch.no_grad():
                action=probs.multinomial(1).squeeze(1)
            selected=probs[env.ids,action]
            probabilities.append(torch.where(live,selected,1.).float().reshape(batch,rollouts))
            new,vehicle,dest=env.decode_action(action,frame)
            history.append(torch.stack([env.time[:100],vehicle[:100],dest[:100],new[:100],live[:100]],1).clone())
            env.step(action)
            if progress is not None and step%20==0: progress(step)
        else:
            raise RuntimeError('Exceeded configured max_steps; do not score truncated run as complete')
    metrics=env.metrics()
    reward=metrics['reward'].float().reshape(batch,rollouts)
    shifted=reward-reward[:, :1]
    advantage=shifted-shifted.mean(1,keepdim=True)
    raw_advantage_std=advantage.std(1,keepdim=True)
    if params.get('normalize_advantage_std', True):
        advantage=advantage/raw_advantage_std.clamp(min=1e-6)
    log_prob=torch.stack(probabilities,dim=2).log().sum(2)
    loss=-(advantage*log_prob).mean()
    model.zero_grad()
    if progress is not None: progress(step,True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    skipped=not bool(torch.isfinite(norm))
    if skipped and not amp: raise RuntimeError('Nonfinite gradient without AMP')
    scaler.step(opt); scaler.update()
    summary={k:float(v.double().mean()) for k,v in metrics.items()}
    summary.update(reward_std_min=float(raw_advantage_std.min()),
                   reward_std_mean=float(raw_advantage_std.mean()),
                   reward_std_max=float(raw_advantage_std.max()),
                   reward_std_below_1e4_fraction=float((raw_advantage_std<1e-4).float().mean()),
                   standardized_advantage_abs_max=float(advantage.abs().max()))
    summary.update(best_rollout_reward=float(reward.max(1).values.mean()),loss=float(loss.detach()),
                   gradient_norm=None if skipped else float(norm),optimizer_step_skipped=skipped,
                   amp_scale=scaler.get_scale(),seconds=time.monotonic()-start,
                   peak_gpu_mb=torch.cuda.max_memory_allocated()/1024**2)
    detail=dict(rollout_metrics={k:v[:100].cpu().tolist() for k,v in metrics.items()},
                decision_history=torch.stack(history).cpu().tolist(),
                history_columns=['time','vehicle','destination','new_vehicle','live'],
                history_rollouts=min(100,batch*rollouts),
                action_slots='NEW(customer) or (existing vehicle, destination); no SELECT',
                vehicle_histogram=torch.bincount(env.k,minlength=env.v+1).cpu().tolist())
    return summary,detail


def run(args):
    global SOURCE
    if getattr(args, 'source', None) is not None:
        from experiments.concurrent_fleet_v3 import run as source_module
        SOURCE = args.source.resolve()
        source_module.SOURCE = SOURCE
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required; no CPU fallback')
    if not 0 <= args.epsilon_final <= args.new_customer_epsilon <= 1 or args.epsilon_zero_epoch < 2:
        raise ValueError('Invalid NEW epsilon schedule')
    torch.set_num_threads(1)
    if args.out.exists() and not args.resume: raise ValueError('Use a new output directory')
    args.out.mkdir(parents=True,exist_ok=True)
    configs={}
    if getattr(args, 'config', None) is None:
        raise ValueError('Portable trainer requires an explicit config')
    if getattr(args, 'config', None) is not None:
        assert len(args.weights) == 1
        configs = {str(args.weights[0]):json.loads(args.config.read_text(encoding='utf-8'))}
    for config in configs.values():
        ep=config['environment']
        if getattr(args, 'seed', None) is not None:
            config['trainer']['seed'] = args.seed
        if getattr(args, 'return_mask', False):
            ep['protect_last_service_vehicle'] = True
        assert ep['late_penalty'] in args.weights
        if args.vehicle_cost is not None:
            if args.vehicle_cost <= 0:
                raise ValueError('vehicle-cost must be positive')
            ep['vehicle_penalty']=args.vehicle_cost*ep['problem_size']/getattr(args, 'reward_denominator', 1696.95)
        ep['normalize_advantage_std']=not args.no_advantage_std
    split=json.loads((SOURCE/'split.json').read_text()); rows={r['name']:r for r in split['train']}
    files=[Path(__file__)]+list((ROOT/'experiments/concurrent_fleet_pair').glob('*.py'))+[ROOT/'experiments/concurrent_fleet_v5/env.py',ROOT/'experiments/concurrent_fleet_v7/policy.py',ROOT/'experiments/concurrent_fleet_v8/env.py',ROOT/'experiments/concurrent_fleet_v8/policy.py',ROOT/'experiments/concurrent_fleet_v3/run.py']+[ROOT/'train_vrptw.py',ROOT/'vrptw_env.py',ROOT/'event_observation.py']
    protocol=dict(version='concurrent_fleet_pair_v1',device='cuda:0',gpu=torch.cuda.get_device_name(0),
                  configs=configs,weights=args.weights,smoke_updates=args.smoke_updates,
                  resume_from=str(args.resume_from) if args.resume_from else None,
                  checkpoint_every_updates=20, optimization='State/KV caching, valid-query packing, SDPA, non-reentrant decoder activation checkpointing with explicit cached KV inputs; batch128 x rollout100 unchanged',
                  model='Legacy routing encoder/decoder; shared vehicle/destination scoring; no SELECT parameters',
                  learning='Full-graph single backward; Adam, AMP/GradScaler, shifted mean baseline; std normalization controlled by configs; clip=1, batch128 x rollout100; no replay',
                  vehicle_cost_minutes=args.vehicle_cost,
                  new_customer_exploration=dict(initial=args.new_customer_epsilon,
                      final=args.epsilon_final, end_epoch=args.epsilon_zero_epoch,
                      schedule='linear by epoch; held at final after end epoch; evaluation disabled',
                      objective='sample and score the same mixture; NEW mass preserved; no detached probability'),
                  changes='At t0 NEW(customer) or existing(vehicle,destination); existing pair closes dispatch and reserves next leg. Pending leg auto-departs at service completion. No hidden ETA in policy; prebook slack channels zero. One-shot test.',
                  source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    if args.resume:
        if json.loads((args.out/'protocol.json').read_text())!=protocol: raise ValueError('Resume source/config mismatch')
    else:
        write(args.out/'protocol.json',protocol)
        for path in files:
            dest=args.out/'source'/path.relative_to(ROOT); dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,dest)
        shutil.copy2(SOURCE/'split.json',args.out/'split.json')
    for weight in args.weights:
        folder=args.out/f'late{weight}'; folder.mkdir(exist_ok=True)
        if (folder/'complete.json').exists(): continue
        config=configs[str(weight)]; tp=config['trainer']; ep=config['environment']
        _set_seed(tp['seed'])
        model=FleetPolicy(**config['model']).cuda()
        opt=torch.optim.Adam(model.parameters(),**config['optimizer']['optimizer'])
        amp=tp.get('use_amp',True); scaler=torch.amp.GradScaler('cuda',enabled=amp)
        state=dict(epoch=1,episodes=0,update=0,trajectories=0,epoch_reward_sum=0.,epoch_loss_sum=0.)
        reward_log=[]; loss_log=[]
        resume_checkpoint = folder/'checkpoint-last.pt'
        if not args.resume and args.resume_from is not None:
            resume_checkpoint = args.resume_from/f'late{weight}'/'checkpoint-last.pt'
        if (args.resume or args.resume_from is not None) and resume_checkpoint.exists():
            saved=torch.load(resume_checkpoint,weights_only=False)
            model.load_state_dict(saved['model_state_dict']); opt.load_state_dict(saved['optimizer_state_dict']); scaler.load_state_dict(saved['scaler'])
            reward_log=saved['reward_log']; loss_log=saved['loss_log']
            state=saved['state']; random.setstate(saved['python_rng']); torch.set_rng_state(saved['cpu_rng']); torch.cuda.set_rng_state(saved['cuda_rng'])
            reconcile_logs(folder,state['update'],len(reward_log))
        cache={}
        def data(row):
            if row['name'] not in cache: cache[row['name']]=load(row)
            return cache[row['name']]
        def snapshot(path):
            temp=path.with_suffix('.tmp')
            saved_state=dict(state); saved_rewards=list(reward_log); saved_losses=list(loss_log)
            # Probe-triggered snapshots precede the normal epoch bookkeeping.
            # Normalize the serialized copy so replay resumes at the next batch.
            if saved_state['episodes']==tp['train_episodes']:
                saved_rewards.append(saved_state['epoch_reward_sum']/saved_state['episodes'])
                saved_losses.append(saved_state['epoch_loss_sum']/saved_state['episodes'])
                saved_state.update(epoch=saved_state['epoch']+1,episodes=0,epoch_reward_sum=0.,epoch_loss_sum=0.)
            torch.save(dict(epoch=len(saved_rewards),model_state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),
                            reward_log=saved_rewards,loss_log=saved_losses,scaler=scaler.state_dict(),state=saved_state,
                            python_rng=random.getstate(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),temp)
            replace(temp,path)
        def checkpoint():
            snapshot(folder/'checkpoint-last.pt')
        incident_count=0
        best_probe=None
        previous_bad=False
        if args.resume:
            incident_count=len(list((folder/'incidents').glob('update_*')))
            for p in sorted((folder/'checks').glob('update_*.json')):
                prior=json.loads(p.read_text())
                if prior['update']>state['update'] or prior['epoch']<150: continue
                score=sum(r['reward'] for r in prior['results'])/len(prior['results'])
                best_probe=score if best_probe is None else max(best_probe,score)
                previous_bad=score<best_probe-.35 or any(r['unserved']>0 for r in prior['results'])
        def status(stage,**extra):
            write(args.out/'status.json',dict(stage=stage,weight=weight,**state,device='cuda:0',
                 target_epochs=tp['epochs'],episodes_per_epoch=tp['train_episodes'],rollouts=ep['pomo_size'],
                 gpu_allocated_mb=torch.cuda.memory_allocated()/1024**2,**extra))
        def probe():
            nonlocal incident_count,best_probe,previous_bad
            model.eval(); results=[]
            for name in (tp['train_instances'][0],tp['train_instances'][10]):
                inst,public=data(rows[name]); env,_,hist=collect(model,inst,public,ep,1,amp,greedy=True)
                results.append(dict(name=name,**{k:float(v[0]) for k,v in env.metrics().items()},
                                    history=torch.stack(hist).cpu().tolist()))
            write(folder/'checks'/f'update_{state["update"]:06d}.json',dict(role='fixed TRAIN probes',**state,results=results))
            if state['epoch']>=150:
                score=sum(r['reward'] for r in results)/len(results)
                unserved=sum(r['unserved'] for r in results)/len(results)
                best_probe=score if best_probe is None else max(best_probe,score)
                bad=score<best_probe-.35 or unserved>0
                if bad and not previous_bad:
                    append(folder/'instability_events.jsonl',dict(epoch=state['epoch'],update=state['update'],reward=score,best_train_probe=best_probe,unserved=unserved,snapshot_saved=incident_count<4))
                    if incident_count<4:
                        dest=folder/'incidents'/f'update_{state["update"]:06d}'
                        dest.mkdir(parents=True,exist_ok=False)
                        shutil.copy2(folder/'checkpoint-last.pt',dest/'before_interval.pt')
                        snapshot(dest/'after_interval.pt')
                        write(dest/'incident.json',dict(**state,reward=score,best_train_probe=best_probe,unserved=unserved,scope='Diagnostic only; never rolls back or selects training checkpoints',previous_checkpoint_update=state['update']-20,interval_updates=20))
                        incident_count+=1
                previous_bad=bad
            model.train()
        if state['update']==0: checkpoint(); probe()
        while state['epoch']<=tp['epochs']:
            if (args.out/'STOP').exists(): checkpoint(); status('paused'); return
            name=random.choice(tp['train_instances']); inst,public=data(rows[name])
            batch=min(tp['train_batch_size'],tp['train_episodes']-state['episodes'])
            def progress(step,backward=False):
                status('backward' if backward else 'sampling',decision_step=step,batch_size=batch,name=name)
            model.new_customer_epsilon=exploration_epsilon(args.new_customer_epsilon,state['epoch'],args.epsilon_zero_epoch,args.epsilon_final)
            summary,detail=train_batch(model,opt,scaler,inst,public,ep,batch,ep['pomo_size'],amp,progress,tp.get('max_steps',600))
            summary['new_customer_epsilon']=model.new_customer_epsilon
            state['update']+=1; state['episodes']+=batch; state['trajectories']+=batch*ep['pomo_size']
            state['epoch_reward_sum']+=summary['best_rollout_reward']*batch
            state['epoch_loss_sum']+=summary['loss']*batch
            row=dict(**state,name=name,batch_size=batch,**summary)
            append(folder/'training.jsonl',row)
            if state['update']<=20 or state['update']%20==0:
                write(folder/'early_history'/f'update_{state["update"]:06d}.json',dict(**row,**detail)); probe()
            if state['episodes']==tp['train_episodes']:
                reward_log.append(state['epoch_reward_sum']/state['episodes'])
                loss_log.append(state['epoch_loss_sum']/state['episodes'])
                append(folder/'epochs.jsonl',dict(epoch=state['epoch'],train_reward=reward_log[-1],loss=loss_log[-1]))
                state['epoch']+=1; state['episodes']=0
                state['epoch_reward_sum']=0.; state['epoch_loss_sum']=0.
            if state['update'] % 20 == 0:
                checkpoint()
            status('training',last_batch_seconds=summary['seconds'])
            if state['episodes']==0 and state['epoch']-1 in tuple(range(100,1001,100)):
                shutil.copy2(folder/'checkpoint-last.pt',folder/f'checkpoint-{state["epoch"]-1}.pt')
            print(json.dumps(dict(weight=weight,**row)),flush=True)
            if args.smoke_updates and state['update']>=args.smoke_updates:
                checkpoint(); status('smoke_complete',full_training=False); return
        (folder/'plots').mkdir(exist_ok=True)
        plot_training_curves(reward_log,loss_log,str(folder/'plots'))
        write(folder/'training_complete.json',dict(epoch=tp['epochs'],test_count=0,evaluation_pending=True))
        write(args.out/'status.json',dict(stage='training_complete',evaluation_pending=True))
        return
        status('evaluation'); model.eval(); results=[]
        for row in split['test']:
            inst,public=data(row); env,_,history=collect(model,inst,public,ep,1,amp,greedy=True)
            results.append(dict(name=row['name'],**{k:float(v[0]) for k,v in env.metrics().items()}))
            write(folder/'results.json',results)
            write(folder/'test_history'/(row['name']+'.json'),torch.stack(history).cpu().tolist())
        write(folder/'complete.json',dict(epoch=tp['epochs'],test_count=len(results),trajectories=state['trajectories']))
    write(args.out/'status.json',dict(stage='complete',weights=args.weights,full_training=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--return-mask',action='store_true')
    parser.add_argument('--reward-denominator',type=float,default=1696.95)
    parser.add_argument('--weights',type=int,nargs='+',default=[5])
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--resume-from',type=Path)
    parser.add_argument('--smoke-updates',type=int,default=0)
    parser.add_argument('--vehicle-cost',type=float,default=None,help='RC101 vehicle cost in minutes (reward denominator 1696.95)')
    parser.add_argument('--no-advantage-std',action='store_true')
    parser.add_argument('--new-customer-epsilon',type=float,default=0.)
    parser.add_argument('--epsilon-zero-epoch','--epsilon-end-epoch',dest='epsilon_zero_epoch',type=int,default=500)
    parser.add_argument('--epsilon-final',type=float,default=0.)
    args=parser.parse_args()
    try: run(args)
    except Exception:
        if args.out.exists(): write(args.out/'failure.json',dict(error=traceback.format_exc()))
        raise
