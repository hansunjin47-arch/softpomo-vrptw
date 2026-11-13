"""Dispatch uses total advantage; routing excludes vehicle cost. Fresh C10."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
import traceback
import torch
from experiments.concurrent_fleet_v3.run import ROOT, SOURCE, load, write
from experiments.concurrent_fleet_v9.env import TensorFleetEnv
from experiments.concurrent_fleet_v9.policy import FleetPolicy
from experiments.concurrent_fleet_pair.io import write, replace
from train_vrptw import _set_seed, plot_training_curves
from experiments.concurrent_fleet_exploration.objective import dispatch_entropy, entropy_beta, first_customer_entropy
from experiments.concurrent_fleet_stop_init.policy import FleetPolicy as BiasedPolicy, calibrate


def centered(value):
    delta=value-value[:, :1]
    return delta-delta.mean(1,keepdim=True)


def phase_loss(reward,vehicle_cost,logp,dispatch):
    total_adv=centered(reward)
    # Preserves the original advantage exactly for a constant-K group.
    route_adv=total_adv+centered(vehicle_cost)
    dispatch_loss=-(total_adv*(logp*dispatch).sum(2)).mean()
    route_loss=-(route_adv*(logp*(~dispatch)).sum(2)).mean()
    return dispatch_loss,route_loss,total_adv,route_adv


def append(path,row):
    with path.open('a',encoding='utf-8') as f: f.write(json.dumps(row)+'\n')


@torch.no_grad()
def collect(model,inst,public,params,count,amp=True,greedy=False,progress=None,env_type=None):
    env=(env_type or TensorFleetEnv)(inst,count,params)
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
        vehicle,_,_=env.next_vehicle()
        vehicle=torch.where(action<env.n,env.k,vehicle)
        history.append(torch.stack([env.time[:4],vehicle[:4],action[:4],live[:4]],1).clone())
        env.step(action)
        if progress is not None and step%20==0: progress(step)
    raise RuntimeError('Exceeded routing action bound')


def train_batch(model,opt,scaler,inst,public,params,batch,rollouts,amp=True,progress=None,max_steps=600, beta=0.,env_type=None,first_beta=0.):
    start=time.monotonic()
    env=(env_type or TensorFleetEnv)(inst,batch*rollouts,params)
    starts,encoded=model.encode_events(inst,public,batch_size=batch)
    probabilities=[]; history=[]; phases=[]
    entropy_sum=torch.zeros(env.count,device=env.device)
    entropy_count=torch.zeros_like(entropy_sum)
    first_sum=torch.zeros_like(entropy_sum); first_count=torch.zeros_like(entropy_sum)
    first_sets=torch.zeros(env.count,env.n,dtype=torch.bool,device=env.device)
    # Match the legacy trainer: retain the complete trajectory graph, then
    # one scaled backward and one optimizer step. No replay or detach.
    with torch.autocast('cuda',enabled=amp):
        for step in range(max_steps):
            if bool(env.done.all()): break
            frame=env.frame(); live=~env.done
            probs=model.probabilities(frame,starts,encoded,env)
            ent, valid = dispatch_entropy(probs, frame['mask'], env.dispatch_open, live, env.n)
            entropy_sum = entropy_sum + ent
            entropy_count = entropy_count + valid.float()
            if first_beta:
                fh,fv=first_customer_entropy(probs,frame['mask'],env.dispatch_open,live,env.n)
                first_sum=first_sum+fh;first_count=first_count+fv.float()
            with torch.no_grad():
                action=probs.multinomial(1).squeeze(1)
                if first_beta:
                    chosen=live & env.dispatch_open & (action<env.n)
                    first_sets[env.ids[chosen],action[chosen]]=True
            selected=probs[env.ids,action]
            probabilities.append(torch.where(live,selected,1.).float().reshape(batch,rollouts))
            phases.append(env.dispatch_open.clone().reshape(batch,rollouts))
            vehicle,_,_=env.next_vehicle()
            vehicle=torch.where(action<env.n,env.k,vehicle)
            history.append(torch.stack([env.time[:4],vehicle[:4],action[:4],live[:4]],1).clone())
            env.step(action)
            if progress is not None and step%20==0: progress(step)
        else:
            raise RuntimeError('Exceeded configured max_steps; do not score truncated run as complete')
    metrics=env.metrics()
    reward=metrics['reward'].float().reshape(batch,rollouts)
    # Remove the common offset first so identical rewards center to exact zero.
    log_prob=torch.stack(probabilities,dim=2).log()
    dispatch_loss,route_loss,advantage,route_advantage=phase_loss(
        reward,metrics['cost_vehicle'].float().reshape(batch,rollouts),
        log_prob,torch.stack(phases,dim=2))
    policy_loss=dispatch_loss+route_loss
    entropy_mean=(entropy_sum/entropy_count.clamp(min=1)).mean()
    first_mean=(first_sum/first_count.clamp(min=1)).mean()
    loss=policy_loss-beta*entropy_mean-first_beta*first_mean
    model.zero_grad()
    if progress is not None: progress(step,True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    skipped=not bool(torch.isfinite(norm))
    if skipped and not amp: raise RuntimeError('Nonfinite gradient without AMP')
    scaler.step(opt); scaler.update()
    summary={k:float(v.double().mean()) for k,v in metrics.items()}
    summary.update(best_rollout_reward=float(reward.max(1).values.mean()),loss=float(loss.detach()),
                   gradient_norm=None if skipped else float(norm),optimizer_step_skipped=skipped,
                   amp_scale=scaler.get_scale(),seconds=time.monotonic()-start,
                   peak_gpu_mb=torch.cuda.max_memory_allocated()/1024**2)
    summary.update(policy_loss=float(policy_loss.detach()), dispatch_entropy=float(entropy_mean.detach()),
                   entropy_beta=beta, entropy_bonus=float((beta*entropy_mean).detach()),
                   reward_identical_fraction=float((reward==reward[:, :1]).all(1).float().mean()),
                   reward_group_std=float(reward.std(1).mean()),
                   advantage_abs_mean=float(advantage.abs().mean()), advantage_abs_max=float(advantage.abs().max()),
                   low_vehicle_fraction=float((env.k<=12).float().mean()),
                   vehicle_min=int(env.k.min()), vehicle_max=int(env.k.max()))
    summary.update(dispatch_policy_loss=float(dispatch_loss.detach()),route_policy_loss=float(route_loss.detach()),
                   route_advantage_abs_mean=float(route_advantage.abs().mean()),
                   vehicle_advantage_abs_mean=float((route_advantage-advantage).abs().mean()),
                   phase_advantage_sign_flip_fraction=float(((advantage*route_advantage<0)&
                       (advantage.abs()>1e-6)&(route_advantage.abs()>1e-6)).float().mean()))
    detail=dict(rollout_metrics={k:v[:100].cpu().tolist() for k,v in metrics.items()},
                decision_history=torch.stack(history).cpu().tolist(),
                history_columns=['time','vehicle','action_slot','live'],
                action_slots='0..N-1: NEW(customer=slot+1); N..2N: existing destination=slot-N; 2N+1: STOP_DISPATCH',
                vehicle_histogram=torch.bincount(env.k,minlength=env.v+1).cpu().tolist())
    if first_beta:
        sets=first_sets.reshape(batch,rollouts,env.n)
        unique=[int(torch.unique(s,dim=0).shape[0]) for s in sets]
        summary.update(first_entropy=float(first_mean.detach()),first_entropy_beta=first_beta,
                       first_entropy_bonus=float((first_beta*first_mean).detach()),
                       first_set_unique_mean=sum(unique)/len(unique))
    return summary,detail


def run(args):
    first_beta=getattr(args,'first_entropy_beta',0.)
    if first_beta<0: raise ValueError('Negative first entropy beta')
    if args.entropy_beta < 0: raise ValueError('Entropy beta must be nonnegative')
    deadline=time.monotonic()+args.max_hours*3600
    if args.resume_from is not None: raise ValueError('This experiment starts fresh; use --resume only for its own checkpoint')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required; no CPU fallback')
    torch.set_num_threads(1)
    if args.out.exists() and not args.resume: raise ValueError('Use a new output directory')
    args.out.mkdir(parents=True,exist_ok=True)
    configs={str(w):json.loads((ROOT/f'result/rc101_learned_first_train20_20261017/late{w}/config.json').read_text()) for w in args.weights}
    split=json.loads((SOURCE/'split.json').read_text()); rows={r['name']:r for r in split['train']}
    reference, _ = load(split['train'][0])
    for config in configs.values():
        ep = config['environment']
        ep['vehicle_penalty'] = args.vehicle_minutes * ep['time_penalty'] * reference['n_customers'] / (reference['T'] * ep['TT_max'])
    files=list(Path(__file__).parent.glob('*.py'))+list((ROOT/'experiments/concurrent_fleet_exploration').glob('*.py'))+list((ROOT/'experiments/concurrent_fleet_v9').glob('*.py'))+[ROOT/'experiments/concurrent_fleet_v8/env.py', ROOT/'experiments/concurrent_fleet_v8/policy.py', ROOT/'experiments/concurrent_fleet_pair/io.py']+[ROOT/'experiments/concurrent_fleet_v5/env.py',ROOT/'experiments/concurrent_fleet_v7/policy.py',ROOT/'experiments/concurrent_fleet_v3/run.py']+[ROOT/'train_vrptw.py',ROOT/'vrptw_env.py',ROOT/'event_observation.py']
    files += [ROOT/'experiments/concurrent_fleet_stop_init/policy.py']
    protocol=dict(version='concurrent_fleet_v9_split_reward_20261024',device='cuda:0',gpu=torch.cuda.get_device_name(0),
                  phase_reward='dispatch including NEW first customers and STOP: total; routing: total plus vehicle cost; independent centered advantages; shared parameters',
                  evaluation_reward='unchanged total reward; greedy one-shot test80',
                  configs=configs,weights=args.weights,smoke_updates=args.smoke_updates,vehicle_equivalent_minutes=args.vehicle_minutes,
                  advantage='shifted mean baseline, no std normalization',
                  arm=args.arm,stop_target=args.stop_target if args.arm=='stop_bias' else None,
                  early_checkpoints=[0,1,2,5,10,20,50,100,200,500,1000],
                  entropy=dict(initial_beta=args.entropy_beta,zero_epoch=args.entropy_zero_epoch,reduction='per-rollout valid-step mean then batch mean',scope='dispatch with both NEW and STOP valid'),
                  resume_from=str(args.resume_from) if args.resume_from else None,
                  checkpoint_every_updates=20, optimization='State-cache invalidation at each transition phase; per-encoding attention KV cache',
                  model='Sector report encoder, STOP token, dispatch context and capacity masks; clamp retained',
                  learning='Legacy full-graph single backward; encoder batch128 (no sharing); Adam, AMP/GradScaler, stable per-episode mean baseline WITHOUT std normalization, clip=1, batch128 x rollout100; no replay',
                  changes='Concurrent global clock and t0 dispatch action mapping; CUDA tensor environment. One-shot test.',
                  source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    if first_beta:
        protocol['first_customer_entropy']=dict(beta=first_beta,schedule='constant',reduction='valid-step mean then rollout mean',scope='conditional NEW distribution; at least two feasible customers')
    if args.resume:
        previous=json.loads((args.out/'protocol.json').read_text())
        if {k:v for k,v in previous.items() if k!='gpu'}!={k:v for k,v in protocol.items() if k!='gpu'}: raise ValueError('Resume source/config mismatch')
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
        model=(BiasedPolicy if args.arm=='stop_bias' else FleetPolicy)(**config['model'])
        if args.arm=='stop_bias' and not args.resume:
            report=calibrate(model,[load(row) for row in split['train']],ep,args.stop_target)
            write(folder/'stop_calibration.json',report)
        model=model.cuda()
        opt=torch.optim.Adam(model.parameters(),**config['optimizer']['optimizer'])
        amp=tp.get('use_amp',True); scaler=torch.amp.GradScaler('cuda',enabled=amp)
        state=dict(epoch=1,episodes=0,update=0,trajectories=0,epoch_reward_sum=0.,epoch_loss_sum=0.)
        reward_log=[]; loss_log=[]
        resume_checkpoint = folder/'checkpoint-last.pt'
        if args.resume and not resume_checkpoint.exists():
            raise FileNotFoundError(resume_checkpoint)
        if not args.resume and args.resume_from is not None:
            resume_checkpoint = args.resume_from/f'late{weight}'/'checkpoint-last.pt'
        if (args.resume or args.resume_from is not None) and resume_checkpoint.exists():
            saved=torch.load(resume_checkpoint,weights_only=False)
            model.load_state_dict(saved['model_state_dict']); opt.load_state_dict(saved['optimizer_state_dict']); scaler.load_state_dict(saved['scaler'])
            reward_log=saved['reward_log']; loss_log=saved['loss_log']
            state=saved['state']; random.setstate(saved['python_rng']); torch.set_rng_state(saved['cpu_rng']); torch.cuda.set_rng_state(saved['cuda_rng'])
        cache={}
        def data(row):
            if row['name'] not in cache: cache[row['name']]=load(row)
            return cache[row['name']]
        def checkpoint():
            temp=folder/'checkpoint.tmp'
            torch.save(dict(epoch=len(reward_log),model_state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),
                            reward_log=reward_log,loss_log=loss_log,scaler=scaler.state_dict(),state=state,
                            python_rng=random.getstate(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),temp)
            replace(temp, folder/'checkpoint-last.pt')
        def status(stage,**extra):
            write(args.out/'status.json',dict(stage=stage,weight=weight,**state,device='cuda:0',
                 target_epochs=tp['epochs'],episodes_per_epoch=tp['train_episodes'],rollouts=ep['pomo_size'],
                 gpu_allocated_mb=torch.cuda.memory_allocated()/1024**2,**extra))
        def probe():
            model.eval(); results=[]
            for name in (tp['train_instances'][0],tp['train_instances'][10]):
                inst,public=data(rows[name]); env,_,hist=collect(model,inst,public,ep,1,amp,greedy=True)
                results.append(dict(name=name,**{k:float(v[0]) for k,v in env.metrics().items()},
                                    history=torch.stack(hist).cpu().tolist()))
            write(folder/'checks'/f'update_{state["update"]:06d}.json',dict(role='fixed TRAIN probes',**state,results=results))
            model.train()
        if state['update']==0:
            checkpoint()
            shutil.copy2(folder/'checkpoint-last.pt',folder/'checkpoint-0.pt')
            initial=[]
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                model.eval()
                for name in (tp['train_instances'][0],tp['train_instances'][10]):
                    inst,public=data(rows[name])
                    env,_,_=collect(model,inst,public,ep,100,amp,greedy=False)
                    initial.append(dict(name=name,vehicle_histogram=torch.bincount(env.k,minlength=env.v+1).cpu().tolist(),
                                        **{k:float(v.double().mean()) for k,v in env.metrics().items()}))
            write(folder/'initial_rollouts.json',initial)
            probe()
        while state['epoch']<=tp['epochs']:
            if (args.out/'STOP').exists(): checkpoint(); status('paused'); return
            if time.monotonic()>=deadline: checkpoint(); status('needs_resume'); return
            name=random.choice(tp['train_instances']); inst,public=data(rows[name])
            batch=min(tp['train_batch_size'],tp['train_episodes']-state['episodes'])
            def progress(step,backward=False):
                status('backward' if backward else 'sampling',decision_step=step,batch_size=batch,name=name)
            summary,detail=train_batch(model,opt,scaler,inst,public,ep,batch,ep['pomo_size'],amp,progress,tp.get('max_steps',600),
                beta=entropy_beta(state['epoch'],args.entropy_beta,args.entropy_zero_epoch),first_beta=first_beta)
            if args.arm=='stop_bias': summary['stop_bias']=float(model.stop_bias.detach())
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
            if state['episodes']==0 and state['epoch']-1 in (1,2,5,10,20,50,100,200,500,1000):
                checkpoint()
                shutil.copy2(folder/'checkpoint-last.pt',folder/f'checkpoint-{state["epoch"]-1}.pt')
            print(json.dumps(dict(weight=weight,**row)),flush=True)
            if args.smoke_updates and state['update']>=args.smoke_updates:
                checkpoint(); status('smoke_complete',full_training=False); return
        (folder/'plots').mkdir(exist_ok=True)
        plot_training_curves(reward_log,loss_log,str(folder/'plots'))
        status('evaluation'); model.eval(); results=[]
        for row in split['test']:
            if time.monotonic()>=deadline: checkpoint(); status('needs_resume'); return
            inst,public=data(row); env,_,history=collect(model,inst,public,ep,1,amp,greedy=True)
            results.append(dict(name=row['name'],**{k:float(v[0]) for k,v in env.metrics().items()}))
            write(folder/'results.json',results)
            write(folder/'test_history'/(row['name']+'.json'),torch.stack(history).cpu().tolist())
        write(folder/'complete.json',dict(epoch=tp['epochs'],test_count=len(results),trajectories=state['trajectories']))
    write(args.out/'status.json',dict(stage='complete',weights=args.weights,full_training=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',choices=['control','stop_bias'],required=True)
    parser.add_argument('--stop-target',type=float,default=.15)
    parser.add_argument('--max-hours',type=float,default=10)
    parser.add_argument('--entropy-beta',type=float,default=0.)
    parser.add_argument('--entropy-zero-epoch',type=int,default=500)
    parser.add_argument('--vehicle-minutes',type=int,choices=[100,200,300,400,500,700,800,900],required=True)
    parser.add_argument('--first-entropy-beta',type=float,default=0.)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--weights',type=int,nargs='+',default=[5,10,1,20])
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--resume-from',type=Path)
    parser.add_argument('--smoke-updates',type=int,default=0)
    args=parser.parse_args()
    try: run(args)
    except Exception:
        if args.out.exists(): write(args.out/'failure.json',dict(error=traceback.format_exc()))
        raise
