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
from experiments.concurrent_fleet_v3.run import ROOT, SOURCE, load, write
from experiments.concurrent_fleet_v4.env import TensorFleetEnv
from experiments.concurrent_fleet_v4.policy import FleetPolicy


def append(path,row):
    with path.open('a',encoding='utf-8') as f: f.write(json.dumps(row)+'\n')


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
        vehicle,_,_=env.next_vehicle()
        vehicle=torch.where(action<env.n,env.k,vehicle)
        history.append(torch.stack([env.time[:4],vehicle[:4],action[:4],live[:4]],1).clone())
        env.step(action)
        if progress is not None and step%20==0: progress(step)
    raise RuntimeError('Exceeded routing action bound')


def train_batch(model,opt,scaler,inst,public,params,batch,rollouts,amp=True,progress=None):
    start=time.monotonic()
    env,records,history=collect(model,inst,public,params,batch*rollouts,amp,progress=progress)
    metrics=env.metrics()
    reward=metrics['reward'].float().reshape(batch,rollouts)
    advantage=reward-reward.mean(1,keepdim=True)
    advantage=advantage/advantage.std(1,keepdim=True).clamp(min=1e-6)
    advantage=advantage.flatten()
    opt.zero_grad()
    starts,original=model.encode_events(inst,public)
    leaves=[e.detach().requires_grad_() for e in original]
    loss_total=0.
    # Backpropagate decoder steps separately, then propagate the accumulated
    # encoding gradient once. No parameter update occurs during this replay.
    for step,frame in enumerate(records):
        with torch.autocast('cuda',enabled=amp):
            dist=model.distribution(frame,starts,leaves,env)
            logp=dist.log_prob(frame['action'])
            if not torch.allclose(logp.detach(),frame['logp'],atol=2e-2 if amp else 5e-5,rtol=1e-4):
                raise RuntimeError('On-policy replay mismatch')
            loss=-(advantage*logp*frame['live']).sum()/(batch*rollouts)
        scaler.scale(loss).backward()
        loss_total+=float(loss.detach())
        if progress is not None and step%20==0: progress(step,True)
    used=[(e,leaf.grad) for e,leaf in zip(original,leaves) if leaf.grad is not None]
    torch.autograd.backward([e for e,g in used],[g for e,g in used])
    scaler.unscale_(opt)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    skipped=not bool(torch.isfinite(norm))
    if skipped and not amp: raise RuntimeError('Nonfinite gradient without AMP')
    scaler.step(opt); scaler.update()
    summary={k:float(v.double().mean()) for k,v in metrics.items()}
    summary.update(best_rollout_reward=float(reward.max(1).values.mean()),loss=loss_total,
                   gradient_norm=None if skipped else float(norm),optimizer_step_skipped=skipped,
                   amp_scale=scaler.get_scale(),seconds=time.monotonic()-start,
                   peak_gpu_mb=torch.cuda.max_memory_allocated()/1024**2)
    detail=dict(rollout_metrics={k:v[:100].cpu().tolist() for k,v in metrics.items()},
                decision_history=torch.stack(history).cpu().tolist(),
                history_columns=['time','vehicle','action_slot','live'],
                action_slots='0..N-1: NEW(customer=slot+1); N..2N: existing destination=slot-N; 2N+1: select existing',
                vehicle_histogram=torch.bincount(env.k,minlength=env.v+1).cpu().tolist())
    return summary,detail


def run(args):
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required; no CPU fallback')
    torch.set_num_threads(1)
    if args.out.exists() and not args.resume: raise ValueError('Use a new output directory')
    args.out.mkdir(parents=True,exist_ok=True)
    configs={str(w):json.loads((ROOT/f'result/rc101_learned_first_train20_20261017/late{w}/config.json').read_text()) for w in args.weights}
    split=json.loads((SOURCE/'split.json').read_text()); rows={r['name']:r for r in split['train']}
    files=list(Path(__file__).parent.glob('*.py'))+[ROOT/'train_vrptw.py',ROOT/'vrptw_env.py',ROOT/'event_observation.py']
    protocol=dict(version='concurrent_fleet_v4_tensor_legacy_model',device='cuda:0',gpu=torch.cuda.get_device_name(0),
                  configs=configs,weights=args.weights,smoke_updates=args.smoke_updates,
                  model='Unchanged VRPTWModel encoder/decoder parameter schema, shared attention decoder for fleet action mapping; no new MLP',
                  learning='Legacy Adam, AMP/GradScaler, per-episode mean/std baseline, clip=1, batch128 x rollout100',
                  changes='Concurrent global clock and t0 dispatch action mapping; CUDA tensor environment. One-shot test.',
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
        random.seed(tp['seed']); torch.manual_seed(tp['seed'])
        model=FleetPolicy(**config['model']).cuda()
        opt=torch.optim.Adam(model.parameters(),**config['optimizer']['optimizer'])
        amp=tp.get('use_amp',True); scaler=torch.amp.GradScaler('cuda',enabled=amp)
        state=dict(epoch=1,episodes=0,update=0,trajectories=0)
        if args.resume and (folder/'checkpoint-last.pt').exists():
            saved=torch.load(folder/'checkpoint-last.pt',weights_only=False)
            model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer']); scaler.load_state_dict(saved['scaler'])
            state=saved['state']; random.setstate(saved['python_rng']); torch.set_rng_state(saved['cpu_rng']); torch.cuda.set_rng_state(saved['cuda_rng'])
        cache={}
        def data(row):
            if row['name'] not in cache: cache[row['name']]=load(row)
            return cache[row['name']]
        def checkpoint():
            temp=folder/'checkpoint.tmp'
            torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),state=state,
                            python_rng=random.getstate(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),temp)
            temp.replace(folder/'checkpoint-last.pt')
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
        if state['update']==0: checkpoint(); probe()
        while state['epoch']<=tp['epochs']:
            if (args.out/'STOP').exists(): checkpoint(); status('paused'); return
            name=random.choice(tp['train_instances']); inst,public=data(rows[name])
            batch=min(tp['train_batch_size'],tp['train_episodes']-state['episodes'])
            def progress(step,backward=False):
                status('backward' if backward else 'sampling',decision_step=step,batch_size=batch,name=name)
            summary,detail=train_batch(model,opt,scaler,inst,public,ep,batch,ep['pomo_size'],amp,progress)
            state['update']+=1; state['episodes']+=batch; state['trajectories']+=batch*ep['pomo_size']
            row=dict(**state,name=name,batch_size=batch,**summary)
            append(folder/'training.jsonl',row)
            if state['update']<=20 or state['update']%20==0:
                write(folder/'early_history'/f'update_{state["update"]:06d}.json',dict(**row,**detail)); probe()
            if state['episodes']==tp['train_episodes']:
                state['epoch']+=1; state['episodes']=0
            checkpoint(); status('training',last_batch_seconds=summary['seconds'])
            if state['episodes']==0 and state['epoch']-1 in (500,1000):
                shutil.copy2(folder/'checkpoint-last.pt',folder/f'checkpoint-{state["epoch"]-1}.pt')
            print(json.dumps(dict(weight=weight,**row)),flush=True)
            if args.smoke_updates and state['update']>=args.smoke_updates:
                status('smoke_complete',full_training=False); return
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
    parser.add_argument('--weights',type=int,nargs='+',default=[5,10])
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--smoke-updates',type=int,default=0)
    args=parser.parse_args()
    try: run(args)
    except Exception:
        if args.out.exists(): write(args.out/'failure.json',dict(error=traceback.format_exc()))
        raise
