"""KIM fresh1000 or phase2 pure500+500, with identical bias/entropy trainer."""
import argparse
import hashlib
import json
import random
import shutil
import traceback
import torch
from experiments.concurrent_fleet_bias_guided.support import (
    ROOT, PURE, OUT, read, load, GuidedBiasPolicy, cached_env, GUIDANCE_BETA)
from experiments.concurrent_fleet_v3.run import SOURCE
from experiments.concurrent_fleet_pair.io import write, replace
from experiments.concurrent_fleet_stop_init.train import train_batch, collect, append
from experiments.concurrent_fleet_stop_init.policy import calibrate
from experiments.concurrent_fleet_exploration.objective import entropy_beta
from experiments.concurrent_fleet_v9_guided.online import OpenRouterClient, evaluate_instance
from train_vrptw import _set_seed, plot_training_curves


def annealed_guidance_beta(epoch):
    return GUIDANCE_BETA * max(0., min(1., (1000 - epoch) / 500))


def run(method, resume=False, checkpoint_epochs=None, anneal_from=None,
        fresh_feedback=False, matched_feedback=False):
    if anneal_from is not None and method != 'kim':
        raise ValueError('Annealing experiment requires a KIM checkpoint')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(1)
    artifact=OUT/method/'artifacts';out=OUT/method/'training';folder=out/'late10'
    if out.exists() and not resume:raise ValueError('Existing run; use --resume')
    folder.mkdir(parents=True,exist_ok=True)
    cfg=read(PURE/'protocol.json')['configs']['10'];tp=cfg['trainer'];ep=cfg['environment']
    if matched_feedback:
        tp['epochs'] = 500
    split=read(SOURCE/'split.json');rows={r['name']:r for r in split['train']}
    assert set(p.stem for p in (artifact/'train_cache').glob('*.json'))==set(rows)
    BaseEnv=cached_env(artifact)
    guidance_value=GUIDANCE_BETA
    class Env(BaseEnv):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.guidance.beta=guidance_value
    files=sorted((ROOT/'experiments').rglob('*.py'))+[ROOT/name for name in
        ['train_vrptw.py','vrptw_env.py','event_observation.py']]
    protocol=dict(method=method,configs=cfg,pure=str(PURE),guidance_beta=GUIDANCE_BETA,
        stop_target=.15,advantage='shifted mean, no std',entropy=dict(beta=1.,zero_epoch=500),
        source_hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        cache_hashes={p.relative_to(artifact).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(artifact.rglob('*.json')) if 'pure500_test80' not in p.parts},
        initial_checkpoint_sha256=hashlib.sha256((PURE/'late10/checkpoint-500.pt').read_bytes()).hexdigest() if method=='phase2' else None,
        test='one greedy rollout, exact request cache only, online accident disclosure')
    from experiments.concurrent_fleet_v9_guided.env import GuidedFleetEnv
    protocol.update(reference_start_gate=GuidedFleetEnv.require_reference_start,
                    matched_feedback=matched_feedback, fresh_feedback=fresh_feedback,
                    optimizer_reset=matched_feedback)
    if fresh_feedback:
        protocol['initial_checkpoint_sha256'] = None
    save_epochs = checkpoint_epochs or [1,2,5,10,20,50,100,200,500,1000]
    if checkpoint_epochs is not None:
        protocol['checkpoint_epochs'] = list(checkpoint_epochs)
    if anneal_from is not None:
        protocol.update(initial_checkpoint_sha256=hashlib.sha256(anneal_from.read_bytes()).hexdigest(),
                        initial_checkpoint=str(anneal_from),
                        guidance_schedule=dict(start_epoch=500,end_epoch=1000,start_beta=5.,end_beta=0.),
                        test='one greedy rollout, beta=0, no LLM client or calls')
    if resume:
        assert read(out/'protocol.json')==protocol,'Resume config/source/cache mismatch'
    else:
        write(out/'protocol.json',protocol)
        for p in files:
            dest=out/'source'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    _set_seed(tp['seed'])
    model=GuidedBiasPolicy(**cfg['model'])
    if (method=='kim' or fresh_feedback) and not resume and anneal_from is None:
        write(folder/'stop_calibration.json',calibrate(model,[load(r) for r in split['train']],ep,.15))
    model=model.cuda()
    opt=torch.optim.Adam(model.parameters(),**cfg['optimizer']['optimizer'])
    amp=tp.get('use_amp',True);scaler=torch.amp.GradScaler('cuda',enabled=amp)
    state=dict(epoch=1,episodes=0,update=0,trajectories=0,epoch_reward_sum=0.,epoch_loss_sum=0.)
    rewards=[];losses=[]
    if resume or (method=='phase2' and not fresh_feedback) or anneal_from is not None:
        path=folder/'checkpoint-last.pt' if resume else (anneal_from or PURE/'late10/checkpoint-500.pt')
        saved=torch.load(path,weights_only=False)
        if not resume:assert saved['epoch']==500 and saved['state']['epoch']==501 and saved['state']['episodes']==0
        model.load_state_dict(saved['model_state_dict'],strict=True)
        if resume or not matched_feedback:
            opt.load_state_dict(saved['optimizer_state_dict']);scaler.load_state_dict(saved['scaler'])
            rewards=saved['reward_log'];losses=saved['loss_log'];state=saved['state']
            random.setstate(saved['python_rng']);torch.set_rng_state(saved['cpu_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
    if matched_feedback and not resume:
        _set_seed(tp['seed'])
    def checkpoint():
        temp=folder/'checkpoint.tmp'
        torch.save(dict(epoch=len(rewards),model_state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),
            scaler=scaler.state_dict(),state=state,reward_log=rewards,loss_log=losses,
            python_rng=random.getstate(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),temp)
        replace(temp,folder/'checkpoint-last.pt')
    def status(stage,**extra):write(out/'status.json',dict(stage=stage,method=method,**state,**extra))
    cache={}
    def data(row):
        if row['name'] not in cache:cache[row['name']]=load(row)
        return cache[row['name']]
    def probe():
        model.eval();results=[]
        for name in [tp['train_instances'][0],tp['train_instances'][10]]:
            inst,public=data(rows[name]);env,_,history=collect(model,inst,public,ep,1,amp,greedy=True,env_type=Env)
            results.append(dict(name=name,**{k:float(v[0]) for k,v in env.metrics().items()},history=torch.stack(history).cpu().tolist()))
        write(folder/'checks'/f'update_{state["update"]:06d}.json',dict(role='TRAIN probes',**state,results=results))
        model.train()
    if not resume:
        checkpoint();shutil.copy2(folder/'checkpoint-last.pt',folder/f'checkpoint-{len(rewards)}.pt');probe()
    model.train()
    while state['epoch']<=tp['epochs']:
        if (out/'STOP').exists():checkpoint();status('paused');return
        if anneal_from is not None:
            guidance_value=annealed_guidance_beta(state['epoch'])
        inst,public=data(rows[random.choice(tp['train_instances'])])
        batch=min(tp['train_batch_size'],tp['train_episodes']-state['episodes'])
        def progress(step,backward=False):status('backward' if backward else 'sampling',decision_step=step)
        beta=entropy_beta(state['epoch'],1.,500)
        summary,detail=train_batch(model,opt,scaler,inst,public,ep,batch,ep['pomo_size'],amp,progress,
            tp.get('max_steps',600),beta=beta,env_type=Env)
        summary['stop_bias']=float(model.stop_bias.detach())
        if anneal_from is not None:
            summary['guidance_beta']=guidance_value
        state['update']+=1;state['episodes']+=batch;state['trajectories']+=batch*ep['pomo_size']
        state['epoch_reward_sum']+=summary['best_rollout_reward']*batch;state['epoch_loss_sum']+=summary['loss']*batch
        row=dict(**state,name=inst['name'],batch_size=batch,**summary)
        append(folder/'training.jsonl',row)
        if state['update']<=20 or state['update']%20==0:
            write(folder/'early_history'/f'update_{state["update"]:06d}.json',dict(**row,**detail));probe()
        if state['episodes']==tp['train_episodes']:
            rewards.append(state['epoch_reward_sum']/state['episodes']);losses.append(state['epoch_loss_sum']/state['episodes'])
            append(folder/'epochs.jsonl',dict(epoch=state['epoch'],train_reward=rewards[-1],loss=losses[-1]))
            state.update(epoch=state['epoch']+1,episodes=0,epoch_reward_sum=0.,epoch_loss_sum=0.)
        if state['update']%20==0:checkpoint()
        if state['episodes']==0 and state['epoch']-1 in save_epochs:
            checkpoint();shutil.copy2(folder/'checkpoint-last.pt',folder/f'checkpoint-{state["epoch"]-1}.pt')
        status('training');print(json.dumps(row),flush=True)
    (folder/'plots').mkdir(exist_ok=True)
    plot_training_curves(rewards,losses,str(folder/'plots'))
    checkpoint();status('evaluation');model.eval()
    if anneal_from is not None:
        assert annealed_guidance_beta(tp['epochs']) == 0
        results=[]
        for row in split['test']:
            if (out/'STOP').exists():status('paused');return
            inst,public=data(row)
            env,_,history=collect(model,inst,public,ep,1,amp,greedy=True)
            result=dict(name=row['name'],**{k:float(v[0]) for k,v in env.metrics().items()})
            results.append(result)
            write(folder/'test_history'/(row['name']+'.json'),dict(metrics=result,
                history=torch.stack(history).cpu().tolist(),guidance_beta=0.,paid_calls=0))
            write(folder/'results.json',results);status('evaluation',test_completed=len(results))
        write(folder/'complete.json',dict(epoch=1000,test_count=80,trajectories=state['trajectories'],
                                         guidance_beta=0.,paid_calls=0))
        status('complete');return
    groups=read(artifact/'groups.json');template=read(artifact/'request_template.json')
    from experiments.concurrent_fleet_bias_guided.llm_retry import RetryingClient
    from experiments.concurrent_fleet_v9_guided.prompts import parse_test_response
    old=OpenRouterClient(ROOT/'result/v9kim_calls_20261020/test',allow_paid=False)
    call=RetryingClient(OUT/method/'calls/test',
        lambda request,payload: parse_test_response(payload,groups,
            json.loads(request['messages'][1]['content'])['selectable_customers']),
        fallback=old if method=='kim' else None)
    results=[]
    for row in split['test']:
        if (out/'STOP').exists():status('paused');return
        inst,public=data(row);target=folder/'online_test'/row['name']
        if (target/'result.json').exists():result=read(target/'result.json')
        else:result=evaluate_instance(model,inst,ep,groups,public,template=template,client=call,
            beta=GUIDANCE_BETA,group_source='fixed_instance' if method=='kim' else 'training_only',out=target,amp=amp)
        results.append(dict(name=row['name'],**result['metrics']));write(folder/'results.json',results)
        status('evaluation',test_completed=len(results))
    write(folder/'complete.json',dict(epoch=tp['epochs'],test_count=80,trajectories=state['trajectories']))
    status('complete')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('method',choices=['kim','phase2']);ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    try:run(args.method,args.resume)
    except Exception:
        write(OUT/args.method/'training/failure.json',dict(error=traceback.format_exc()));raise
