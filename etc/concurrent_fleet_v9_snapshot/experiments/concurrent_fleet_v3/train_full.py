"""Full legacy episode/rollout budget with memory-bounded gradient accumulation."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

import torch

from experiments.concurrent_fleet_v3.run import ROOT, SOURCE, load, write, diagnostics
from experiments.concurrent_fleet_v3.env import FleetEnv
from experiments.concurrent_fleet_v3.policy import FleetPolicy, rollout
from experiments.concurrent_fleet_v3.gpu_batch import sample, backward


def append(path, row):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row) + '\n')


def detail(env, metrics, config):
    result = diagnostics(env, metrics)
    p = config['environment']
    result.update(first_positive_time=next((r['time'] for r in env.trace if r['time'] > 0), None),
                  cost_vehicle=p['vehicle_penalty'] * metrics['vehicles']/env.n,
                  cost_travel=p['time_penalty']*metrics['travel_time']/env.horizon/p['TT_max'],
                  cost_late=p['late_penalty']*metrics['total_late']/env.horizon/p['Lt_max'],
                  cost_unserved=p['unserved_penalty']*metrics['unserved']/env.n)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--weights', type=int, nargs='+', default=[5, 10])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; CPU fallback is disabled')
    device = torch.device('cuda:0')
    if args.out.exists() and not args.resume:
        parser.error('Existing run: use --resume, never overwrite')
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    configs = {str(w): json.loads((ROOT/f'result/rc101_learned_first_train20_20261017/late{w}/config.json').read_text()) for w in args.weights}
    split = json.loads((SOURCE/'split.json').read_text())
    rows = {r['name']: r for r in split['train']}
    files = list(Path(__file__).parent.glob('*.py')) + [ROOT/'train_vrptw.py', ROOT/'vrptw_env.py', ROOT/'event_observation.py', ROOT/'advantage.py']
    hashes = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    protocol = dict(version='concurrent_fleet_v3_full_gpu', weights=args.weights, configs=configs,
                    device=str(device), gpu=torch.cuda.get_device_name(device), source_hashes=hashes, seed=42,
                    budget='1000 epochs, 2500 episodes/epoch, batch 128 (last 68), 100 rollouts/episode',
                    optimizer='One step per legacy batch; accumulate episode gradients; independent mean/std baseline per 100 rollouts',
                    evaluation='One greedy rollout per test instance, all 80, after training each lambda',
                    limitations='New decoder and concurrent environment; not identical architecture. Batched CUDA policy and backward; CPU event simulation.')
    if args.resume:
        if json.loads((args.out/'protocol.json').read_text()) != protocol:
            raise ValueError('Source/config changed; refuse incompatible resume')
    else:
        write(args.out/'protocol.json', protocol)
        for f in files:
            target = args.out/'source'/f.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
        shutil.copy2(SOURCE/'split.json', args.out/'split.json')
    for weight in args.weights:
        folder = args.out/f'late{weight}'
        folder.mkdir(exist_ok=True)
        if (folder/'complete.json').exists():
            continue
        config = configs[str(weight)]
        tp = config['trainer']
        count = config['environment']['pomo_size']
        random.seed(tp['seed']); torch.manual_seed(tp['seed'])
        model = FleetPolicy(config['model']).to(device)
        opt = torch.optim.Adam(model.parameters(), **config['optimizer']['optimizer'])
        train = [rows[n] for n in tp['train_instances']]
        state = dict(epoch=1, episode=0, group=0, update=0, trajectories=0, name=None)
        opt.zero_grad()
        if args.resume and (folder/'checkpoint-last.pt').exists():
            saved = torch.load(folder/'checkpoint-last.pt', weights_only=False)
            model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer'])
            state = saved['state']
            random.setstate(saved['python_rng']); torch.set_rng_state(saved['torch_rng'])
            torch.cuda.set_rng_state(saved['cuda_rng'],device)
            for p, grad in zip(model.parameters(), saved['grads']):
                p.grad = grad

        loaded = {}

        def env_for(row):
            if row['name'] not in loaded:
                loaded[row['name']] = load(row)
            inst, public = loaded[row['name']]
            return inst, FleetEnv(inst, public, weight, environment_params=config['environment'])

        def checkpoint():
            temp = folder/'checkpoint.tmp'
            torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), state=state,
                            grads=[p.grad for p in model.parameters()],
                            python_rng=random.getstate(), torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state(device)), temp)
            temp.replace(folder/'checkpoint-last.pt')

        def status(stage, **extra):
            write(args.out/'status.json', dict(stage=stage, weight=weight, **state,
                  target_epochs=tp['epochs'], episodes_per_epoch=tp['train_episodes'],
                  rollouts=count, device=str(device), gpu=torch.cuda.get_device_name(device),
                  gpu_allocated_mb=torch.cuda.memory_allocated(device)/1024**2,
                  pending_weights=[w for w in args.weights if not (args.out/f'late{w}'/'complete.json').exists()], **extra))

        def probe():
            model.eval(); results=[]
            with torch.no_grad():
                for row in (train[0], train[10]):
                    inst, env = env_for(row)
                    metrics, _ = rollout(model, env, model.encode(inst), greedy=True)
                    results.append(dict(name=row['name'], **detail(env, metrics, config), trace=env.trace))
            write(folder/'checks'/f'update_{state["update"]:06d}.json',
                  dict(**state, role='fixed TRAIN probes, not test selection', results=results))
            model.train()

        if state['update'] == 0 and state['group'] == 0:
            probe(); checkpoint()
        model.train()
        while state['epoch'] <= tp['epochs']:
            if (args.out/'STOP').exists():
                checkpoint(); status('paused'); return
            batch = min(tp['train_batch_size'], tp['train_episodes']-state['episode'])
            if state['name'] is None:
                state['name'] = random.choice(train)['name']
            row = rows[state['name']]
            inst, _ = env_for(row)
            start = time.monotonic()
            envs=[env_for(row)[1] for _ in range(count)]
            metrics, records=sample(model, inst, envs,
                progress=lambda step, active: status('sampling',batch_size=batch,decision_step=step,active_rollouts=active))
            samples=[detail(env,m,config) for env,m in zip(envs,metrics)]
            traces=[env.trace for env in envs]
            rewards=torch.tensor([s['reward'] for s in samples])
            adv=rewards-rewards.mean()
            adv=adv/adv.std().clamp(min=1e-6)
            status('backward',batch_size=batch,rollout_completed=count)
            loss_value=backward(model,inst,records,adv,batch)
            record=dict(**state, batch_size=batch, seconds=time.monotonic()-start,
                        loss_contribution=loss_value, best_reward=float(rewards.max()),
                        **{'mean_'+k:sum(s[k] for s in samples)/count for k in samples[0]
                           if all(isinstance(s[k], (int, float)) for s in samples)})
            append(folder/'episodes.jsonl', record)
            if state['update']<20 or state['group']==0:
                write(folder/'early_history'/f'update_{state["update"]:06d}_episode_{state["group"]:03d}.json',
                      dict(**record, samples=samples, first_rollout_trace=traces[0]))
            state['group']+=1; state['trajectories']+=count
            if state['group']==batch:
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(norm):
                    raise RuntimeError('Nonfinite gradient')
                opt.step(); opt.zero_grad()
                state['update']+=1; state['episode']+=batch; state['group']=0; state['name']=None
                append(folder/'updates.jsonl', dict(**state, gradient_norm=float(norm)))
                if state['update']<=20 or state['update']%20==0:
                    probe()
                if state['episode']==tp['train_episodes']:
                    state['epoch']+=1; state['episode']=0
            checkpoint(); status('training', last_episode_seconds=record['seconds'])
            print(json.dumps(dict(weight=weight, **state, mean_reward=record['mean_reward'],
                                  mean_vehicles=record['mean_vehicles'], mean_unserved=record['mean_unserved'],
                                  seconds=record['seconds'])), flush=True)
        status('evaluation')
        model.eval(); results=[]
        with torch.no_grad():
            for row in split['test']:
                inst, env=env_for(row)
                metrics, _=rollout(model, env, model.encode(inst), greedy=True)
                results.append(dict(name=row['name'], **metrics))
                write(folder/'results.json', results)
                write(folder/'test_traces'/(row['name']+'.json'), env.trace)
        write(folder/'complete.json', dict(epoch=tp['epochs'], test_count=len(results), trajectories=state['trajectories']))
    write(args.out/'status.json', dict(stage='complete', weights=args.weights, full_training=True))


if __name__ == '__main__':
    main()
