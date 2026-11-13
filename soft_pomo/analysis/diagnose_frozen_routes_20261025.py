"""Frozen train-only sampling; no optimization, LLM or best-of-N test claims."""
import json
import sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from experiments.concurrent_fleet_v3.run import SOURCE, load, write
from experiments.concurrent_fleet_v9.env import TensorFleetEnv
from experiments.concurrent_fleet_stop_init.policy import FleetPolicy

BASE = ROOT/'server_export/be_completed_received_20261022_091410/rc101_v9_bias_entropy_vehicle300_late10_20261021/runs/vehicle300'
OUT = ROOT/'analysis/frozen_start_route_split_20261025'


@torch.no_grad()
def rollout(model, inst, public, params, count, greedy=False, prefix=None):
    env = TensorFleetEnv(inst, count, params, device='cuda')
    starts, encoded = model.encode_events(inst, public)
    actions, dispatch, vehicles = [], [], []
    concentration, entropies = [], []
    new_concentration, new_entropies = [], []
    for step in range(600):
        if bool(env.done.all()):
            break
        frame = env.frame()
        p = model.probabilities(frame, starts, encoded, env)
        new_valid = ~env.done & env.dispatch_open & (frame['mask'][:, :env.n].sum(1)>1)
        if new_valid.any():
            q = p[new_valid, :env.n].double()
            q = q / q.sum(1, keepdim=True).clamp_min(1e-300)
            new_concentration.append(q.max(1).values)
            new_entropies.append(-(q*q.clamp_min(1e-300).log()).sum(1))
        valid = ~env.done & ~env.dispatch_open & (frame['mask'].sum(1)>1)
        if valid.any():
            q = p[valid].double()
            concentration.append(q.max(1).values)
            entropies.append(-(q*q.clamp_min(1e-300).log()).sum(1))
        if prefix is not None and step < len(prefix):
            assert bool(env.dispatch_open.all())
            action = torch.full((count,), prefix[step], device='cuda', dtype=torch.long)
        else:
            if prefix is not None:
                assert not bool(env.dispatch_open.any())
            action = p.argmax(1) if greedy else p.multinomial(1).squeeze(1)
        if greedy and bool(env.dispatch_open[0]):
            dispatch.append(int(action[0]))
        actions.append(torch.where(env.done, -1, action))
        vehicles.append(torch.where(action < env.n, env.k, env.next_vehicle()[0]))
        env.step(action, validate=True)
    else:
        raise RuntimeError('Rollout exceeded bound')
    histories = torch.stack(actions, 1).cpu().tolist()
    vehicle_ids = torch.stack(vehicles, 1).cpu().tolist()
    canonical = []
    for history, ids in zip(histories, vehicle_ids):
        routes = {}
        for a, vehicle in zip(history, ids):
            if a < 0 or a == 2*env.n+1:
                continue
            destination = a+1 if a < env.n else a-env.n
            routes.setdefault(vehicle, []).append(destination)
        canonical.append(tuple(sorted(tuple(route) for route in routes.values())))
    histories = canonical
    return ({k:v.cpu().double() for k,v in env.metrics().items()}, histories, dispatch,
            dict(route_top1=float(torch.cat(concentration).mean()),
                 route_entropy=float(torch.cat(entropies).mean()),
                 new_given_new_top1=float(torch.cat(new_concentration).mean()),
                 new_given_new_entropy=float(torch.cat(new_entropies).mean())))


def summarize(metrics, histories, mask, greedy):
    indices = mask.nonzero().flatten().tolist()
    if not indices:
        return dict(count=0)
    reward = metrics['reward'][mask]
    threshold = 1e-4
    best = indices[int(reward.argmax())]
    route_counts = Counter(histories[i] for i in indices)
    start_counts = Counter(tuple(sorted(route[0] for route in histories[i])) for i in indices)
    return dict(count=len(indices), unique_route_sets=len({histories[i] for i in indices}),
        unique_first_customer_sets=len(start_counts),
        modal_first_customer_set_fraction=max(start_counts.values())/len(indices),
        first_customer_sets=[dict(customers=list(k),count=v) for k,v in start_counts.most_common()],
        modal_route_set_fraction=max(route_counts.values())/len(indices),
        better_fraction=float((reward>greedy['reward']+threshold).double().mean()),
        reward_mean=float(reward.mean()), reward_std=float(reward.std(unbiased=False)),
        reward_best=float(reward.max()), reward_top10_mean=float(reward.topk(max(1,len(indices)//10)).values.mean()),
        best_gain=float(reward.max()-greedy['reward']),
        means={k:float(metrics[k][mask].mean()) for k in ['vehicles','travel_time','total_late','unserved']},
        best_metrics={k:float(metrics[k][best]) for k in ['reward','vehicles','travel_time','total_late']})


def main():
    torch.set_num_threads(1)
    cfg=json.loads((BASE/'protocol.json').read_text())['configs']['10']
    model=FleetPolicy(**cfg['model']).cuda().eval()
    checkpoint=torch.load(BASE/'late10/checkpoint-500.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    report=dict(checkpoint=str(BASE/'late10/checkpoint-500.pt'), epoch=500,
        scope='train20, CUDA FP32, 1000 samples per mode, no gradients or LLM',
        improvement_tolerance=1e-4, missing_checkpoint_1000=True, results=[])
    rows=json.loads((SOURCE/'split.json').read_text())['train']
    for index,row in enumerate(rows):
        inst,public=load(row)
        gm,_,prefix,_=rollout(model,inst,public,cfg['environment'],1,greedy=True)
        greedy={k:float(gm[k][0]) for k in ['reward','vehicles','travel_time','total_late']}
        record=dict(name=row['name'],greedy=greedy,dispatch_prefix=prefix)
        for mode in ['full','fixed_dispatch']:
            chunks=[];histories=[];stats=[]
            for chunk in range(4):
                torch.manual_seed(20261025+index*10+chunk)
                m,h,_,s=rollout(model,inst,public,cfg['environment'],250,
                              prefix=prefix if mode=='fixed_dispatch' else None)
                chunks.append(m);histories.extend(h);stats.append(s)
            metrics={k:torch.cat([m[k] for m in chunks]) for k in chunks[0]}
            same=metrics['vehicles']==greedy['vehicles']
            if mode=='fixed_dispatch':
                assert bool(same.all())
            record[mode]=dict(all_samples=summarize(metrics,histories,torch.ones(1000,dtype=torch.bool),greedy),
                same_k=summarize(metrics,histories,same,greedy),
                **{k:sum(s[k] for s in stats)/len(stats) for k in stats[0]})
        report['results'].append(record)
        write(OUT/'report.json',report)
        print(json.dumps(record),flush=True)
    write(OUT/'complete.json',dict(instances=20,sampled_rollouts=40000))


if __name__=='__main__':
    main()
