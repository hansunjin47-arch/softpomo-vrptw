"""Full train20 evaluation, causal saved-call replay and no-guide ablation. No API."""
import collections
import json
from pathlib import Path
from statistics import mean

import torch
from experiments.concurrent_fleet_pair.env import TensorFleetEnv
from experiments.concurrent_fleet_pair.policy import FleetPolicy
from experiments.concurrent_fleet_pair_kim.run import GuidedEnv, GuidedPolicy, OnlineEnv
from experiments.concurrent_fleet_bias_guided.support import load, read
from experiments.concurrent_fleet_v9_guided.prompts import parse_test_response
from experiments.concurrent_fleet_pair.io import write

ROOT = Path(__file__).resolve().parents[1]
PURE = ROOT/'result/pair300_epsilon005_floor001_std_20261026'
PHASE = ROOT/'result/pair300_phase2_feedback500_20261028'
KIM = ROOT/'result/pair300_epsilon005_floor001_std_kim_example_20261027'
OUT = ROOT/'result/pair_phase2_diagnosis_20261029'
KEYS = ['reward', 'vehicles', 'travel_time', 'total_late', 'unserved']


def average(rows):
    return {k:mean(r['metrics'][k] for r in rows) for k in KEYS}


@torch.no_grad()
def rollout(model, inst, reports, cfg, groups=None, cache=None, calls=None, beta=5., detail=False):
    counts = collections.Counter()
    call_index = 1
    if groups is None:
        env = TensorFleetEnv(inst, 1, cfg['environment'], 'cpu')
    else:
        initial = parse_test_response(calls[0]['response'] if calls else cache['initial'], groups, list(range(1,101)))
        if calls:
            def refresh(env, disclosure):
                nonlocal call_index
                call = calls[call_index]
                ctx = json.loads(call['request']['messages'][1]['content'])
                remaining = [n for n in range(1,101) if int(env.customer[0,n]) == 0]
                assert remaining == ctx['selectable_customers'], 'Saved-call replay diverged'
                assert abs(float(env.time[0])-call['clock_min']) < 1e-3
                scores = parse_test_response(call['response'], groups, remaining)
                env.update_event_confidence(scores, revealed_at_min=disclosure, effective_time_min=float(env.time[0]))
                call_index += 1
            env = OnlineEnv(inst, 1, cfg['environment'], 'cpu', groups=groups, scores=initial,
                            beta=beta, reports=reports, on_disclosure=refresh)
        else:
            env = GuidedEnv(inst, 1, cfg['environment'], 'cpu', groups=groups, scores=initial, beta=beta)
            if cache.get('post_disclosure') is not None:
                t=cache['revealed_at_min']
                env.update_event_confidence(parse_test_response(cache['post_disclosure'], groups, list(range(1,101))),
                                            revealed_at_min=t,effective_time_min=t)
    starts, encoded = model.encode_events(inst, reports)
    anchors=[];trace=[];margins=[]
    for step in range(600):
        if bool(env.done.all()):break
        frame=env.frame();p=model.probabilities(frame,starts,encoded,env);raw=p;a=p.argmax(1)
        new,v,d=env.decode_action(a)
        if bool(new[0]):anchors.append(int(d[0]))
        if groups is not None and not bool(env.dispatch_open[0]):
            counts['routing_steps']+=1
            env.guidance.beta=0
            raw=model.probabilities(frame,starts,encoded,env)
            env.guidance.beta=beta
            counts['bias_argmax_changes']+=int(raw.argmax(1)[0]!=a[0])
            top=raw[0].topk(2).values
            if top[1]>0:margins.append(float((top[0]/top[1]).log()))
            if env.guidance.updates:
                updates=env.guidance.updates;env.guidance.updates=[]
                prior=model.probabilities(frame,starts,encoded,env)
                env.guidance.updates=updates
                counts['post_steps']+=1
                counts['refresh_argmax_changes']+=int(prior.argmax(1)[0]!=a[0])
        if detail and int(d[0])==0:
            free=(env.customer[0,1:]==0).nonzero().flatten()+1
            viable=[int(n) for n in free if env.demand[n] <= env.load[0,v[0]]+1e-5]
            item=dict(step=step,time=float(env.time[0]),vehicle=int(v[0]),destination=0,
                      remaining=free.tolist(),eligible_by_capacity=viable,
                      remaining_load=float(env.load[0,v[0]]),selected_probability=float(p[0,a[0]]),
                      vehicle_status=env.status[0].tolist(),vehicle_loads=env.load[0].tolist())
            if groups is not None:
                item['raw_argmax_action']=int(raw.argmax(1)[0])
                group=int(env.guidance.vehicle_group[0,v[0]])
                item['anchor_group']=group
                item['remaining_in_anchor_group']=[n for n in free.tolist() if group>=0 and n in groups[group]]
            trace.append(item)
        env.step(a,validate=True)
    assert bool(env.done.all())
    if calls:assert call_index==len(calls)
    return dict(name=inst['name'],metrics={k:float(v[0]) for k,v in env.metrics().items()},
                starts=sorted(anchors),counters=dict(counts),depot_decisions=trace,
                final_remaining=(env.customer[0,1:]==0).nonzero().flatten().add(1).tolist(),
                mean_raw_margin=mean(margins) if margins else None)


def main():
    torch.set_num_threads(2)
    cfg=read(PURE/'protocol.json')['configs']['10']
    split=read(PHASE/'artifacts/split.json')
    models={}
    for label,folder,guided in [('pure',PURE,False),('phase2',PHASE/'training',True),('kim',KIM/'training',True)]:
        cls=GuidedPolicy if guided else FleetPolicy
        model=cls(**cfg['model']).cpu().eval()
        model.load_state_dict(torch.load(folder/'late10/checkpoint-1000.pt',map_location='cpu',weights_only=False)['model_state_dict'])
        models[label]=model
    groups=read(PHASE/'artifacts/groups.json')
    records={}
    for label in ['pure','phase2','kim']:
        rows=[]
        for row in split['train']:
            inst,reports=load(row)
            source=PHASE if label=='phase2' else KIM
            cache=read(source/'artifacts/train_cache'/(row['name']+'.json')) if label!='pure' else None
            result=rollout(models[label],inst,reports,cfg,groups if cache else None,cache)
            rows.append(result)
        records[label+'_train']=rows
        write(OUT/'report.json',records)
        print(label,'TRAIN',average(rows),flush=True)
    rows=[];off=[]
    original={r['name']:r for r in read(PHASE/'training/late10/results.json')}
    for row in split['test']:
        inst,reports=load(row)
        calls=read(PHASE/'late10/online_test'/row['name']/'calls.json')
        result=rollout(models['phase2'],inst,reports,cfg,groups,calls=calls,detail='_P027' in row['name'])
        assert abs(result['metrics']['reward']-original[row['name']]['reward'])<1e-4, row['name']
        rows.append(result)
        # No guide means no LLM state reuse, even when the trajectory changes.
        plain=FleetPolicy(**cfg['model']).cpu().eval()
        plain.load_state_dict(models['phase2'].state_dict())
        off.append(rollout(plain,inst,reports,cfg,detail='_P027' in row['name']))
        records['phase2_test']=rows;records['phase2_test_no_guide']=off
        write(OUT/'report.json',records)
        print('TEST',len(rows),row['name'],result['metrics']['reward'],off[-1]['metrics']['reward'],flush=True)
    summary={k:average(v) for k,v in records.items()}
    for k in ['phase2_test','phase2_train','kim_train']:
        c=collections.Counter()
        for r in records[k]:c.update(r['counters'])
        summary[k]['counters']=dict(c)
    summary['no_guide_changed_cases']=sum(abs(a['metrics']['reward']-b['metrics']['reward'])>1e-4 for a,b in zip(rows,off))
    write(OUT/'summary.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
