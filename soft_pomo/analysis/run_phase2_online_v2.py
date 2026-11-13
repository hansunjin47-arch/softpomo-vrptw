"""Versioned GPT Phase2: privileged train references, causal one-shot online evaluation."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

API_KEY = os.environ.get('OPENROUTER_API_KEY')

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p/'analysis/run_learned_first_train20.py').exists())
PURE = ROOT / 'result/rc101_learned_first_train20_20261017'
OUT = ROOT / 'result/rc101_phase2_online_gpt_train20_v2_2_20261018'
sys.path.insert(0, str(ROOT/'server'))
import train_cached as io

PUBLIC = {'event_type','affected_nodes','regions','start_min','end_min','revealed_at_min',
          'synthetic','rainfall_mm_per_hour','involved_vehicles','vehicles_involved','news_text'}


def immutable(path, value):
    if path.exists():
        assert io.read(path) == value, f'Immutable input changed: {path}'
    else:
        io.write(path, value)


def freeze():
    if (OUT/'version.json').exists():
        verify()
        return
    shutil.copytree(PURE/'source',OUT/'source')
    for filename in ('phase2_online_v2_runtime.py','run_phase2_online_v2.py'):
        shutil.copy2(ROOT/'analysis'/filename,OUT/'source'/filename)
    split=io.read(OUT/'source/split.json')
    for section,subfolder in (('train','data'),('test','eval_addon/data')):
        observations=ROOT/'server_export/soft_pomo_seed_sweep/train20'/subfolder
        for row in split[section]:
            assert io.sha(observations/(row['name']+'.txt'))==row['sha256']
            shutil.copy2(observations/(row['name']+'.observations.json'),
                         OUT/'source/data'/(row['name']+'.observations.json'))
    template_path = next((ROOT/'result/rc101_phase2_train20_bks_late10_gpt54_v1/requests').glob('*/request.json'))
    request = io.read(template_path)
    request['messages'] = []
    immutable(OUT/'request_template.json',request)
    kim = io.read(ROOT/'result/rc101_kim_train20_bks_customer_only_gpt54_v1/protocol.json')
    assert sorted(n for g in kim['clusters'] for n in g) == list(range(1,101))
    immutable(OUT/'test_groups.json',kim['clusters'])
    for weight in (1,5,10,20):
        immutable(OUT/f'late{weight}/config.json',io.read(PURE/f'late{weight}/config.json'))
        legacy=ROOT/f'result/rc101_phase2_train20_bks_late{weight}_gpt54_v1/requests'
        for row in split['train']:
            source=legacy/row['name']
            for filename in ('request.json','context.json'):
                immutable(OUT/f'legacy_train/late{weight}'/row['name']/filename,io.read(source/filename))
    immutable(OUT/'version.json',dict(version='phase2_online_v2.2',model='openai/gpt-5.4',
        training='new Pure500 branch; privileged train-only single-greedy references; cached guidance',
        train_prompt='Original per-lambda GPT prompt, ontology and schema unchanged; new reference groups only',
        evaluation='no test reference rollout; fixed Kim groups; initial public-only call; accident state call per vehicle on first disclosure',
        time_scope='sequential route construction with per-vehicle clocks; NOT a global chronological fleet simulator',
        disclosure='per-vehicle clock; accident guidance gated by disclosure, initial guidance before disclosure',
        decoder='beta5 active group; zero depot; no guidance for first customer of each route',
        files={p.relative_to(OUT).as_posix():io.sha(p) for p in OUT.rglob('*') if p.is_file()}))


def verify():
    for name, digest in io.read(OUT/'version.json')['files'].items():
        assert io.sha(OUT/name)==digest, name


def build(weight, oneshot=False, cpu=False):
    verify()
    sys.path.insert(0,str(OUT/'source'))
    config=copy.deepcopy(io.read(OUT/f'late{weight}/config.json'))
    if oneshot:
        config['environment']['pomo_size']=1
        config['trainer'].update(use_augmentation=False,n_mc_samples=1,pomo_auto=False,eval_top_k=None)
    trainer=io.build_trainer(config,OUT/'source',OUT/f'late{weight}/runtime',cpu=cpu)
    return trainer, config


def public_observation(name):
    # Only the explicitly public sidecar is read by the request builder.
    raw=io.read(OUT/'source/data'/(name+'.observations.json'))
    return {k:v for k,v in raw.items() if k in PUBLIC}


def fixed_info(inst, groups, reference=False):
    import numpy as np
    coords=np.concatenate([inst['depot_xy'].numpy(),inst['node_xy'].numpy()])*inst['max_coord']
    def values(field,scale):
        return (np.concatenate([inst['depot_'+field].numpy(),inst['node_'+field].numpy()])*scale).tolist()
    demand=values('demand',inst['vehicle_capacity'])
    opening,closing,service=[values(f,inst['T']) for f in ('tw_open','tw_close','service')]
    info=dict(capacity=inst['vehicle_capacity'],customer_columns=['id','x','y','demand','open_min','close_min','service_min'],
        customers=[[i,*map(float,xy),demand[i],opening[i],closing[i],service[i]] for i,xy in enumerate(coords)],
        groups=[dict(cluster=k+1,customers=list(g)) for k,g in enumerate(groups)])
    if reference:
        info['train_only_reference_routes']=groups
    return info


def prompt(weight, training):
    return f'''Assign static customer confidence scores in [0,1].
Minimize K/14+(T+{weight}*L)/1696.95. T is driving time, L includes customer and depot lateness.
All customers must be served; capacity is hard and time windows are soft.
Scores are preferences, not probabilities. Beta=5 multiplies scores as additive decoder bias.
The policy chooses the first customer of EACH vehicle with no LLM bias. That customer's
group becomes active until depot return. Only active-group customers receive bias.
Other groups and depot receive zero bias. The policy may cross groups or return to depot.
Groups are soft guidance memberships, not forced route assignments. Do not assign depot scores.
Accident scores are applied only once the current vehicle clock reaches public disclosure.
Each new vehicle clock starts at zero. Initial guidance applies before disclosure.
Online accident guidance is scoped to the supplied vehicle and cannot be reused for another vehicle.
The observed sequence contains only that vehicle's past. Other-route customer reservations are
allocation constraints, not evidence that another vehicle has already completed its journey.
{'Training feedback: reference routes come from train-only Pure500 rollouts and may use training outcomes. Use them as advice, not a verified optimum.' if training else 'Online inference: groups are fixed prior memberships, NOT future routes. Use ONLY supplied public events and observed state. Never infer a known future realization.'}
When observed state is provided, advise on remaining service; visited scores may be zero.
Output exactly 100 customer entries using the supplied confidence JSON schema.
No route edits, explanations, extra pairs, or customer zero.
'''


def make_request(folder, weight, inst, groups, events, training=False, state=None):
    if training:
        assert state is None and len(events)==1
        legacy=OUT/f'legacy_train/late{weight}'/inst['name']
        context=io.read(legacy/'context.json')
        request=io.read(legacy/'request.json')
        old_json=json.dumps(context,separators=(',',':'),allow_nan=False)
        assert request['messages'][0]['content'].endswith(old_json)
        prefix=request['messages'][0]['content'][:-len(old_json)]
        context['fixed_instance']['groups']=[dict(id=f'g{k+1}',type='ReferenceGroup',
            cluster=k+1,route=f'r{k+1}',reference_order=g,required_output_customers=g)
            for k,g in enumerate(groups)]
        context['public_event']=events[0]
        request['messages'][0]['content']=prefix+json.dumps(context,separators=(',',':'),allow_nan=False)
        immutable(folder/'context.json',context)
        immutable(folder/'groups.json',groups)
        immutable(folder/'request.json',request)
        return folder
    context=dict(fixed_instance=fixed_info(inst,groups,reference=training),public_events=events)
    if state is not None:
        context['observed_state']=state
    request=copy.deepcopy(io.read(OUT/'request_template.json'))
    request['model']='openai/gpt-5.4'
    request['messages']=[dict(role='user',content=prompt(weight,training)+'\n'+json.dumps(context,separators=(',',':'),allow_nan=False))]
    immutable(folder/'context.json',context)
    immutable(folder/'groups.json',groups)
    immutable(folder/'request.json',request)
    return folder


def call(folder):
    # Isolated process: trainer's offline network guard must never be disabled.
    import urllib.request
    response=folder/'response.txt'
    request=io.read(folder/'request.json')
    digest=io.sha(folder/'request.json')
    if (folder/'attempted.json').exists():
        assert io.read(folder/'attempted.json')['request_sha256']==digest
    if not response.exists():
        key=os.environ.get('OPENROUTER_API_KEY')
        if not key:
            raise RuntimeError('OPENROUTER_API_KEY is not set')
        if (folder/'attempted.json').exists():
            raise RuntimeError(f'Paid call previously attempted; inspect before retry: {folder}')
        immutable(folder/'attempted.json',dict(request_sha256=digest))
        req=urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',
            data=json.dumps(request).encode(),headers={'Content-Type':'application/json',
                                                       'Authorization':'Bearer '+key},method='POST')
        with urllib.request.urlopen(req,timeout=1200) as reply:
            body=json.loads(reply.read())
        io.write(folder/'api_response.json',body)
        assert body['choices'][0]['finish_reason']=='stop'
        response.write_text(body['choices'][0]['message']['content'],encoding='utf-8')
    wire=io.read(response)
    groups=io.read(folder/'groups.json')
    expected={(k+1,n) for k,g in enumerate(groups) for n in g}
    assert set(wire)=={'confidence'} and len(wire['confidence'])==100
    cache={str(k):{} for k in range(len(groups))}
    seen=set()
    for entry in wire['confidence']:
        assert set(entry)=={'cluster','customer','score'}
        k,n,score=entry['cluster'],entry['customer'],entry['score']
        assert type(k) is int and type(n) is int and (k,n) in expected and (k,n) not in seen
        assert type(score) in (int,float) and math.isfinite(score) and 0<=score<=1
        cache[str(k-1)][str(n)]=score
        seen.add((k,n))
    assert seen==expected
    cache['_depot']={str(k):0.0 for k in range(len(groups))}
    immutable(folder/'scores.json',cache)
    immutable(folder/'receipt.json',dict(request_sha256=digest,response_sha256=io.sha(response)))


def obtain(folder):
    if not API_KEY:
        raise RuntimeError('OpenRouter key was not available before offline trainer initialization')
    child_env = dict(os.environ, OPENROUTER_API_KEY=API_KEY)
    subprocess.run([sys.executable,'-u',str(OUT/'source/run_phase2_online_v2.py'),
        'call','--folder',str(folder),'--allow-api'],cwd=ROOT,check=True,env=child_env)
    return io.read(folder/'scores.json')


def prepare_train(weight):
    import torch
    trainer,_=build(weight,oneshot=True)
    import train_vrptw as pure
    checkpoint=PURE/f'late{weight}/training/checkpoint-500.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert state['epoch']==500 and state['server_contract']['config']['model']['learn_first_customer']
    trainer.model.load_state_dict(state['model_state_dict'],strict=True)
    trainer.model.eval()
    case=OUT/f'late{weight}'
    immutable(case/'reference_contract.json',dict(checkpoint_sha256=io.sha(checkpoint),selection='single greedy train-only'))
    for index,inst in enumerate(trainer.instance_pool):
        folder=case/'train_requests'/inst['name']
        if (folder/'reference.json').exists():
            groups=io.read(folder/'reference.json')['groups']
        else:
            pure._set_seed(42+index)
            with torch.no_grad():
                routes,_=trainer._best_solution(inst)
            groups=[[int(n) for n in r if n] for r in routes if any(r)]
            assert sorted(n for g in groups for n in g)==list(range(1,101))
            immutable(folder/'reference.json',dict(groups=groups))
        observation=public_observation(inst['name'])
        rain='RAIN' in inst['name']
        make_request(folder/('initial' if rain else 'accident'),weight,inst,groups,[observation],training=True)
    io.write(case/'prepare_status.json',dict(status='complete',scenarios=20))


def training_records(weight):
    records={}
    for row in io.read(OUT/'source/split.json')['train']:
        folder=OUT/f'late{weight}/train_requests'/row['name']
        groups=io.read(folder/'reference.json')['groups']
        neutral={str(k):{str(n):0.0 for n in g} for k,g in enumerate(groups)}
        record=dict(groups=groups,initial=io.read(folder/'initial/scores.json')
                    if (folder/'initial/scores.json').exists() else neutral)
        if (folder/'accident').exists():
            observation=public_observation(row['name'])
            record.update(accident=io.read(folder/'accident/scores.json'),
                          trigger=float(observation.get('revealed_at_min',observation['start_min'])))
        records[row['name']]=record
    return records


def train(weight, smoke=False):
    trainer,config=build(weight,cpu=smoke)
    import torch
    import phase2_online_v2_runtime as runtime
    records=training_records(weight)
    runtime.install_training(trainer,records)
    checkpoint=PURE/f'late{weight}/training/checkpoint-500.pt'
    out=OUT/f'late{weight}/training'
    contract=dict(bundle_sha256=io.sha(OUT/'version.json'),config=config,
        pure500_sha256=io.sha(checkpoint),guidance_sha256={str(p.relative_to(OUT)):io.sha(p)
        for p in (OUT/f'late{weight}/train_requests').rglob('scores.json')})
    last=out/'checkpoint-last.pt'
    epoch=io.restore(trainer,last,contract) if last.exists() else io.restore(trainer,checkpoint)
    if not last.exists():
        trainer.reward_log,trainer.loss_log=[],[]
    assert epoch>=500
    if smoke:
        for kind in ('RAIN','ACC'):
            inst=next(i for i in trainer.instance_pool if '_'+kind+'_' in i['name'])
            values=trainer._train_one_batch(inst,1)
            assert all(math.isfinite(float(v)) for v in values)
        print('PASS Phase2 backward smoke',flush=True)
        return
    io.write(out/'settings.json',contract)
    while epoch<1000:
        if (OUT/'STOP').exists():
            io.save_checkpoint(trainer,epoch,last,contract)
            raise RuntimeError('STOP requested; checkpoint saved')
        trainer._current_epoch=epoch+1
        reward,loss=trainer._train_one_epoch(epoch+1)
        assert math.isfinite(reward) and math.isfinite(loss)
        epoch+=1
        trainer.reward_log.append(reward)
        trainer.loss_log.append(loss)
        if epoch%10==0:
            io.save_checkpoint(trainer,epoch,last,contract)
        io.write(out/'status.json',dict(epoch=epoch,target=1000,status='training'))
        print(f'PHASE2 lambda={weight} epoch={epoch} reward={reward:.6f}',flush=True)
    io.save_checkpoint(trainer,epoch,out/'checkpoint-1000.pt',contract)
    io.write(out/'status.json',dict(epoch=epoch,status='complete'))


def evaluate(weight):
    trainer,config=build(weight,oneshot=True)
    import torch
    import train_vrptw as pure
    from vrptw_env import load_solomon,_solution_stats
    import phase2_online_v2_runtime as runtime
    checkpoint=OUT/f'late{weight}/training/checkpoint-1000.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert state['epoch']==1000
    trainer.model.load_state_dict(state['model_state_dict'],strict=True)
    runtime.install(trainer)
    trainer.model.eval()
    groups=io.read(OUT/'test_groups.json')
    out=OUT/f'late{weight}/eval_1000_test80_online'
    immutable(out/'contract.json',dict(checkpoint_sha256=io.sha(checkpoint),config=config,
        version_sha256=io.sha(OUT/'version.json'),test_reference_rollouts=0))
    results=io.read(out/'results.json') if (out/'results.json').exists() else []
    entries=io.read(OUT/'source/split.json')['test']
    assert [r['name'] for r in results]==[e['name'] for e in entries[:len(results)]]
    for index,entry in enumerate(entries):
        if index<len(results):
            continue
        inst=load_solomon(str(OUT/'source/data'/(entry['name']+'.txt')))
        observation=public_observation(inst['name'])
        rain='RAIN' in inst['name']
        folder=out/'requests'/inst['name']
        initial=obtain(make_request(folder/'initial',weight,inst,groups,[observation] if rain else []))
        trigger=None if rain else float(observation.get('revealed_at_min',observation['start_min']))
        def accident(state):
            observed=runtime.observed_state(trainer,inst)
            assert observed['vehicle_clock_min']>=trigger
            update_folder=folder/'accident'/f'vehicle_{observed["vehicle_id"]:03d}'
            immutable(update_folder/'scope.json',dict(vehicle_id=observed['vehicle_id'],
                valid_from_min=observed['vehicle_clock_min']))
            return obtain(make_request(update_folder,weight,inst,groups,[observation],state=observed))
        trainer.model.guidance_provider=runtime.Guidance(trainer,inst,groups,initial,
            trigger=trigger,on_accident=None if rain else accident)
        pure._set_seed(42+index)
        with torch.no_grad():
            routes,reward=trainer._best_solution(inst)
        assert trainer.env.pomo_size==1
        routes=[[int(n) for n in r] for r in routes]
        stats=_solution_stats(inst,routes)
        assert stats['n_served']==100
        row=dict(name=inst['name'],kind=entry['kind'],routes=routes,reward=float(reward),
            vehicles=len(stats['per_vehicle']),travel_time=float(stats['total_travel_time']),
            total_late=float(stats['total_late']),late_count=int(stats['late_count']))
        row['cost']=row['vehicles']/14+(row['travel_time']+weight*row['total_late'])/1696.95
        assert abs(row['cost']+row['reward'])<1e-4
        results.append(row)
        io.write(out/'results.json',results)
        io.write(out/'status.json',dict(status='evaluating',completed=len(results)))
    io.write(out/'summary.json',{key:dict(mean=statistics.mean(r[key] for r in results),
        std=statistics.stdev(r[key] for r in results)) for key in
        ('reward','cost','vehicles','travel_time','total_late','late_count')})
    io.write(out/'status.json',dict(status='complete',completed=80))


def queue():
    while io.read(PURE/'queue_status.json')['status']!='complete':
        if (OUT/'STOP').exists():
            return
        io.write(OUT/'queue_status.json',dict(status='waiting_for_pure_rl'))
        time.sleep(60)
    runner=OUT/'source/run_phase2_online_v2.py'
    for weight in (1,5,10,20):
        for stage in ('prepare','call-train','smoke','train','evaluate'):
            if (OUT/'STOP').exists():
                return
            io.write(OUT/'queue_status.json',dict(status='running',weight=weight,stage=stage))
            subprocess.run([sys.executable,'-u',str(runner),stage,'--weight',str(weight),'--allow-api'],cwd=ROOT,check=True)
    io.write(OUT/'queue_status.json',dict(status='complete'))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('stage',choices=['freeze','queue','prepare','call','call-train','smoke','train','evaluate'])
    ap.add_argument('--weight',type=int,choices=[1,5,10,20],default=1)
    ap.add_argument('--folder',type=Path)
    ap.add_argument('--allow-api',action='store_true')
    args=ap.parse_args()
    if args.stage!='freeze':
        verify()
    if args.stage in ('queue','call','call-train','evaluate') and not args.allow_api:
        ap.error('Paid calls require --allow-api')
    if args.stage=='freeze':
        freeze()
    elif args.stage=='queue':
        verify()
        queue()
    elif args.stage=='call':
        call(args.folder)
    elif args.stage=='call-train':
        for p in sorted((OUT/f'late{args.weight}/train_requests').glob('*/*/request.json')):
            obtain(p.parent)
    elif args.stage=='prepare':
        prepare_train(args.weight)
    elif args.stage in ('smoke','train'):
        train(args.weight,smoke=args.stage=='smoke')
    else:
        evaluate(args.weight)


if __name__=='__main__':
    main()
