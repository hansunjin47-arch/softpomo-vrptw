"""One RC BASE API pilot with a fixed, separate sequential RL example."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'server_export/rc101_seq_train20_late5_seeds42_46_20261023/seed42'
OUT=ROOT/'result/kim_rl_example_base_pilot_rc_20261027'
NAME='RC101_RAIN_REFERENCE_20261027'
sys.path.insert(0,str(ROOT/'server'))
sys.path.insert(0,str(BUNDLE))
import train_cached as io

CASE='rc101'
K0,T0=14,1696.95


def configure(case):
    global CASE, BUNDLE, OUT, NAME, K0, T0
    from generate_kim_l5_all_gpt import CASES
    CASE=case
    K0,T0,_=CASES[case]
    short=case.replace('101','')
    OUT=ROOT/f'result/kim_rl_example_base_pilot_{short}_20261027'
    NAME=f'{case.upper()}_RAIN_REFERENCE_20261027'
    if case != 'rc101':
        BUNDLE=ROOT/f'server_export/cr101_seq_l5_10_15_20_s42_46_20261023/{case}/late5/seed42'
    sys.path.insert(0,str(BUNDLE))


def reference():
    import torch
    from vrptw_env import load_solomon, _solution_stats
    digest=io.verify_bundle(BUNDLE)
    config=io.read(BUNDLE/'configs/pure.json')
    checkpoint=BUNDLE/'runs/pure/checkpoint-500.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert saved['server_contract']==dict(bundle_sha256=digest,config=config)
    assert not (OUT/'reference.json').exists(), 'Reference already fixed'
    split=io.read(BUNDLE/'split.json')
    names=[r['name'] for section in ['train','test'] for r in split[section]]
    assert NAME not in names
    # Fix a new event before decoding; no search or selection by performance.
    source_name=f'{CASE.upper()}_RAIN_P018'
    inst=load_solomon(str(BUNDLE/'data'/(source_name+'.txt')))
    inst['name']=NAME
    event=copy.deepcopy(inst['preset_events'][0])
    event.update(trigger_time=123.456,duration=44.321)
    inst['preset_events']=[event]
    for name in names:
        existing=load_solomon(str(BUNDLE/'data'/(name+'.txt')))
        assert existing['preset_events'] != inst['preset_events'], 'Duplicate event case'
    extract=copy.deepcopy(config)
    extract['environment']['pomo_size']=1
    extract['model'].update(learn_first_customer=True,eval_type='argmax')
    extract['trainer'].update(train_instances=[source_name],pomo_auto=False,
        use_augmentation=False,n_mc_samples=1,eval_top_k=None)
    trainer=io.build_trainer(extract,BUNDLE,OUT/'reference_runtime',cpu=True)
    trainer.model.load_state_dict(saved['model_state_dict'],strict=True)
    trainer.model.eval()
    with torch.no_grad(): routes,reward=trainer._best_solution(inst)
    assert trainer.env.pomo_size==1
    routes=[[int(n) for n in route] for route in routes]
    assert sorted(n for route in routes for n in route if n)==list(range(1,101))
    stats=_solution_stats(inst,routes)
    metrics=dict(reward=float(reward),vehicles=len(stats['per_vehicle']),
        travel_time=float(stats['total_travel_time']),total_late=float(stats['total_late']),
        late_count=int(stats['late_count']),served=int(stats['n_served']))
    assert abs(-metrics['reward']-(metrics['vehicles']/K0+(metrics['travel_time']+5*metrics['total_late'])/T0))<1e-4
    io.write(OUT/'reference.json',dict(name=NAME,routes=routes,metrics=metrics,event=event,
        checkpoint_sha256=io.sha(checkpoint),checkpoint_epoch=500,seed=42,
        evaluation='CPU greedy one rollout, learned first customer, no augmentation, no sampling search',
        exclusion='Not in train20/test80; event differs from every train/test event',
        selection='Fixed event before decoding; one decode only'))
    print(json.dumps(metrics),flush=True)


def prepare():
    from generate_kim_baseacc import make_request
    from run_gpt_bks_late10 import immutable
    old=ROOT/f'result/kim_l5_gpt_rc_r_c_20261026/{CASE}/train'
    protocol=io.read(old/'protocol.json')
    source=old/'requests'/protocol['names'][0]
    request,context=make_request(io.read(source/'request.json'),io.read(source/'context.json'))
    ref=io.read(OUT/'reference.json')
    example_fixed=copy.deepcopy(context['fixed_instance'])
    example_fixed.pop('groups')
    event=ref['event']
    context['rl_reference_example']=dict(scenario=NAME,fixed_instance=example_fixed,
        public_event=dict(event_type='RAIN',affected_nodes=event['nodes'],rainfall_mm_per_hour=event['rainfall_mm'],
            start_min=event['trigger_time'],end_min=event['trigger_time']+event['duration'],revealed_at_min=0),
        routes=ref['routes'],observed_performance=ref['metrics'],
        objective=f'vehicles/{K0} + (driving_minutes + 5*lateness_minutes)/{T0}',
        decoder=ref['evaluation'])
    text=request['messages'][0]['content'].rsplit('\n',1)[0]
    text=text.replace('No RL route, simulator-derived arrival/slack, travel-time multiplier, or counterfactual schedule is supplied.',
        'No RL solution or measured performance for the TARGET instance is supplied.')
    text+='\nThe rl_reference_example is a SEPARATE case, not the target BASE. Its RL route and measured performance are a single non-optimal demonstration, not a correct answer or confidence label. The example has rain; the target BASE has NO event. Do not transfer that rain to the target. Do not copy its customer order or infer target performance from it. Use it only as context for possible RL behavior and cost trade-offs. Judge target confidence from the target instance and heuristic groups.'
    request['messages']=[dict(role='user',content=text+'\n'+json.dumps(context,separators=(',',':'),allow_nan=False))]
    immutable(OUT/'protocol.json',dict(names=['BASE'],clusters=protocol['clusters'],model=request['model'],
        variant='Kim plus fixed separate RL example; BASE only pilot',reference_sha256=io.sha(OUT/'reference.json')))
    immutable(OUT/'requests/BASE/context.json',context)
    immutable(OUT/'requests/BASE/request.json',request)
    immutable(OUT/'request_hashes.json',{'BASE':io.sha(OUT/'requests/BASE/request.json')})
    assert context['public_event']=={} and 'rl_reference_example' in context
    print('Prepared one mixed BASE request; target has no event',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['reference','prepare','call']);p.add_argument('--allow-api',action='store_true');p.add_argument('--case',choices=['rc101','r101','c101'],default='rc101');a=p.parse_args()
    configure(a.case)
    if a.stage=='reference':reference()
    elif a.stage=='prepare':prepare()
    else:
        if not a.allow_api:p.error('--allow-api required')
        from run_gpt_bks_late10 import call
        call(OUT,kim=True)
