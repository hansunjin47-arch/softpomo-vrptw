"""Portable C/R/RC input audit, reference/cache generation and training CLI."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import torch
from experiments.concurrent_fleet_pair_kim import portable_trainer as trainer
from experiments.concurrent_fleet_pair_kim.run import GuidedEnv, GuidedPolicy
from experiments.concurrent_fleet_pair_kim.confidence_head_guidance import ConfidenceHeadGuidedPolicy
from experiments.concurrent_fleet_pair.policy import FleetPolicy
from experiments.concurrent_fleet_pair.env import TensorFleetEnv
from experiments.concurrent_fleet_pair.io import write
from experiments.concurrent_fleet_bias_guided.support import load, read
from experiments.concurrent_fleet_v3 import run as data
from experiments.concurrent_fleet_v9_guided.prompts import parse_response, parse_test_response, build_test_request
from experiments.concurrent_fleet_guided.prompts import PUBLIC_FIELDS
from experiments.concurrent_fleet_bias_guided.llm_retry import RetryingClient
from analysis import pair_kim_rl_example as shared

ROOT=Path(__file__).resolve().parents[2]
DEN={'c101':828.94,'r101':1650.8,'rc101':1696.95}

def inputs(args):
    source=ROOT/'experiment_inputs/new_head'/args.case
    manifest=read(source/'manifest.json')
    for rel,digest in manifest['sha256'].items():
        if hashlib.sha256((source/rel).read_bytes()).hexdigest()!=digest:
            raise ValueError('Input hash mismatch: '+rel)
    split=read(source/'dataset/split.json');groups=read(source/'groups.json')
    assert len(split['train'])==20 and len(split['test'])==80
    assert sorted(n for g in groups for n in g)==list(range(1,101))
    assert all(r['name'].startswith(args.case.upper()+'_') for r in split['train']+split['test'])
    cfg=read(source/'config.json')
    cfg['environment'].update(vehicle_penalty=args.vehicle_cost*100/DEN[args.case],
        late_penalty=5,protect_last_service_vehicle=True,normalize_advantage_std=True)
    cfg['trainer'].update(epochs=1000,seed=42,train_instances=[r['name'] for r in split['train']])
    data.SOURCE=source/'dataset'
    for row in split['train']+split['test']:load(row)
    return source,split,groups,cfg

def artifact(args):
    return (args.artifact or ROOT/f'cache/new_head/{args.case}_v{args.vehicle_cost}').resolve()

def validate_reference(example,case,cfg):
    if example.get('reference_role')=='cross_setting_example':
        assert case=='c101' and example.get('routing_mode')=='sequential'
        assert example.get('objective') and example.get('source_provenance')
        assert example['checkpoint_epoch']==500
    else:
        expected=cfg['environment']['vehicle_penalty']/100
        assert abs(example['objective_coefficients']['vehicles']-expected)<1e-8,'Generate a reference for this vehicle cost first'

def enrich_reference(request,base,example,*,offline=False):
    request=shared.enrich(request,base,example,offline=offline)
    if example.get('reference_role')=='cross_setting_example':
        request['messages'][0]['content']+= (
            '\nREFERENCE PROVENANCE: The C101 rl_reference_example was produced by a '
            'SEQUENTIAL-routing Pure RL policy (seed42, epoch500), whereas the TARGET '
            'uses CONCURRENT routing. The example also uses its own objective '
            '(vehicles/10 + (driving_minutes + 5*lateness_minutes)/828.94). '
            'Its reported reward belongs only to that source objective and simulator. '
            'Do not interpret its route, vehicle count, timing, or reward as a concurrent '
            'target result or optimum. Use it only as a qualitative illustration; '
            'apply the TARGET objective_coefficients, current public information and '
            'concurrent action rules when producing target confidence.')
    return request

def reference(args):
    source,split,groups,cfg=inputs(args)
    run=args.pure_run.resolve();actual=read(run/'protocol.json')['configs']['5']
    assert actual['model']==cfg['model'] and actual['environment']==cfg['environment'],'Pure checkpoint settings differ'
    assert read(run/'split.json')['train']==split['train']
    art=artifact(args);art.mkdir(parents=True,exist_ok=True)
    if (art/'reference.json').exists():raise ValueError('Reference already exists; use another artifact directory')
    for name,value in [('split.json',split),('groups.json',groups),('request_template.json',read(source/'request_template.json'))]:
        write(art/name,value)
    shared.PURE=run;shared.OLD=art;shared.ART=art;shared.LATE=5
    shared.EXAMPLE=args.case.upper()+'_RAIN_REFERENCE_NEWHEAD'
    torch.set_num_threads(1)
    value=shared.reference()
    print('Independent reference generated without API:',value['observed_performance'])

def prepare(args):
    source,split,groups,cfg=inputs(args);art=artifact(args)
    ref=art/'reference.json'
    if not ref.exists():ref=source/'reference.json'
    if not ref.exists():raise ValueError('Run reference --pure-run PATH first; no matching concurrent Pure RL example is packaged for this case')
    example=read(ref)
    validate_reference(example,args.case,cfg)
    template=read(source/'request_template.json')
    provenance=dict(case=args.case,vehicle_cost=args.vehicle_cost,late=5,reference_sha256=hashlib.sha256(ref.read_bytes()).hexdigest(),
        inputs=read(source/'manifest.json')['sha256'],prompt_code={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'analysis/pair_kim_rl_example.py',ROOT/'experiments/concurrent_fleet_v9_guided/prompts.py']})
    if (art/'generation.json').exists():assert read(art/'generation.json')==provenance,'Cache provenance mismatch'
    else:write(art/'generation.json',provenance)
    if args.allow_api and not os.environ.get('OPENROUTER_API_KEY'):raise ValueError('Set OPENROUTER_API_KEY in this terminal')
    for name,value in [('groups.json',groups),('split.json',split),('config.json',cfg),('reference.json',example),('request_template.json',template)]:
        if (art/name).exists():assert read(art/name)==value,name+' mismatch'
        else:write(art/name,value)
    def validate(req,payload):parse_test_response(payload,groups,json.loads(req['messages'][1]['content'])['selectable_customers'])
    client=RetryingClient(art/'calls',validate,allow_paid=args.allow_api)
    inst,_=load(split['train'][0]);inst=copy.deepcopy(inst);inst['preset_events']=[];inst['name']=args.case.upper()+'_BASE_REFERENCE'
    base_path=art/'base_confidence.json'
    if base_path.exists():base=read(base_path)
    else:
        base=shared.as_list(client(enrich_reference(build_test_request(inst,cfg['environment'],groups,[],template=template),None,example)),groups)
        write(base_path,base)
    for i,row in enumerate(split['train']):
        path=art/'train_cache'/(row['name']+'.json')
        if not path.exists():
            inst,reports=load(row)
            req=build_test_request(inst,cfg['environment'],groups,reports,template=template)
            ctx=json.loads(req['messages'][1]['content'])
            ctx['public_events']=[{k:v for k,v in r.items() if k in PUBLIC_FIELDS} for r in reports]
            req['messages'][1]['content']=json.dumps(ctx)
            event=shared.as_list(client(enrich_reference(req,base,example,offline=True)),groups)
            acc=reports[0]['event_type']=='ACCIDENT'
            write(path,dict(initial=base if acc else event,post_disclosure=event if acc else None,revealed_at_min=reports[0]['revealed_at_min']))
        cache=read(path);parse_response(cache['initial'],groups)
        if cache['post_disclosure'] is not None:parse_response(cache['post_disclosure'],groups)
        print('CACHE',i+1,20,row['name'],flush=True)
    write(art/'complete.json',dict(train_count=20,test='causal evaluation separately; not cached here'))

def train(args):
    source,split,groups,cfg=inputs(args)
    out=(args.out or ROOT/f'result/max_confidence_head/{args.case}_{args.method}_v{args.vehicle_cost}_s{args.seed}').resolve()
    art=artifact(args);caches={}
    if args.method!='pure':
        assert read(art/'complete.json')['train_count']==20
        assert read(art/'split.json')==split and read(art/'groups.json')==groups
        assert read(art/'config.json')['environment']==cfg['environment'],'Cache environment mismatch'
        caches={r['name']:read(art/'train_cache'/(r['name']+'.json')) for r in split['train']}
        for c in caches.values():
            parse_response(c['initial'],groups)
            if c['post_disclosure'] is not None:parse_response(c['post_disclosure'],groups)
    class CachedEnv(GuidedEnv):
        def __init__(self,inst,count,params,device='cuda'):
            c=caches[inst['name']]
            super().__init__(inst,count,params,device,groups=groups,scores=parse_response(c['initial'],groups),beta=5.)
            if c['post_disclosure'] is not None:
                t=c['revealed_at_min'];self.update_event_confidence(parse_response(c['post_disclosure'],groups),revealed_at_min=t,effective_time_min=t)
    cfg['environment']['require_dispatch_capacity']=getattr(args,'dispatch_capacity_mask',False)
    cfg['trainer']['lr_decay_after500']=getattr(args,'lr_decay',False)
    trainer.TensorFleetEnv=TensorFleetEnv if args.method=='pure' else CachedEnv
    trainer.FleetPolicy={'pure':FleetPolicy,'kim':GuidedPolicy,'newhead':ConfidenceHeadGuidedPolicy}[args.method]
    ConfidenceHeadGuidedPolicy.new_head_beta=args.head_beta;trainer.load=load
    provenance=dict(case=args.case,method=args.method,seed=args.seed,vehicle_cost=args.vehicle_cost,head_beta=args.head_beta,
        dispatch_capacity_mask=cfg['environment']['require_dispatch_capacity'],lr_decay=cfg['trainer']['lr_decay_after500'],
        inputs=read(source/'manifest.json'),cache_sha256={n:hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest() for n,c in caches.items()},
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'experiments').rglob('*.py')})
    if args.resume:assert read(out/'experiment.json')==provenance,'Resume mismatch'
    elif (out/'run').exists() or (out/'experiment.json').exists():raise ValueError('Output already exists')
    else:write(out/'experiment.json',provenance);write(out/'config.json',cfg)
    a=argparse.Namespace(out=out/'run',seed=args.seed,vehicle_cost=args.vehicle_cost,resume=args.resume,smoke_updates=args.smoke_updates,
        config=out/'config.json',source=source/'dataset',weights=[5],reward_denominator=DEN[args.case],return_mask=True,
        no_advantage_std=False,new_customer_epsilon=.05,epsilon_final=.01,epsilon_zero_epoch=1000,resume_from=None)
    trainer.run(a)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['validate','prepare','reference','train'])
    p.add_argument('--case',choices=list(DEN),required=True)
    p.add_argument('--vehicle-cost',type=int,choices=[100,200],default=100)
    p.add_argument('--artifact',type=Path)
    p.add_argument('--out',type=Path)
    p.add_argument('--pure-run',type=Path)
    p.add_argument('--allow-api',action='store_true',help='Allow paid cache generation (normally 21 logical requests)')
    p.add_argument('--method',choices=['pure','kim','newhead'],default='newhead')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--head-beta',type=float,default=5.)
    p.add_argument('--dispatch-capacity-mask',action='store_true',help='Prevent closing dispatch below ceil(total demand/capacity)')
    p.add_argument('--lr-decay',action='store_true',help='Keep initial LR through500, linearly decay to10% at1000')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--smoke-updates',type=int,default=0)
    args=p.parse_args()
    if args.command=='validate':
        _,split,groups,_=inputs(args);print('VALID',args.case,'train',len(split['train']),'test',len(split['test']),'heads',[g[0] for g in groups])
    elif args.command=='reference':
        if args.pure_run is None:p.error('--pure-run required for reference')
        reference(args)
    elif args.command=='prepare':prepare(args)
    else:train(args)

if __name__=='__main__':main()
