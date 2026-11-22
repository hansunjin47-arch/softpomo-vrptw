"""Portable RC101 RL+LKH example experiment; no dependency on local result folders."""
import argparse, copy, hashlib, json
from pathlib import Path
import torch
from analysis import pair_kim_rl_example as shared
from experiments.concurrent_fleet_pair_kim import portable as legacy
from experiments.concurrent_fleet_pair_kim import rl_lkh_trainer as trainer
from experiments.concurrent_fleet_pair_kim.run import GuidedEnv
from experiments.concurrent_fleet_pair_kim.confidence_head_guidance import ConfidenceHeadGuidedPolicy
from experiments.concurrent_fleet_bias_guided.llm_retry import RetryingClient
from experiments.concurrent_fleet_v9_guided.prompts import build_test_request,parse_response,parse_test_response
from experiments.concurrent_fleet_guided.prompts import PUBLIC_FIELDS
from experiments.concurrent_fleet_pair.io import write

ROOT=Path(__file__).resolve().parents[2]
INPUT=ROOT/'experiment_inputs/rc_rl_lkh'
read=legacy.read
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def inputs(late):
    for name,digest in read(INPUT/'manifest.json')['sha256'].items():
        assert sha(INPUT/name)==digest,'Comparison input hash mismatch: '+name
    source,split,groups,_=legacy.inputs(argparse.Namespace(case='rc101',vehicle_cost=100))
    cfg=read(INPUT/'config_l5.json');cfg['environment']['late_penalty']=late
    assert cfg['environment']['require_dispatch_capacity'] and cfg['environment']['protect_last_service_vehicle']
    assert cfg['environment']['normalize_advantage_std'] and cfg['trainer']['epochs']==1000
    ref=read(source/'reference.json');comparison=read(INPUT/'comparison_example.json')
    assert comparison['scenario']==ref['scenario']
    return source,split,groups,cfg,ref,comparison

def enrich(request,base,ref,comparison,*,offline=False):
    req=shared.enrich(request,base,ref,offline=offline,new_head_beta=5.)
    ctx=json.loads(req['messages'][1]['content']);ctx['rl_lkh_comparison_example']=comparison
    req['messages'][1]['content']=json.dumps(ctx,separators=(',',':'),allow_nan=False)
    req['messages'][0]['content']+='\nThe rl_lkh_comparison_example evaluates two routes on the SAME separate example instance and event, not the target. Compare customer assignments, visit order, service timing, travel and lateness. The shared fixed_instance supplies time windows. Both routes are illustrative, not optimal and not confidence labels. Do not copy routes, fleet count or customer rankings mechanically. Infer target-specific scores from the TARGET public information, objective and guidance semantics. LKH-3 used a historical Train-only matrix; its reported metrics are event-aware replay measurements, not claims of future-event knowledge. The Pure RL route is identical to the historical example above; its timing details are added only for comparison.'
    ratio=ctx['objective_coefficients']['total_lateness_min']/ctx['objective_coefficients']['driving_time_min']
    if abs(ratio-5)>1e-8:
        req['messages'][0]['content']+=f'\nREFERENCE OBJECTIVE: Both historical example results were evaluated with lateness weight 5. The TARGET lateness weight is {ratio:g}. Their reported rewards are historical, not target-weight scores; use TARGET objective_coefficients for guidance. No new Pure RL training or LKH optimization at the target weight is implied.'
    return req

def identity(late):
    return dict(late=late,vehicle_cost=100,new_head_beta=5,routing_beta=5,llm='openai/gpt-5.4',
        prompt='max_confidence_new_head_v1_plus_rl_lkh',
        comparison_hashes=read(INPUT/'manifest.json')['sha256'],
        input_hashes=read(ROOT/'experiment_inputs/new_head/rc101/manifest.json')['sha256'],
        prompt_code={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),Path(shared.__file__),ROOT/'experiments/concurrent_fleet_v9_guided/prompts.py']})

def prepare(args):
    source,split,groups,cfg,ref,comparison=inputs(args.late)
    art=args.artifact;meta=identity(args.late)
    if (art/'generation.json').exists():assert read(art/'generation.json')==meta,'Cache settings/source mismatch; choose a fresh artifact directory'
    else:write(art/'generation.json',meta)
    template=read(source/'request_template.json');assert template['model']=='openai/gpt-5.4'
    for name,value in [('config.json',cfg),('split.json',split),('groups.json',groups),('reference.json',ref),('comparison_example.json',comparison),('request_template.json',template)]:
        path=art/name
        if path.exists():assert read(path)==value,'Artifact mismatch: '+name
        else:write(path,value)
    client=RetryingClient(art/'calls',lambda req,payload:parse_test_response(payload,groups,json.loads(req['messages'][1]['content'])['selectable_customers']),allow_paid=args.allow_api)
    inst,_=legacy.load(split['train'][0]);inst=copy.deepcopy(inst);inst['preset_events']=[];inst['name']='RC101_BASE_REFERENCE'
    path=art/'base_confidence.json'
    if path.exists():base=read(path)
    else:
        base=shared.as_list(client(enrich(build_test_request(inst,cfg['environment'],groups,[],template=template),None,ref,comparison)),groups)
        write(path,base)
    parse_response(base,groups)
    for index,row in enumerate(split['train']):
        path=art/'train_cache'/(row['name']+'.json')
        if not path.exists():
            inst,reports=legacy.load(row)
            req=build_test_request(inst,cfg['environment'],groups,reports,template=template)
            ctx=json.loads(req['messages'][1]['content']);ctx['public_events']=[{k:v for k,v in r.items() if k in PUBLIC_FIELDS} for r in reports]
            req['messages'][1]['content']=json.dumps(ctx)
            event=shared.as_list(client(enrich(req,base,ref,comparison,offline=True)),groups)
            acc=reports[0]['event_type']=='ACCIDENT'
            write(path,dict(initial=base if acc else event,post_disclosure=event if acc else None,revealed_at_min=reports[0]['revealed_at_min']))
        cache=read(path);parse_response(cache['initial'],groups)
        if cache['post_disclosure'] is not None:parse_response(cache['post_disclosure'],groups)
        print('CACHE',index+1,20,row['name'],flush=True)
    write(art/'complete.json',dict(train_count=20,base_count=1,test='Separate causal evaluation required',generation=meta))

def train(args):
    source,split,groups,cfg,ref,comparison=inputs(args.late)
    art=args.artifact
    assert read(art/'complete.json')['train_count']==20
    assert read(art/'generation.json')==identity(args.late),'Do not reuse caches from another setting/prompt'
    assert read(art/'config.json')==cfg and read(art/'split.json')==split
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
    trainer.TensorFleetEnv=CachedEnv;trainer.FleetPolicy=ConfidenceHeadGuidedPolicy;trainer.load=legacy.load
    calls=0
    if args.resume:
        saved=torch.load(args.out/f'run/late{args.late}/checkpoint-last.pt',map_location='cpu',weights_only=False)
        calls=int(saved['state']['update']);del saved
    original=trainer.train_batch
    def batch(model,opt,*a,**kw):
        nonlocal calls
        epoch=calls//20+1
        for pg in opt.param_groups:
            pg.setdefault('stability_initial_lr',pg['lr'])
            pg['lr']=pg['stability_initial_lr']*(1. if epoch<=500 else 1.-.9*(epoch-500)/500.)
        summary,detail=original(model,opt,*a,**kw)
        summary.update(new_head_beta=5.,learning_rate=opt.param_groups[0]['lr']);calls+=1
        return summary,detail
    trainer.train_batch=batch
    meta=dict(seed=args.seed,generation=identity(args.late),epochs=1000,capacity_mask=True,return_mask=True,
        lr='1e-4 through500, linear to1e-5 at1000',
        caches={n:hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest() for n,c in caches.items()},
        code={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),Path(trainer.__file__),ROOT/'experiments/concurrent_fleet_pair_kim/confidence_head_guidance.py',ROOT/'experiments/concurrent_fleet_pair_kim/run.py',ROOT/'experiments/concurrent_fleet_guided/guidance.py']})
    if args.validate_only:
        for row in split['train']:
            inst,_=legacy.load(row);env=CachedEnv(inst,1,cfg['environment'],'cpu')
            assert env.guidance.beta==5 and env.params['require_dispatch_capacity']
        print(json.dumps(dict(seed=args.seed,late=args.late,caches=20,capacity_mask=True,lr_decay=True,new_bias=5,routing_beta=5)));return
    if args.resume:assert read(args.out/'experiment.json')==meta,'Resume provenance mismatch'
    else:
        assert not args.out.exists(),'Use a new output directory or --resume'
        write(args.out/'experiment.json',meta)
    write(args.out/'config.json',cfg)
    a=argparse.Namespace(out=args.out/'run',seed=args.seed,vehicle_cost=100.,resume=args.resume,smoke_updates=0,
        config=args.out/'config.json',source=source/'dataset',weights=[args.late],reward_denominator=1696.95,
        return_mask=True,no_advantage_std=False,new_customer_epsilon=.05,epsilon_final=.01,epsilon_zero_epoch=1000,resume_from=None)
    trainer.run(a)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['validate','prepare','train'])
    parser.add_argument('--late',type=int,choices=[5,10],default=10)
    parser.add_argument('--seed',type=int,choices=[44,45,46],default=44)
    parser.add_argument('--artifact',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--allow-api',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    args.artifact=(args.artifact or ROOT/f'cache/rc_rl_lkh_v100_l{args.late}').resolve()
    args.out=(args.out or ROOT/f'result/rc_rl_lkh_v100_l{args.late}_s{args.seed}').resolve()
    if args.command=='validate':
        _,split,groups,cfg,ref,comparison=inputs(args.late)
        print(json.dumps(dict(late=args.late,train=len(split['train']),test=len(split['test']),groups=len(groups),reference=ref['scenario'],capacity_mask=cfg['environment']['require_dispatch_capacity'])));return
    if args.command=='prepare':prepare(args)
    else:train(args)
if __name__=='__main__':main()
