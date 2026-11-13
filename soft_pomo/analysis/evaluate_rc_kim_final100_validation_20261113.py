"""Frozen final100 validation. Default is offline; live requires --execute-api."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch
from analysis.evaluate_rc101_mask_kim_20261112 import BUNDLE, RECEIVED, ROOT
from analysis.validate_rc_curriculum_events_20261112 import DATA, SOURCE, signature
from analysis.pair_kim_rl_example import enrich
from analysis.train_rc_curriculum_confirmatory_20261112 import compare
from experiments.concurrent_fleet_pair_kim.run import GuidedPolicy, OnlineEnv
from experiments.concurrent_fleet_v9_guided import online
from experiments.concurrent_fleet_v9_guided.prompts import build_test_request, parse_test_response
from experiments.concurrent_fleet_bias_guided.llm_retry import RetryingClient
from experiments.concurrent_fleet_pair.io import write
from vrptw_env import load_solomon

OUT=ROOT/'result/rc_kim_final100_validation_20261113'
MODELS=ROOT/'result/rc_kim_persistence_20261112'
OLD=ROOT/'result/rc101_pair100_l5_mask_kim_test_20261112/calls'
ART=BUNDLE/'artifacts/vehicle100'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def key(req): return hashlib.sha256(json.dumps(req,sort_keys=True,allow_nan=False).encode()).hexdigest()
def cfg(seed): return read(RECEIVED/f'runs/vehicle100/seed{seed}/protocol.json')['configs']['5']
def instance(name):
    p=DATA/(name+'.txt'); inst=load_solomon(str(p)); inst['name']=name
    return inst,[read(p.with_suffix('.observations.json'))]
GROUPS=read(ART/'groups.json'); TEMPLATE=read(ART/'request_template.json')
BASE=read(ART/'base_confidence.json'); EXAMPLE=read(ART/'reference.json')

def request(*args,**kwargs):
    req=enrich(build_test_request(*args,**kwargs),BASE,EXAMPLE)
    ctx=json.loads(req['messages'][1]['content']); env=kwargs.get('env')
    clock=0. if env is None else float(env.time[kwargs.get('rollout_id',0)])
    assert all(e['revealed_at_min']<=clock and 'multiplier' not in e for e in ctx['public_events'])
    if env is not None:
        assert ctx['fleet']['clock_min']==clock and 'vehicles' in ctx['fleet']
        remaining=[i for i in range(1,env.n+1) if int(env.customer[0,i])==0]
        assert ctx['selectable_customers']==remaining
        assert all(e['customer'] in remaining for e in ctx['previous_confidence']['confidence'])
    return req

def validate(req,payload):
    return parse_test_response(payload,GROUPS,json.loads(req['messages'][1]['content'])['selectable_customers'])

def prepare():
    frozen=read(DATA.parent/'protocol.json'); assert sha(DATA/'manifest.json')==frozen['data_sha256']
    cases=read(DATA/'manifest.json')['cases']; assert len(cases)==20
    split=read(SOURCE/'split.json')
    excluded={signature(load_solomon(str(SOURCE/'data'/(r['name']+'.txt')))['preset_events'][0]) for r in split['train']+split['test']}
    ref=dict(load_solomon(str(SOURCE/'data'/(split['train'][0]['name']+'.txt')))['preset_events'][0]);ref.update(trigger_time=123.456,duration=44.321)
    excluded.add(signature(ref))
    old={p.parent.name:p for p in OLD.rglob('response.json')}
    rows=[]; hashes={}; seen=set()
    for case in cases:
        name=case['name']; inst,reports=instance(name); sig=signature(inst['preset_events'][0])
        assert sig not in excluded and sig not in seen; seen.add(sig)
        assert sha(DATA/(name+'.txt'))==case['file_sha256']
        assert sha(DATA/(name+'.observations.json'))==frozen['observation_hashes'][name]
        req=request(inst,cfg(42)['environment'],GROUPS,reports,template=TEMPLATE,beta=5.)
        if case['kind']!='RAIN':
            assert req==request(inst,cfg(42)['environment'],GROUPS,[],template=TEMPLATE,beta=5.)
        k=key(req); cached=old.get(k)
        if cached: validate(req,read(cached))
        rows.append(dict(name=name,request_sha256=k,exact_cache=str(cached) if cached else None))
        for suffix in ('.txt','.observations.json'): hashes[str(DATA/(name+suffix))]=sha(DATA/(name+suffix))
    models={}
    for seed in (42,43):
        for arm in ('control','curriculum'):
            p=MODELS/f'seed{seed}'/arm/'checkpoint-100.pt'
            saved=torch.load(p,map_location='cpu',weights_only=False); assert saved['epoch']==100
            model=GuidedPolicy(**cfg(seed)['model']);model.load_state_dict(saved['model_state_dict'],strict=True)
            models[f'{seed}/{arm}']=dict(path=str(p),sha256=sha(p))
    for p in ART.glob('*.json'): hashes[str(p)]=sha(p)
    for seed in (42,43):
        p=RECEIVED/f'runs/vehicle100/seed{seed}/protocol.json';hashes[str(p)]=sha(p)
    for folder in ('concurrent_fleet_pair','concurrent_fleet_pair_kim','concurrent_fleet_v9_guided','concurrent_fleet_bias_guided','concurrent_fleet_guided','concurrent_fleet_v8','concurrent_fleet_v7','concurrent_fleet_v5'):
        for p in (ROOT/'experiments'/folder).glob('*.py'): hashes[str(p)]=sha(p)
    for p in (Path(__file__),ROOT/'analysis/pair_kim_rl_example.py',ROOT/'vrptw_env.py'):hashes[str(p)]=sha(p)
    plan=dict(models=models,hashes=hashes,cases=rows,excluded_identity_count=len(excluded),
        max_new_unique_requests=50,max_HTTP_attempts=150,attempts_per_request=3,
        conditions='RC101 vehicle100 lambda5 return-mask ON beta5; final100 only; single greedy AMP',
        purpose='New-event validation on same geometry, not untouched Test or new geometry',
        selection='No target/schedule/checkpoint selection using these results',
        baseline='Existing Pure validation40 is different extra-training budget and seed set; do not pool as matched final100. LKH must run same20/cfg before reporting a direct baseline.',
        api_executed=False,gpu_executed=False,test_performance_accessed=False)
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists(): assert read(OUT/'protocol.json')==plan,'Frozen protocol mismatch'
    else: write(OUT/'protocol.json',plan)
    write(OUT/'dry_run.json',dict(passed=True,models=4,cases=20,model_case_evaluations=80,
        exact_cached_initial_cases=sum(bool(r['exact_cache']) for r in rows),
        new_unique_initial=len({r['request_sha256'] for r in rows if not r['exact_cache']}),
        causal_initial_ACC_equivalence_verified=True,live_remaining_fleet_assertions_installed=True,
        api_calls=0,gpu_calls=0,next_step_requires_API_notice=True))
    return plan

def execute(plan):
    assert os.environ.get('OPENROUTER_API_KEY') and torch.cuda.is_available()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    existing={p.parent.name:p for p in OLD.rglob('response.json')}
    def fallback(req):
        p=existing.get(key(req))
        if p is None: raise RuntimeError('No exact cache')
        return read(p)
    paid=RetryingClient(OUT/'calls',validate,allow_paid=True,fallback=fallback)
    ledger=read(OUT/'request_ledger.json') if (OUT/'request_ledger.json').exists() else []
    def client(req):
        k=key(req)
        try: payload=fallback(req);validate(req,payload);return payload
        except (RuntimeError,ValueError,AssertionError,KeyError,TypeError): pass
        if k not in ledger:
            assert len(ledger)<50,'API unique-request bound reached'
            ledger.append(k);write(OUT/'request_ledger.json',ledger)
        return paid(req)
    online.build_test_request=request;online.OnlineTestEnv=OnlineEnv
    for seed in (42,43):
        results={}
        for arm in ('control','curriculum'):
            model=GuidedPolicy(**cfg(seed)['model']).cuda().eval()
            model.load_state_dict(torch.load(plan['models'][f'{seed}/{arm}']['path'],map_location='cpu',weights_only=False)['model_state_dict'])
            rows=[]
            for case in plan['cases']:
                name=case['name'];folder=OUT/f'seed{seed}'/arm/name
                if (folder/'result.json').exists(): result=read(folder/'result.json')
                else:
                    inst,reports=instance(name)
                    result=online.evaluate_instance(model,inst,cfg(seed)['environment'],GROUPS,reports,template=TEMPLATE,client=client,beta=5.,out=folder,amp=True)
                rows.append(dict(name=name,**result['metrics']))
                write(OUT/f'seed{seed}'/(arm+'.json'),rows)
                write(OUT/'status.json',dict(stage='evaluating',seed=seed,arm=arm,completed=len(rows),target=20))
            results[arm]=rows;del model
        write(OUT/f'seed{seed}'/'comparison.json',compare(results))
    write(OUT/'status.json',dict(stage='complete',models=4,cases=20))

if __name__=='__main__':
    ap=argparse.ArgumentParser();group=ap.add_mutually_exclusive_group()
    group.add_argument('--dry-run',action='store_true');group.add_argument('--execute-api',action='store_true')
    args=ap.parse_args();protocol=prepare()
    print('Offline validation passed: four final100 models; API=0, GPU=0',flush=True)
    if args.execute_api: execute(protocol)
