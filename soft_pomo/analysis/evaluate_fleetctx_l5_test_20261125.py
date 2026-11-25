"""Causal Test80 for one completed NEW-head Kim seed, queued after training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch
from analysis import prepare_rl_lkh_l5_20261122 as prompt_module
enrich=prompt_module.enrich
from experiments.concurrent_fleet_pair_kim.run import OnlineEnv
from experiments.concurrent_fleet_pair_kim.fleet_context import FleetContextKimPolicy as GuidedPolicy
from experiments.concurrent_fleet_v9_guided import online
from experiments.concurrent_fleet_v9_guided.prompts import parse_test_response
from experiments.concurrent_fleet_bias_guided.llm_retry import RetryingClient
from experiments.concurrent_fleet_bias_guided.support import ROOT, read, load
from experiments.concurrent_fleet_pair.io import write
from experiments.concurrent_fleet_v3 import run as data_module

NAME='rc_kim_rl_lkh_cap_decay_v100_l5_s44_46_20261122'
BUNDLE=ROOT/'server_export'/NAME
RECEIVED=ROOT/'server_export'/(NAME+'_received')
prompt_module.ART=BUNDLE/'artifact'
RUN=ROOT/'result/rc_kim_fleetctx_v100_l5_s46_20261124/run'
SEED=46
OUT=ROOT/'result/rc_maxconf_v100_l5_s43_test_20261115'


def validate():
    cfg=read(RUN/'protocol.json')['configs']['5']
    assert read(RUN/'late5/training_complete.json')['epoch']==1000
    assert cfg['trainer']['seed']==46
    assert cfg['environment']['require_dispatch_capacity']
    assert cfg['environment']['late_penalty']==5
    assert read(RUN/'split.json')==read(BUNDLE/'dataset/split.json')
    saved=torch.load(RUN/'late5/checkpoint-1000.pt',map_location='cpu',weights_only=False)
    assert saved['epoch']==1000
    GuidedPolicy(**cfg['model']).load_state_dict(saved['model_state_dict'],strict=True)
    for row in read(RUN/'split.json')['test']:
        assert hashlib.sha256((BUNDLE/'dataset/data'/(row['name']+'.txt')).read_bytes()).hexdigest()==row['sha256']
    return {'46':hashlib.sha256((RUN/'late5/checkpoint-1000.pt').read_bytes()).hexdigest()}


def main(check_only=False):
    torch.set_num_threads(1)
    hashes=validate()
    print('Validated requested epoch1000 checkpoints, source and test data',flush=True)
    if check_only:return
    assert os.environ.get('OPENROUTER_API_KEY'), 'Missing API key'
    assert torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(0.12)
    OUT.mkdir(exist_ok=True)
    protocol=dict(arm='fleet_context_lr_decay',dispatch_capacity_mask=True,checkpoints=hashes,seeds=list([SEED]),late=5,vehicle_weight=100,mask=True,
        decode='single greedy; AMP; no augmentation',test_count=80,
        guidance='NEW per-group max-confidence beta5 during dispatch; legacy routing beta5; exact initial cache and causal live ACC refresh',
        prompt='max_confidence_new_head_v1_plus_rl_lkh; independent paired example; target lambda5',
        cache_generation_sha256=hashlib.sha256((BUNDLE/'artifact/generation.json').read_bytes()).hexdigest(),
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==protocol
    else:write(OUT/'protocol.json',protocol)
    data_module.SOURCE=BUNDLE/'dataset'
    art=BUNDLE/'artifact'
    groups,template=read(art/'groups.json'),read(art/'request_template.json')
    base,example=read(art/'base_confidence.json'),read(art/'reference.json')
    original=online.build_test_request
    initial=True
    def request(*args,**kwargs):
        nonlocal initial
        initial=kwargs.get('env') is None
        req=enrich(original(*args,**kwargs),base,example,new_head_beta=5.)
        return req
    online.build_test_request=request
    online.OnlineTestEnv=OnlineEnv
    def validate_payload(req,payload):
        parse_test_response(payload,groups,json.loads(req['messages'][1]['content'])['selectable_customers'])
    shared_client=RetryingClient(ROOT/'result/rl_lkh_l5_shared_test_20261123/calls',validate_payload,allow_paid=True)
    for seed in [SEED]:
        run=RUN
        cfg=read(run/'protocol.json')['configs']['5']
        model=GuidedPolicy(**cfg['model']).cuda().eval()
        model.load_state_dict(torch.load(run/'late5/checkpoint-1000.pt',map_location='cpu',weights_only=False)['model_state_dict'])
        live_client=RetryingClient(OUT/f'calls/seed{seed}',validate_payload,allow_paid=True)
        def client(req):return shared_client(req) if initial else live_client(req)
        results=[]
        for row in read(run/'split.json')['test']:
            folder=OUT/f'seed{seed}/instances'/row['name']
            write(OUT/'status.json',dict(pid=os.getpid(),stage='evaluating',seed=seed,completed=len(results),target=80,name=row['name']))
            if (folder/'result.json').exists():result=read(folder/'result.json')
            else:
                inst,reports=load(row)
                result=online.evaluate_instance(model,inst,cfg['environment'],groups,reports,template=template,
                    client=client,beta=5.,out=folder,amp=True)
            results.append(dict(name=row['name'],**result['metrics']))
            write(OUT/f'seed{seed}/results.json',results)
            print('TEST',seed,len(results),row['name'],result['metrics']['reward'],flush=True)
        write(OUT/f'seed{seed}/complete.json',dict(epoch=1000,test_count=80))
        del model
    write(OUT/'status.json',dict(stage='complete',seeds=list([SEED]),test_count_per_seed=80))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--seed',type=int,choices=(46,),required=True)
    args=parser.parse_args()
    SEED=args.seed
    OUT=ROOT/f'result/fleetctx_l5_s{SEED}_test_20261125'
    try:main(args.validate_only)
    except Exception as error:
        if not args.validate_only:write(OUT/'status.json',dict(stage='failed',error=repr(error)))
        raise
