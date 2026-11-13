"""Causal local test80 for two downloaded server Kim checkpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch
from analysis.pair_kim_rl_example import enrich
from experiments.concurrent_fleet_pair_kim.run import GuidedPolicy, OnlineEnv
from experiments.concurrent_fleet_v9_guided import online
from experiments.concurrent_fleet_v9_guided.prompts import parse_test_response
from experiments.concurrent_fleet_bias_guided.llm_retry import RetryingClient
from experiments.concurrent_fleet_bias_guided.support import ROOT, read, load
from experiments.concurrent_fleet_pair.io import write
from experiments.concurrent_fleet_v3 import run as data_module

NAME='rc101_pair100_200_l5_mask_kim_s42_46_20261111'
BUNDLE=ROOT/'server_export'/NAME
RECEIVED=ROOT/'server_export/maintenance_snapshot_080cy48t'
OUT=ROOT/'result/rc200_kim_test_20261113'


def validate():
    manifest=read(BUNDLE/'manifest.json')['sha256']
    # The server trainer alone has an intentional training-only patch.
    for relative,digest in manifest.items():
        if relative.startswith('experiments/') and relative.endswith('.py') and relative!='experiments/concurrent_fleet_pair/train.py':
            assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest, relative+' changed'
    hashes={}
    for seed in range(42,43):
        run=RECEIVED/f'runs/vehicle200/seed{seed}'
        assert read(run/'late5/training_complete.json')['epoch']==1000
        cfg=read(run/'protocol.json')['configs']['5']
        assert cfg['trainer']['seed']==seed and cfg['environment']['protect_last_service_vehicle']
        assert cfg['environment']['late_penalty']==5
        assert abs(cfg['environment']['vehicle_penalty']-200*100/1696.95)<1e-8
        checkpoint=run/'late5/checkpoint-1000.pt'
        hashes[str(seed)]=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert saved['epoch']==1000
        model=GuidedPolicy(**cfg['model'])
        model.load_state_dict(saved['model_state_dict'],strict=True)
        split=read(run/'split.json')
        assert split==read(BUNDLE/'dataset/split.json') and len(split['test'])==80
        for row in split['test']:
            assert hashlib.sha256((BUNDLE/'dataset/data'/(row['name']+'.txt')).read_bytes()).hexdigest()==row['sha256']
    return hashes


def main(check_only=False):
    torch.set_num_threads(1)
    hashes=validate()
    print('Validated requested epoch1000 checkpoints, source and test data',flush=True)
    if check_only:return
    assert os.environ.get('OPENROUTER_API_KEY'), 'Missing API key'
    assert torch.cuda.is_available()
    OUT.mkdir(exist_ok=True)
    protocol=dict(checkpoints=hashes,seeds=list(range(42,43)),late=5,vehicle_weight=200,mask=True,
        decode='single greedy; AMP; no augmentation',test_count=80,
        guidance='shared exact initial requests; per-seed causal live ACC refresh; beta5',
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==protocol
    else:write(OUT/'protocol.json',protocol)
    data_module.SOURCE=BUNDLE/'dataset'
    art=BUNDLE/'artifacts/vehicle200'
    groups,template=read(art/'groups.json'),read(art/'request_template.json')
    base,example=read(art/'base_confidence.json'),read(art/'reference.json')
    original=online.build_test_request
    initial=True
    def request(*args,**kwargs):
        nonlocal initial
        initial=kwargs.get('env') is None
        return enrich(original(*args,**kwargs),base,example)
    online.build_test_request=request
    online.OnlineTestEnv=OnlineEnv
    def validate_payload(req,payload):
        parse_test_response(payload,groups,json.loads(req['messages'][1]['content'])['selectable_customers'])
    shared_client=RetryingClient(OUT/'calls/shared_initial',validate_payload,allow_paid=True)
    for seed in range(42,43):
        run=RECEIVED/f'runs/vehicle200/seed{seed}'
        cfg=read(run/'protocol.json')['configs']['5']
        model=GuidedPolicy(**cfg['model']).cuda().eval()
        model.load_state_dict(torch.load(run/'late5/checkpoint-1000.pt',map_location='cpu',weights_only=False)['model_state_dict'])
        live_client=RetryingClient(OUT/f'calls/seed{seed}',validate_payload,allow_paid=True)
        def client(req):return shared_client(req) if initial else live_client(req)
        results=[]
        for row in read(run/'split.json')['test']:
            folder=OUT/f'seed{seed}/instances'/row['name']
            write(OUT/'status.json',dict(stage='evaluating',seed=seed,completed=len(results),target=80,name=row['name']))
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
    write(OUT/'status.json',dict(stage='complete',seeds=list(range(42,43)),test_count_per_seed=80))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    try:main(args.validate_only)
    except Exception as error:
        if not args.validate_only:write(OUT/'status.json',dict(stage='failed',error=repr(error)))
        raise
