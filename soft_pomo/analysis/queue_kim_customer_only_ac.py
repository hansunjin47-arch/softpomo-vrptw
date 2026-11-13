"""Generate isolated customer-only Kim caches, then wait for the Pure40 queue."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import run_phase2_customer_only_ac as customer
from travel_time_prompt_contract import ROOT, prompt as previous_prompt

sys.path.insert(0,str(ROOT))
import VRPTWLLMModule as backend
sys.path.insert(0,str(ROOT/'server'))
import train_cached as io

VERSION='kim_customer_only_ac_v1'
QUEUE=ROOT/'result/kim_customer_only_ac_queue_20261012'


def location(variant):
    return ROOT/f'result/rc101_kim_train20_config{variant}_{VERSION}'


def prompt(variant):
    text=customer.prompt(variant)
    before=text.split('INTERPRETING LEARNED ROUTES\n')[0]
    after=text.split('\nOUTPUT\n')[1]
    heuristic=previous_prompt(variant,'kim').split('INTERPRETING HEURISTIC GROUPS\n')[1].split('\nOUTPUT')[0]
    return before+'INTERPRETING HEURISTIC GROUPS\n'+heuristic+'\nOUTPUT\n'+after


def immutable(path,value):
    customer.immutable(io,path,value)


def prepare(variant):
    old=ROOT/f'result/rc101_kim_train20_config{variant}_ontology_routes_only_ac_travel_time_v1'
    out=location(variant)
    protocol=io.read(old/'protocol.json')
    text=prompt(variant)
    prompt_path=ROOT/'analysis/prompts'/VERSION/f'kim_config{variant}.txt'
    prompt_path.parent.mkdir(parents=True,exist_ok=True)
    if prompt_path.exists():
        assert prompt_path.read_text(encoding='utf-8')==text
    else:
        prompt_path.write_text(text,encoding='utf-8')
    protocol.update(version=VERSION,prompt_sha256=io.sha(prompt_path),
        decoder='beta*c customers only; depot zero; no negative conversion',
        original_protocol_sha256=io.sha(old/'protocol.json'))
    immutable(out/'protocol.json',protocol)
    original_hashes=io.read(old/'request_hashes.json')
    for name in protocol['train_names']:
        path=old/'requests'/name/'request.json'
        assert io.sha(path)==original_hashes[name]
        request=io.read(path)
        context=json.loads(request['messages'][0]['content'].split('\n')[-1])
        for group in context['fixed_instance']['groups']:
            group['required_output_customers']=[n for n in group['required_output_customers'] if n]
        assert [g['members'] for g in context['fixed_instance']['groups']]==protocol['clusters']
        request['messages']=[dict(role='user',content=text+'\n'+json.dumps(context,separators=(',',':'),allow_nan=False))]
        schema=request['response_format']['json_schema']['schema']['properties']['confidence']
        schema['minItems']=schema['maxItems']=100
        schema['items']['properties']['customer']['minimum']=1
        immutable(out/'requests'/name/'request.json',request)
    immutable(out/'request_hashes.json',{n:io.sha(out/'requests'/n/'request.json') for n in protocol['train_names']})
    print('PREPARED',variant,len(protocol['train_names']),flush=True)


def call(variant):
    out=location(variant)
    protocol=io.read(out/'protocol.json')
    hashes=io.read(out/'request_hashes.json')
    for index,name in enumerate(protocol['train_names']):
        folder=out/'requests'/name
        assert io.sha(folder/'request.json')==hashes[name]
        request=io.read(folder/'request.json')
        response=folder/'response.txt'
        io.write(QUEUE/'status.json',dict(stage='calling',variant=variant,completed=index,total=20))
        if not response.exists():
            os.environ['OPENROUTER_RESPONSE_AUDIT_DIR']=str(folder/'api_metadata')
            print('CALL',variant,index+1,name,flush=True)
            response.write_text(backend.query_llm('',request['model'],messages=request['messages'],
                response_format=request['response_format']),encoding='utf-8')
        # Invalid outputs stop the queue; no resampling or score fabrication.
        cache=customer.decode(response.read_text(encoding='utf-8'),protocol['clusters'])
        immutable(out/'cache'/(name+'_cluster.json'),cache)
        print('VALIDATED',variant,index+1,flush=True)
    immutable(out/'cache_manifest.json',dict(request_hashes_sha256=io.sha(out/'request_hashes.json'),
        files={p.name:io.sha(p) for p in (out/'cache').glob('*.json')}))
    io.write(out/'status.json',dict(stage='cache_complete',completed=20))


def bundle(variant):
    out=location(variant)
    target=out/'training_bundle'
    source=ROOT/f'server_export/kim_tt_ac_train20_20261010/{variant}'
    if (target/'bundle_manifest.json').exists():
        io.verify_bundle(target)
        return
    target.mkdir(parents=True,exist_ok=True)
    for path in source.glob('*.py'):
        shutil.copy2(path,target/path.name)
    shutil.copytree(source/'data',target/'data',dirs_exist_ok=True)
    (target/'server').mkdir(exist_ok=True)
    shutil.copy2(ROOT/'server/train_cached.py',target/'server/train_cached.py')
    config=io.read(source/'configs/kim.json')
    assert config['trainer']['epochs']==1000 and config['trainer']['train_batch_size']==128
    assert config['environment']['dist_penalty_weight']==0 and config['environment']['time_penalty']==1
    assert config['method']=='kim' and 'initial_checkpoint' not in config
    manifest=io.read(out/'cache_manifest.json')
    for name in config['trainer']['train_instances']:
        path=out/'cache'/(name+'_cluster.json')
        assert io.sha(path)==manifest['files'][path.name]
        cache=io.validate_cache(path)
        assert all(v==0 for v in cache['_depot'].values())
        dest=target/'cache/kim'/path.name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest)
    io.write(target/'configs/kim.json',config)
    files=[p for p in target.rglob('*') if p.is_file()]
    io.write(target/'bundle_manifest.json',{'sha256':{p.relative_to(target).as_posix():io.sha(p) for p in files}})


def main(stage):
    for variant in ('A','C'):
        prepare(variant)
    if stage=='prepare':
        return
    assert backend.OPENROUTER_API_KEY
    for variant in ('A','C'):
        call(variant)
        bundle(variant)
    dependency=ROOT/'result/pure_tt40_e2000_queue_20261012/status.json'
    io.write(QUEUE/'status.json',dict(stage='waiting_for_pure40',A_cache=20,C_cache=20))
    while True:
        state=io.read(dependency)
        if state['stage']=='complete':
            break
        if state['stage']=='failed':
            raise RuntimeError(f'Pure40 dependency failed: {state}')
        time.sleep(60)
    for variant in ('A','C'):
        out=location(variant)
        target=out/'training_bundle'
        runner=target/'server/train_cached.py'
        io.write(QUEUE/'status.json',dict(stage='training',variant=variant,target=1000))
        if not (out/'training/checkpoint-last.pt').exists() and not (out/'smoke/smoke-checkpoint.pt').exists():
            subprocess.run([sys.executable,'-u',str(runner),'--method','kim','--smoke',
                '--smoke-batch-size','128','--out',str(out/'smoke')],cwd=ROOT,check=True)
        while True:
            command=[sys.executable,'-u',str(runner),'--method','kim','--epochs','1000','--seed','42',
                '--save-every','500','--max-hours','11','--out',str(out/'training')]
            if (out/'training/checkpoint-last.pt').exists():
                command.append('--resume')
            subprocess.run(command,cwd=ROOT,check=True)
            state=io.read(out/'training/status.json')
            if state['status']=='complete':
                break
            if state['status']!='needs_resume':
                raise RuntimeError(state)
    io.write(QUEUE/'status.json',dict(stage='complete',epochs=1000,evaluation=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('prepare','queue'))
    parser.add_argument('--allow-api',action='store_true')
    args=parser.parse_args()
    if args.stage=='queue' and not args.allow_api:
        parser.error('Paid calls require --allow-api')
    try:
        main(args.stage)
    except Exception as error:
        io.write(QUEUE/'status.json',dict(stage='failed',error=repr(error)))
        raise
