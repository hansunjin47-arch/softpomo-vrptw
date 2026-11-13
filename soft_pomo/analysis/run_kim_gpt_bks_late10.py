"""Fresh Kim late10 training followed by GPT test-cache generation and evaluation."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import run_gpt_bks_late10 as shared
from phase2_routes_only import PUBLIC

ROOT, BASE, OUT, io = shared.ROOT, shared.BASE, shared.KIM, shared.io
immutable = shared.immutable
LATE_WEIGHT = 10
MODEL = 'openai/gpt-5.4'


def prepare():
    io.verify_bundle(BASE)
    protocol = io.read(OUT/'protocol.json')
    manifest = io.read(OUT/'cache_manifest.json')
    assert protocol['objective']==f'K/14+(T+{LATE_WEIGHT}*L)/1696.95'
    assert protocol['model'] == MODEL
    assert manifest['request_hashes_sha256']==io.sha(OUT/'request_hashes.json')
    names = protocol['names']
    assert len(names)==20 and names==io.read(BASE/'configs/pure.json')['trainer']['train_instances']
    target = OUT/'bundle'
    if (target/'bundle_manifest.json').exists():
        io.verify_bundle(target)
        return
    (target/'server').mkdir(parents=True,exist_ok=True)
    for path in BASE.glob('*.py'):
        shutil.copy2(path,target/path.name)
    shutil.copytree(BASE/'data',target/'data',dirs_exist_ok=True)
    shutil.copy2(BASE/'server/train_cached.py',target/'server/train_cached.py')
    for name in names:
        path=OUT/'cache'/(name+'_cluster.json')
        assert io.sha(path)==manifest['files'][path.name]
        cache=io.validate_cache(path)
        assert all(v==0 for v in cache['_depot'].values())
        immutable(target/'cache/kim'/path.name,cache)
    config=io.read(ROOT/'server_export/kim_tt_ac_train20_20261010/A/configs/kim.json')
    pure=io.read(BASE/'configs/pure.json')
    assert pure['environment']['late_penalty'] == LATE_WEIGHT
    config['environment']=copy.deepcopy(pure['environment'])
    config['model']=copy.deepcopy(pure['model'])
    config['optimizer']=copy.deepcopy(pure['optimizer'])
    config['trainer'].update(train_instances=names,epochs=1000,train_batch_size=128,seed=42)
    config['llm'].update(model=MODEL,reward_config=f'BKS_LATE{LATE_WEIGHT}',bias_strength=5.0)
    assert 'initial_checkpoint' not in config and config['method']=='kim'
    immutable(target/'configs/kim.json',config)
    immutable(target/'split.json',io.read(BASE/'split.json'))
    immutable(target/'provenance.json',dict(protocol=protocol,fresh_training=True,
        training_cache_manifest_sha256=io.sha(OUT/'cache_manifest.json')))
    immutable(target/'bundle_manifest.json',dict(sha256={p.relative_to(target).as_posix():io.sha(p) for p in target.rglob('*') if p.is_file()}))
    io.verify_bundle(target)


def train():
    prepare()
    target=OUT/'bundle'
    runner=target/'server/train_cached.py'
    if not (OUT/'smoke/smoke-checkpoint.pt').exists():
        subprocess.run([sys.executable,'-u',str(runner),'--method','kim','--smoke','--smoke-batch-size','1',
            '--out',str(OUT/'smoke')],cwd=target,check=True)
    command=[sys.executable,'-u',str(runner),'--method','kim','--max-hours','168','--save-every','100',
        '--out',str(OUT/'training')]
    if (OUT/'training/checkpoint-last.pt').exists(): command.append('--resume')
    subprocess.run(command,cwd=target,check=True)
    assert io.read(OUT/'training/status.json')['status']=='complete'


def test_cache():
    protocol=io.read(OUT/'protocol.json')
    template=io.read(OUT/'requests'/protocol['names'][0]/'request.json')
    prefix,raw=template['messages'][0]['content'].rsplit('\n',1)
    fixed=json.loads(raw)['fixed_instance']
    entries=io.read(BASE/'split.json')['test']
    names=[r['name'] for r in entries]
    assert len(set(names))==80 and not set(names)&set(protocol['names'])
    test=OUT/'test80'
    immutable(test/'protocol.json',dict(names=names,clusters=protocol['clusters'],model=protocol['model'],
        objective=protocol['objective'],feedback=False,checkpoint_independent=True))
    obsroot=ROOT/'server_export/soft_pomo_seed_sweep/train20/eval_addon/data'
    for entry in entries:
        name=entry['name']
        assert io.sha(obsroot/(name+'.txt'))==io.sha(BASE/'data'/(name+'.txt'))==entry['sha256']
        obs=io.read(obsroot/(name+'.observations.json'))
        assert set(obs)<=PUBLIC
        request=copy.deepcopy(template)
        context=dict(scenario=name,fixed_instance=fixed,public_event=obs)
        request['messages'][0]['content']=prefix+'\n'+json.dumps(context,separators=(',',':'),allow_nan=False)
        immutable(test/'requests'/name/'request.json',request)
    immutable(test/'request_hashes.json',{n:io.sha(test/'requests'/n/'request.json') for n in names})
    shared.call(test,kim=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','train','test-cache','evaluate','pipeline'])
    parser.add_argument('--allow-api',action='store_true')
    args=parser.parse_args()
    if args.stage in ('test-cache','pipeline') and not args.allow_api: parser.error('--allow-api required')
    try:
        if args.stage=='pipeline':
            for stage in ('train','test-cache','evaluate'):
                io.write(OUT/'pipeline.json',dict(stage=stage))
                cmd=[sys.executable,'-u',str(Path(__file__).resolve()),stage]
                if stage=='test-cache': cmd.append('--allow-api')
                subprocess.run(cmd,cwd=ROOT,check=True)
        elif args.stage=='prepare': prepare()
        elif args.stage=='train': train()
        elif args.stage=='test-cache': test_cache()
        else:
            shared.OUT=OUT
            shared.evaluate()
    except Exception as error:
        io.write(OUT/'pipeline.json',dict(stage='failed',failed_stage=args.stage,error=repr(error)))
        raise
