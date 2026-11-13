"""GPT late10 Phase2 pipeline, plus a separate Kim train20 cache-only experiment."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'result/bks_normalized_late10_train20_20261013/rc101'
OUT = ROOT / 'result/rc101_phase2_train20_bks_late10_gpt54_v1'
KIM = ROOT / 'result/rc101_kim_train20_bks_late10_gpt54_v1'
LATE_WEIGHT = 10
MODEL = 'openai/gpt-5.4'
TRAIN_SIZE = 20
OBSERVATION_ROOT = None
OLDKIM = ROOT / 'result/rc101_kim_train20_bks_customer_only_gpt54_v1'
sys.path.insert(0, str(ROOT / 'server'))
import train_cached as io
from run_phase2_customer_only_ac import prompt as prior_prompt, decode
from phase2_routes_only import payload_for, validate_payload

OBJECTIVE = '''Minimize J=K/14+(T+10*Lt)/1696.95 for fully served solutions.
K is the number of nonempty vehicle routes. T is actual event-dependent driving time in minutes,
including depot departure and return legs; it excludes waiting and service durations.
Lt is total customer AND depot lateness in minutes. Late-stop count is not separately penalized.
The references are RC101 best-known 14 vehicles and 1696.95 distance units; at base speed 1
the latter is the reference no-event driving time in minutes.
One lateness minute costs ten driving minutes. One vehicle costs 121.210714 driving minutes,
or 12.121071 lateness minutes with other terms fixed. These are weighted trade-offs,
NOT strict lexicographic priorities. Geometric distance is NOT a separate objective term.
The complete cost also includes 500*(unserved customers/100); full service is required.
Do not claim a measured improvement without an evaluated alternative; use only supplied evidence.
'''


def immutable(path, value):
    if path.exists():
        assert io.read(path) == value, str(path)
    else:
        io.write(path, value)


def prompt():
    text = prior_prompt('A')
    return text[:text.index('Minimize J_A=')] + OBJECTIVE + '\n' + text[text.index('Capacity is hard;'):]


def section(test=False):
    return OUT / 'test80' if test else OUT


def template():
    protocol = io.read(OLDKIM / 'protocol.json')
    return io.read(OLDKIM / 'requests' / protocol['train_names'][0] / 'request.json')


def prepare(test=False, execution_feedback_enabled=False):
    if execution_feedback_enabled and test:
        raise ValueError('Execution feedback is train-only')
    if execution_feedback_enabled and OUT.exists():
        raise ValueError('Feedback preparation requires a new output directory')
    from seq_execution_feedback import execution_feedback, feedback_prompt, attach
    request_prompt = feedback_prompt(prompt()) if execution_feedback_enabled else prompt()
    sys.path.insert(0, str(BASE))
    import torch
    import train_vrptw as pure
    digest = io.verify_bundle(BASE)
    config = io.read(BASE / 'configs/pure.json')
    assert config['environment']['late_penalty'] == LATE_WEIGHT
    assert config['environment']['TT_max'] == config['environment']['Lt_max'] == 1696.95/240
    assert config['environment']['vehicle_penalty'] == 100/14
    ckpt = BASE / 'runs/pure/checkpoint-500.pt'
    ckpt_hash = io.sha(ckpt)
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    assert state['epoch'] == 500
    assert state['server_contract'] == dict(bundle_sha256=digest, config=config)
    split = io.read(BASE / 'split.json')
    entries = split['test' if test else 'train']
    names = [r['name'] for r in entries]
    assert len(names) == (80 if test else TRAIN_SIZE)
    out = section(test)
    immutable(out / 'protocol.json', dict(version=OUT.name, names=names, test=test,
        checkpoint_sha256=ckpt_hash, model='openai/gpt-5.4', prompt=request_prompt,
        objective=f'K/14+(T+{LATE_WEIGHT}*L)/1696.95', feedback=execution_feedback_enabled,
        reference_selection=('Pure500 best-of-800; train execution metrics included' if execution_feedback_enabled
                             else 'Pure500 best-of-800, existing evaluation protocol; realized reward not sent to LLM'),
        decoder='beta5 active-group customer-only; zero depot; no negative conversion',
        temperature='omitted as in approved GPT pilot'))
    extract = copy.deepcopy(config)
    extract['trainer'].update(train_instances=names, use_augmentation=True, n_mc_samples=1)
    if execution_feedback_enabled:
        extract['trainer'].update(pomo_auto=False, eval_top_k=None)
        extract['environment']['pomo_size'] = 100
        extract['model']['eval_type'] = 'argmax'
    trainer = io.build_trainer(extract, BASE, out / 'reference_runtime')
    trainer.model.load_state_dict(state['model_state_dict'], strict=True)
    trainer.model.eval()
    assert len(trainer.instance_pool) == len(names)
    obsroot = OBSERVATION_ROOT or ROOT / 'server_export/soft_pomo_seed_sweep/train20' / ('eval_addon/data' if test else 'data')
    for index, inst in enumerate(trainer.instance_pool):
        name = names[index]
        assert inst['name'] == name
        folder = out / 'requests' / name
        assert io.sha(BASE / 'data' / (name+'.txt')) == entries[index]['sha256']
        assert io.sha(obsroot / (name+'.txt')) == entries[index]['sha256']
        if (folder / 'reference.json').exists():
            ref = io.read(folder / 'reference.json')
            assert ref['checkpoint_sha256'] == ckpt_hash
        else:
            pure._set_seed(42+index)
            with torch.no_grad():
                routes, selected_reward = trainer._best_solution(inst)
            clusters = [[int(n) for n in route if n] for route in routes if any(route)]
            assert sorted(n for group in clusters for n in group) == list(range(1,101))
            ref = dict(clusters=clusters, checkpoint_sha256=ckpt_hash)
            if execution_feedback_enabled:
                ref['execution_feedback'] = execution_feedback(inst, routes, selected_reward,
                    late_weight=LATE_WEIGHT, checkpoint_sha256=ckpt_hash, training=True)
            immutable(folder / 'reference.json', ref)
        context = payload_for(inst, ref['clusters'], io.read(obsroot / (name+'.observations.json')))
        for group in context['fixed_instance']['groups']:
            group['required_output_customers'] = [n for n in group['required_output_customers'] if n]
        validate_payload(context)
        if execution_feedback_enabled:
            context = attach(context, ref['execution_feedback'], training=True)
        request = template()
        request['messages'] = [dict(role='user', content=request_prompt+'\n'+json.dumps(context,separators=(',',':'),allow_nan=False))]
        assert 'temperature' not in request and request['model'] == 'openai/gpt-5.4'
        immutable(folder / 'context.json', context)
        immutable(folder / 'request.json', request)
        io.write(out / 'status.json', dict(stage='preparing', completed=index+1, total=len(names)))
        print('PREPARED', index+1, name, flush=True)
    immutable(out / 'request_hashes.json', {n: io.sha(out/'requests'/n/'request.json') for n in names})


def call(out, kim=False):
    from compare_llm_single_customer_only import call_once
    protocol = io.read(out / 'protocol.json')
    names = protocol['names']
    hashes = io.read(out / 'request_hashes.json')
    for index, name in enumerate(names):
        folder = out / 'requests' / name
        assert io.sha(folder / 'request.json') == hashes[name]
        io.write(out / 'status.json', dict(stage='calling', completed=index, total=len(names), scenario=name))
        response = folder / 'response.txt'
        if not response.exists():
            if (folder / 'attempted.json').exists():
                raise RuntimeError('Previous paid attempt needs inspection: '+name)
            immutable(folder / 'attempted.json', dict(model=protocol['model']))
            print('CALL', out.name, index+1, name, flush=True)
            response.write_text(call_once(io.read(folder/'request.json'), folder), encoding='utf-8')
        clusters = protocol['clusters'] if kim else io.read(folder / 'reference.json')['clusters']
        cache = decode(response.read_text(encoding='utf-8'), clusters)
        immutable(out/'cache'/(name+('_cluster.json' if kim else '_rl_cluster.json')), cache)
        print('VALIDATED', out.name, index+1, flush=True)
    immutable(out/'cache_manifest.json', dict(request_hashes_sha256=io.sha(out/'request_hashes.json'),
        files={p.name:io.sha(p) for p in (out/'cache').glob('*.json')}))
    io.write(out/'status.json', dict(stage='cache_complete', completed=len(names)))


def kim_call():
    old = io.read(OLDKIM/'protocol.json')
    names = old['train_names']
    immutable(KIM/'protocol.json', dict(version=KIM.name,names=names,clusters=old['clusters'],
        model='openai/gpt-5.4',objective='K/14+(T+10*L)/1696.95',training_started=False))
    hashes = io.read(OLDKIM/'request_hashes.json')
    for name in names:
        path = OLDKIM/'requests'/name/'request.json'
        assert io.sha(path) == hashes[name]
        request = io.read(path)
        text = request['messages'][0]['content']
        request['messages'][0]['content'] = text[:text.index('Minimize J=')] + OBJECTIVE+'\n'+text[text.index('Capacity is hard;'):]
        immutable(KIM/'requests'/name/'request.json',request)
    immutable(KIM/'request_hashes.json',{n:io.sha(KIM/'requests'/n/'request.json') for n in names})
    call(KIM, kim=True)


def bundle():
    target = OUT/'bundle'
    if (target/'bundle_manifest.json').exists():
        io.verify_bundle(target)
        return
    (target/'server').mkdir(parents=True,exist_ok=True)
    for path in BASE.glob('*.py'):
        shutil.copy2(path,target/path.name)
    shutil.copytree(BASE/'data',target/'data',dirs_exist_ok=True)
    shutil.copy2(BASE/'server/train_cached.py',target/'server/train_cached.py')
    manifest = io.read(OUT/'cache_manifest.json')
    for name,digest in manifest['files'].items():
        path = OUT/'cache'/name
        assert io.sha(path)==digest
        io.validate_cache(path)
    assert len(manifest['files'])==TRAIN_SIZE
    shutil.copytree(OUT/'cache',target/'cache',dirs_exist_ok=True)
    (target/'checkpoints').mkdir(exist_ok=True)
    shutil.copy2(BASE/'runs/pure/checkpoint-500.pt',target/'checkpoints/pure500.pt')
    config = io.read(BASE/'configs/pure.json')
    llm = io.read(ROOT/'result/rc101_relation_train20_cache_v1/settings.json')['llm']
    llm.update(model=MODEL,reward_config=f'BKS_LATE{LATE_WEIGHT}',bias_strength=5.0,use_ontology=True)
    config.update(method='phase2',cache_dir='cache',initial_checkpoint='checkpoints/pure500.pt',llm=llm)
    config['trainer'].update(cluster_mode='rl',bias_mode='cluster',start_mode='random',pomo_size=100,
        benchmark='rc1',experience_refresh_epoch=-1,rl_recluster_epoch=None,require_reference_start=False)
    immutable(target/'configs/phase2.json',config)
    immutable(target/'split.json',io.read(BASE/'split.json'))
    immutable(target/'provenance.json',dict(protocol=io.read(OUT/'protocol.json'),
        branch='Existing Phase2 protocol: model and optimizer transfer; RNG reset to seed42, not exact resume'))
    immutable(target/'bundle_manifest.json',dict(sha256={p.relative_to(target).as_posix():io.sha(p) for p in target.rglob('*') if p.is_file()}))


def train():
    bundle()
    target = OUT/'bundle'
    runner = target/'server/train_cached.py'
    smoke = OUT/'smoke'
    if not (smoke/'smoke-checkpoint.pt').exists():
        subprocess.run([sys.executable,'-u',str(runner),'--method','phase2','--smoke',
            '--smoke-batch-size','1','--out',str(smoke)],cwd=target,check=True)
    command = [sys.executable,'-u',str(runner),'--method','phase2','--max-hours','168',
        '--save-every','100','--out',str(OUT/'training')]
    if (OUT/'training/checkpoint-last.pt').exists():
        command.append('--resume')
    subprocess.run(command,cwd=target,check=True)
    assert io.read(OUT/'training/status.json')['status']=='complete'


def evaluate():
    target = OUT/'bundle'
    sys.path.insert(0,str(target))
    import torch
    import train_vrptw as pure
    from vrptw_env import load_solomon, _solution_stats
    digest = io.verify_bundle(target)
    saved = io.read(OUT/'training/settings.json')
    assert saved['bundle_sha256']==digest
    checkpoint = OUT/'training/checkpoint-1000.pt'
    state = torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert state['epoch']==1000 and state['server_contract']==saved
    test = section(True)
    manifest = io.read(test/'cache_manifest.json')
    assert manifest['request_hashes_sha256']==io.sha(test/'request_hashes.json')
    for name,h in manifest['files'].items():
        assert io.sha(test/'cache'/name)==h
        io.validate_cache(test/'cache'/name)
    assert len(manifest['files'])==80
    config = copy.deepcopy(saved['config'])
    config['cache_dir']=str(test/'cache')
    entries = io.read(target/'split.json')['test']
    test_names = [entry['name'] for entry in entries]
    assert len(set(test_names)) == 80
    assert not set(test_names) & set(saved['config']['trainer']['train_instances'])
    suffix = '_cluster.json' if saved['config']['method'] == 'kim' else '_rl_cluster.json'
    assert set(manifest['files']) == {name+suffix for name in test_names}
    # Trainer initialization eagerly loads caches for its instance pool.
    config['trainer'].update(train_instances=test_names,use_augmentation=True,n_mc_samples=1)
    config['model']['eval_type']='argmax'
    out = OUT/'eval_1000_test80_gpu'
    immutable(out/'contract_v2.json',dict(checkpoint_sha256=io.sha(checkpoint),config=config,
        cache_manifest_sha256=io.sha(test/'cache_manifest.json'),device='cuda',selection='best-of-800'))
    trainer = io.build_trainer(config,target,out/'runtime')
    trainer.model.load_state_dict(state['model_state_dict'],strict=True)
    trainer.model.eval()
    results = io.read(out/'results.json') if (out/'results.json').exists() else []
    assert [r['name'] for r in results]==[r['name'] for r in entries[:len(results)]]
    for index,entry in enumerate(entries):
        if index<len(results):
            continue
        path = target/'data'/(entry['name']+'.txt')
        assert io.sha(path)==entry['sha256']
        pure._set_seed(42+index)
        inst=load_solomon(str(path))
        with torch.no_grad():
            routes,reward=trainer._best_solution(inst)
        routes=[[int(n) for n in route] for route in routes]
        assert sorted(n for route in routes for n in route if n)==list(range(1,101))
        stats=_solution_stats(inst,routes)
        row=dict(name=entry['name'],kind=entry['kind'],routes=routes,vehicles=len(stats['per_vehicle']),
            travel_time=float(stats['total_travel_time']),late_count=int(stats['late_count']),
            total_late=float(stats['total_late']),served=int(stats['n_served']),reward=float(reward))
        row['cost']=row['vehicles']/14+(row['travel_time']+LATE_WEIGHT*row['total_late'])/1696.95
        assert abs(row['cost']+row['reward'])<1e-4
        results.append(row)
        io.write(out/'results.json',results)
        io.write(out/'status.json',dict(stage='evaluating',completed=len(results),total=80))
    keys=('vehicles','travel_time','late_count','total_late','served','reward','cost')
    summary={}
    for kind in ('ALL','RAIN','ACCIDENT'):
        rows=[r for r in results if kind=='ALL' or r['kind']==kind]
        summary[kind]=dict(count=len(rows),mean={k:statistics.mean(r[k] for r in rows) for k in keys},
            std={k:statistics.stdev(r[k] for r in rows) for k in keys})
    io.write(out/'summary.json',summary)
    io.write(out/'status.json',dict(stage='complete',completed=80))
    io.write(OUT/'pipeline.json',dict(stage='complete',method=saved['config']['method']))


def pipeline():
    for stage in ('prepare','call','kim-call','train','prepare-test','call-test','evaluate'):
        io.write(OUT/'pipeline.json',dict(stage=stage))
        command=[sys.executable,'-u',str(Path(__file__).resolve()),stage]
        if stage in ('call','kim-call','call-test'):
            command.append('--allow-api')
        subprocess.run(command,cwd=ROOT,check=True)
    io.write(OUT/'pipeline.json',dict(stage='complete',kim_training_started=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','call','kim-call','train','prepare-test','call-test','evaluate','pipeline'])
    parser.add_argument('--allow-api',action='store_true')
    parser.add_argument('--train-feedback', action='store_true')
    parser.add_argument('--out', type=Path)
    args=parser.parse_args()
    if args.train_feedback and (args.stage != 'prepare' or args.out is None):
        parser.error('--train-feedback requires prepare and a new --out directory')
    if args.out is not None:
        OUT = args.out.resolve()
    if args.stage in ('pipeline','call','kim-call','call-test') and not args.allow_api:
        parser.error('Paid calls require --allow-api')
    try:
        if args.stage=='pipeline': pipeline()
        elif args.stage.startswith('prepare'): prepare(test=args.stage=='prepare-test', execution_feedback_enabled=args.train_feedback)
        elif args.stage=='kim-call': kim_call()
        elif args.stage.startswith('call'): call(section(args.stage=='call-test'))
        elif args.stage=='train': train()
        else: evaluate()
    except Exception as error:
        io.write(OUT/'pipeline.json',dict(stage='failed',failed_stage=args.stage,error=repr(error)))
        raise
