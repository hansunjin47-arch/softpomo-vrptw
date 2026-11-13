"""Generate R101 masked pair Kim training caches without changing running code."""
import copy
import hashlib
import json
import tarfile
import torch
from analysis import pair_kim_rl_example as shared
from experiments.concurrent_fleet_v3 import run as data_module
from experiments.concurrent_fleet_guided.prompts import PUBLIC_FIELDS

ROOT = shared.ROOT
SOURCE = ROOT/'server_export/pair_returnmask_cr_v100_200_l5_s42_46_20261031/datasets/r101'
PURE = ROOT/'server_export/pair_mask_completed_received_20261106_123111/pair_returnmask_cr_v100_200_l5_s42_46_20261031/runs/r101/vehicle100/seed42'
ART = ROOT/'result/r101_pair100_l5_mask_kim_cache_20261106'


def main():
    torch.set_num_threads(1)
    ART.mkdir(exist_ok=True)
    data_module.SOURCE = SOURCE
    shared.adapter.train.SOURCE = SOURCE
    cfg = shared.read(PURE/'protocol.json')['configs']['5']
    assert cfg['environment']['protect_last_service_vehicle']
    assert cfg['environment']['normalize_advantage_std']
    assert cfg['trainer']['seed'] == 42
    assert abs(cfg['environment']['vehicle_penalty']*1650.8/100-100)<1e-6
    split = shared.read(SOURCE/'split.json')
    source = ROOT/'result/kim_l5_gpt_rc_r_c_20261026/r101/train/requests/R101_RAIN_P018'
    context = shared.read(source/'context.json')
    groups = [g['members'] for g in context['fixed_instance']['groups']]
    assert sorted(n for g in groups for n in g)==list(range(1,101))
    template = shared.read(source/'request.json')
    for name, value in [('groups.json',groups),('split.json',split),('request_template.json',template),('config.json',cfg)]:
        if (ART/name).exists():
            assert shared.read(ART/name)==value
        else:
            shared.write(ART/name,value)
    shared.PURE, shared.OLD, shared.ART, shared.LATE = PURE, ART, ART, 5
    shared.EXAMPLE = 'R101_RAIN_REFERENCE_PAIR_20261106'
    example = shared.read(ART/'reference.json') if (ART/'reference.json').exists() else shared.reference()
    print('REFERENCE',example['observed_performance'],flush=True)
    params = cfg['environment']
    client = shared.RetryingClient(ART/'calls',lambda req,payload: shared.parse_test_response(
        payload,groups,json.loads(req['messages'][1]['content'])['selectable_customers']),allow_paid=True)
    inst,_ = shared.load(split['train'][0])
    inst = copy.deepcopy(inst)
    inst['name']='R101_BASE_REFERENCE'
    inst['preset_events']=[]
    base_path = ART/'base_confidence.json'
    if base_path.exists():
        base=shared.read(base_path)
    else:
        request=shared.enrich(shared.build_test_request(inst,params,groups,[],template=template),None,example)
        base=shared.as_list(client(request),groups)
        shared.write(base_path,base)
    for i,row in enumerate(split['train']):
        path=ART/'train_cache'/(row['name']+'.json')
        if not path.exists():
            inst,reports=shared.load(row)
            request=shared.build_test_request(inst,params,groups,reports,template=template)
            ctx=json.loads(request['messages'][1]['content'])
            ctx['public_events']=[{k:v for k,v in r.items() if k in PUBLIC_FIELDS} for r in reports]
            request['messages'][1]['content']=json.dumps(ctx)
            event=shared.as_list(client(shared.enrich(request,base,example,offline=True)),groups)
            accident=reports[0]['event_type']=='ACCIDENT'
            shared.write(path,dict(initial=base if accident else event,post_disclosure=event if accident else None,
                                   revealed_at_min=reports[0]['revealed_at_min']))
        shared.write(ART/'status.json',dict(stage='generating',completed=i+1,target=20))
        print('CACHE',i+1,row['name'],flush=True)
    shared.write(ART/'reuse_audit.json',dict(rl_reference=False,separate_rl_example=True))
    shared.adapter.ARTIFACT=ART
    shared.adapter.audit()
    shared.write(ART/'complete.json',dict(train_count=20,base_count=1,test='not generated; causal online test required'))
    files=[p for p in ART.rglob('*.json') if 'calls' not in p.parts]
    shared.write(ART/'manifest.json',{p.relative_to(ART).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    archive=ROOT/'server_export/r101_pair100_l5_mask_kim_cache_20261106.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in files+[ART/'manifest.json']:
            tar.add(p,arcname=ART.name+'/'+p.relative_to(ART).as_posix())
    archive.with_suffix(archive.suffix+'.sha256').write_text(hashlib.sha256(archive.read_bytes()).hexdigest()+'  '+archive.name+'\n',encoding='ascii')
    shared.write(ART/'status.json',dict(stage='complete',train_count=20,archive=str(archive)))
    print('COMPLETE',archive,flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        shared.write(ART/'status.json',dict(stage='failed',error=repr(error)))
        raise
