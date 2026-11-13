"""Build route-only context without replaying reference or counterfactual schedules."""
import copy
import json

PUBLIC = {'event_type','affected_nodes','regions','start_min','end_min','revealed_at_min',
          'synthetic','rainfall_mm_per_hour','involved_vehicles','vehicles_involved','news_text'}


def validate_payload(payload):
    if set(payload) != {'ontology','fixed_instance','public_event'}:
        raise ValueError('Unexpected feedback in route-only payload')
    if not set(payload['public_event']) <= PUBLIC:
        raise ValueError('Non-public event field')
    fixed = payload['fixed_instance']
    if set(fixed) != {'source','capacity','max_vehicles','customer_columns','customers','groups'}:
        raise ValueError('Unexpected schedule fields')
    if fixed['customer_columns'] != ['id','x','y','demand','open_min','close_min','service_min']:
        raise ValueError('Unexpected customer columns')
    for group in fixed['groups']:
        if set(group) != {'id','type','cluster','route','reference_order','required_output_customers'}:
            raise ValueError('Unexpected route feedback')


def payload_for(inst, clusters, observation):
    import numpy as np
    from prepare_ontology_train20 import instance_context
    coords = np.concatenate([inst['depot_xy'].numpy(),inst['node_xy'].numpy()])*inst['max_coord']
    payload = dict(ontology=dict(version='routes_only_v1',
        entities=['Customer','ReferenceGroup','PublicEvent'],
        relations=['memberOf','precedesInReferenceRoute','affectedByPublicEvent']),
        fixed_instance=instance_context(inst,coords,clusters),
        public_event={k:v for k,v in observation.items() if k in PUBLIC})
    validate_payload(payload)
    return payload


def prepare(variant, test=False, cpu=False):
    from prepare_phase2_tt_train20 import setup
    from travel_time_prompt_contract import ROOT, VERSION, prompt, export
    run, source, train, io = setup(variant)
    export()
    out = train/'eval_1000_test80' if test else train
    split = io.read(run/'split.json')
    names = [r['name'] for r in split['test']] if test else io.read(run/'config.json')['trainer']['train_instances']
    checkpoint = run/'training/checkpoint-500.pt'
    import torch
    state = torch.load(checkpoint,map_location='cpu',weights_only=False)
    config = io.read(run/'config.json')
    assert state['epoch'] == 500 and state['server_contract'] == dict(config=config,manifest_sha256=io.sha(run/'manifest.json'))
    for name,digest in io.read(run/'manifest.json').items():
        assert io.sha(run/name) == digest
    protocol = dict(version=VERSION,variant=variant,checkpoint=str(checkpoint),
        checkpoint_sha256=io.sha(checkpoint),prompt_sha256=io.sha(ROOT/'analysis/prompts'/VERSION/f'phase2_config{variant}.txt'),
        information='RL routes only; no simulated schedule feedback',reference_device='cpu' if cpu else 'cuda',
        reference_selection='Existing best-of-800 simulated reward selection; reward is not sent to LLM')
    if test:
        protocol.update(test_names=names,reference_checkpoint_sha256=io.sha(checkpoint),
            final_checkpoint_sha256=io.sha(train/'training/checkpoint-1000.pt'))
    else:
        protocol.update(train_names=names)
    def immutable(path,value):
        if path.exists():
            assert io.read(path)==value, str(path)
        else:
            io.write(path,value)
    immutable(out/'protocol.json',protocol)
    extract=copy.deepcopy(config)
    extract['trainer'].update(train_instances=names,use_augmentation=True,n_mc_samples=1,pomo_auto=False,eval_top_k=None)
    trainer=io.build_trainer(extract,source,out/'reference_runtime',cpu=cpu)
    trainer.model.load_state_dict(state['model_state_dict'],strict=True)
    trainer.model.eval()
    import train_vrptw as pure
    template=io.read(ROOT/'result/rc101_ontology_train20_v5/requests'/config['trainer']['train_instances'][0]/'request.json')
    obsroot=ROOT/'server_export/soft_pomo_seed_sweep/train20'/('eval_addon/data' if test else 'data')
    for index,inst in enumerate(trainer.instance_pool):
        name=inst['name']
        assert name==names[index]
        folder=out/'requests'/name
        if (folder/'reference.json').exists():
            ref=io.read(folder/'reference.json')
            assert ref['checkpoint_sha256']==io.sha(checkpoint)
        else:
            pure._set_seed(42+index)
            with torch.no_grad():
                routes,_=trainer._best_solution(inst)
            clusters=[[int(n) for n in route if n] for route in routes if any(route)]
            assert sorted(n for c in clusters for n in c)==list(range(1,101))
            ref=dict(clusters=clusters,checkpoint_sha256=io.sha(checkpoint))
            immutable(folder/'reference.json',ref)
        assert io.sha(obsroot/(name+'.txt'))==io.sha(source/'data'/(name+'.txt'))
        payload=payload_for(inst,ref['clusters'],io.read(obsroot/(name+'.observations.json')))
        request=copy.deepcopy(template)
        request['messages']=[dict(role='user',content=prompt(variant)+'\n'+json.dumps(payload,separators=(',', ':'),allow_nan=False))]
        schema=request['response_format']['json_schema']['schema']['properties']['confidence']
        schema['minItems']=schema['maxItems']=100+len(ref['clusters'])
        immutable(folder/'context.json',payload)
        immutable(folder/'request.json',request)
        print('PREPARED',index+1,name,flush=True)
    if test:
        immutable(out/'request_hashes.json',{n:io.sha(out/'requests'/n/'request.json') for n in names})
        io.write(out/'status.json',dict(stage='prepared',completed=80))
    else:
        immutable(out/'source_config.json',config)
        immutable(out/'request_manifest.json',{str(p.relative_to(out)):io.sha(p) for p in (out/'requests').glob('*/*.json')})
        io.write(out/'pipeline.json',dict(stage='prepared',completed=20,api_calls=0))
