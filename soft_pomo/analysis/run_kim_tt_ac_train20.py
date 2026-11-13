"""Independent Kim travel-time A/C train20 requests; no training or test calls."""
import copy
import json
import math
import os
import sys
from pathlib import Path

from travel_time_prompt_contract import ROOT, VERSION, prompt, validate_config, export
from phase2_routes_only import PUBLIC

QUEUE = ROOT/'result/kim_tt_ac_train20_cache_queue'


def main():
    source_run = ROOT/'result/rc101_pure_train20_b128_s42_configA_travel_time_1000_20261010_v2'
    source = source_run/'source'
    sys.path.insert(0,str(source))
    sys.path.insert(0,str(source/'server'))
    import train_cached as io
    import train_soft_cluster as guided
    import train_vrptw as pure
    from vrptw_env import load_solomon
    from prepare_ontology_train20 import instance_context
    import numpy as np
    export()
    config=io.read(source_run/'config.json')
    names=config['trainer']['train_instances']
    assert len(names)==20
    split=io.read(source_run/'split.json')
    assert not set(names)&{r['name'] for r in split['test']}
    inst=load_solomon(str(source/'data'/(names[0]+'.txt')))
    base=dict(inst,preset_events=[])
    pure._set_seed(42)
    owner=guided.SoftClusterTrainer.__new__(guided.SoftClusterTrainer)
    clusters=owner._cluster_customers(base,max(1,math.ceil(float(base['node_demand'].sum()))))
    assert sorted(n for group in clusters for n in group)==list(range(1,101))
    assert all(sum(float(base['node_demand'][n-1]) for n in group)<=1+1e-6 for group in clusters)
    coords=np.concatenate([base['depot_xy'].numpy(),base['node_xy'].numpy()])*base['max_coord']
    fixed=instance_context(base,coords,clusters)
    for group in fixed['groups']:
        group['members']=group.pop('reference_order')
        group.pop('route')
        group['type']='HeuristicGroup'
    template=io.read(ROOT/'result/rc101_ontology_train20_v5/requests'/names[0]/'request.json')
    def immutable(path,value):
        if path.exists():
            assert io.read(path)==value, str(path)
        else:
            io.write(path,value)
    outputs=[]
    for variant in ('A','C'):
        run=ROOT/f'result/rc101_pure_train20_b128_s42_config{variant}_travel_time_1000_20261010_v2'
        cfg=io.read(run/'config.json')
        validate_config(cfg,variant)
        assert cfg['trainer']['train_instances']==names
        out=ROOT/f'result/rc101_kim_train20_config{variant}_{VERSION}'
        protocol=dict(version=VERSION,variant=variant,train_names=names,clusters=clusters,
            environment=cfg['environment'],group_seed=42,model=template['model'],feedback=False,
            prompt_sha256=io.sha(ROOT/'analysis/prompts'/VERSION/f'kim_config{variant}.txt'),
            clustering_source_sha256=io.sha(source/'train_soft_cluster.py'),scope='20 training calls only')
        immutable(out/'protocol.json',protocol)
        for name in names:
            data=run/'source/data'/(name+'.txt')
            obsroot=ROOT/'server_export/soft_pomo_seed_sweep/train20/data'
            assert io.sha(data)==io.sha(obsroot/(name+'.txt'))
            other=load_solomon(str(data))
            for key in ('depot_xy','node_xy','node_demand','node_tw_open','node_tw_close','node_service'):
                assert np.array_equal(other[key].numpy(),base[key].numpy())
            observation=io.read(obsroot/(name+'.observations.json'))
            assert set(observation)<=PUBLIC
            payload=dict(scenario=name,fixed_instance=fixed,public_event=observation)
            request=copy.deepcopy(template)
            request['messages']=[dict(role='user',content=prompt(variant,'kim')+'\n'+json.dumps(payload,separators=(',', ':'),allow_nan=False))]
            schema=request['response_format']['json_schema']['schema']['properties']['confidence']
            schema['minItems']=schema['maxItems']=100+len(clusters)
            immutable(out/'requests'/name/'request.json',request)
        immutable(out/'request_hashes.json',{n:io.sha(out/'requests'/n/'request.json') for n in names})
        outputs.append(out)
        print('PREPARED',variant,len(names),'groups',len(clusters),flush=True)
    from run_rc101_kim_conversation import decode
    import VRPTWLLMModule as backend
    assert backend.OPENROUTER_API_KEY
    for variant,out in zip(('A','C'),outputs):
        hashes=io.read(out/'request_hashes.json')
        for index,name in enumerate(names):
            folder=out/'requests'/name
            assert io.sha(folder/'request.json')==hashes[name]
            request=io.read(folder/'request.json')
            io.write(QUEUE/'status.json',dict(stage='calling',variant=variant,completed=index,total=20))
            response=folder/'response.txt'
            for attempt in range(3):
                if not response.exists():
                    print('CALL',variant,index+1,name,'attempt',attempt+1,flush=True)
                    os.environ['OPENROUTER_RESPONSE_AUDIT_DIR'] = str(folder/'api_metadata')
                    text=backend.query_llm(request['messages'][0]['content'],request['model'],response_format=request['response_format'])
                    response.write_text(text,encoding='utf-8')
                try:
                    _,cache=decode(response.read_text(encoding='utf-8'),clusters)
                    break
                except (AssertionError, ValueError, KeyError, TypeError) as exc:
                    rejected=folder/f'response_rejected_{len(list(folder.glob("response_rejected_*.txt")))+1}.txt'
                    response.rename(rejected)
                    print('REJECTED',variant,name,rejected.name,flush=True)
                    if attempt==2:
                        raise ValueError(f'Invalid confidence after three validations: {name}') from exc
            path=out/'cache'/(name+'_cluster.json')
            immutable(path,cache)
            io.validate_cache(path)
            print('VALIDATED',variant,index+1,name,flush=True)
        immutable(out/'cache_manifest.json',dict(request_hashes_sha256=io.sha(out/'request_hashes.json'),
            files={p.name:io.sha(p) for p in (out/'cache').glob('*.json')}))
        io.write(out/'status.json',dict(stage='cache_complete',completed=20,training_started=False))
    io.write(QUEUE/'status.json',dict(stage='complete',A=20,C=20,training_started=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        QUEUE.mkdir(parents=True,exist_ok=True)
        (QUEUE/'failure.json').write_text(json.dumps(dict(error=repr(exc))),encoding='utf-8')
        (QUEUE/'status.json').write_text(json.dumps(dict(stage='failed',error=repr(exc))),encoding='utf-8')
        raise
