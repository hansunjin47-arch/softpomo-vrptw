"""Create held-out Kim customer-only caches without GPU use or evaluation."""
import argparse
import json
import os

import queue_kim_customer_only_ac as train

ROOT, io = train.ROOT, train.io
QUEUE = ROOT/'result/kim_customer_only_test80_cache_queue_20261013'


def prepare(variant):
    root=train.location(variant)
    protocol=io.read(root/'protocol.json')
    old=ROOT/f'result/rc101_kim_train20_config{variant}_ontology_routes_only_ac_travel_time_v1/eval_1000_test80_gpu'
    original=io.read(old/'protocol.json')
    assert original['clusters']==protocol['clusters']
    names=original['test_names']
    assert len(set(names))==80 and not set(names)&set(protocol['train_names'])
    out=root/'eval_test80_customer_only'
    prompt=train.prompt(variant)
    training_request=io.read(root/'requests'/protocol['train_names'][0]/'request.json')
    fixed=json.loads(training_request['messages'][0]['content'].split('\n')[-1])['fixed_instance']
    hashes=io.read(old/'request_hashes.json')
    for name in names:
        path=old/'requests'/name/'request.json'
        assert io.sha(path)==hashes[name]
        request=io.read(path)
        context=json.loads(request['messages'][0]['content'].split('\n')[-1])
        for group in context['fixed_instance']['groups']:
            group['required_output_customers']=[n for n in group['required_output_customers'] if n]
        assert context['fixed_instance']==fixed
        assert context['scenario']==name
        request['messages']=[dict(role='user',content=prompt+'\n'+json.dumps(context,separators=(',',':'),allow_nan=False))]
        request['response_format']=training_request['response_format']
        assert request['model']==training_request['model']
        train.immutable(out/'requests'/name/'request.json',request)
    train.immutable(out/'protocol.json',dict(version=train.VERSION,variant=variant,test_names=names,
        clusters=protocol['clusters'],training_protocol_sha256=io.sha(root/'protocol.json'),
        feedback=False,checkpoint_independent=True,model=training_request['model']))
    train.immutable(out/'request_hashes.json',{n:io.sha(out/'requests'/n/'request.json') for n in names})
    print('PREPARED',variant,80,flush=True)


def call(variant):
    out=train.location(variant)/'eval_test80_customer_only'
    protocol=io.read(out/'protocol.json')
    hashes=io.read(out/'request_hashes.json')
    for index,name in enumerate(protocol['test_names']):
        folder=out/'requests'/name
        assert io.sha(folder/'request.json')==hashes[name]
        request=io.read(folder/'request.json')
        response=folder/'response.txt'
        io.write(QUEUE/'status.json',dict(stage='calling',variant=variant,completed=index,total=80))
        if not response.exists():
            os.environ['OPENROUTER_RESPONSE_AUDIT_DIR']=str(folder/'api_metadata')
            print('CALL',variant,index+1,name,flush=True)
            text=train.backend.query_llm('',request['model'],messages=request['messages'],response_format=request['response_format'])
            response.write_text(text,encoding='utf-8')
        cache=train.customer.decode(response.read_text(encoding='utf-8'),protocol['clusters'])
        train.immutable(out/'cache'/(name+'_cluster.json'),cache)
        io.validate_cache(out/'cache'/(name+'_cluster.json'))
        io.write(out/'status.json',dict(stage='calling',completed=index+1,total=80))
        print('VALIDATED',variant,index+1,flush=True)
    train.immutable(out/'cache_manifest.json',dict(request_hashes_sha256=io.sha(out/'request_hashes.json'),
        files={p.name:io.sha(p) for p in (out/'cache').glob('*.json')}))
    io.write(out/'status.json',dict(stage='cache_complete',completed=80))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--allow-api',action='store_true')
    args=parser.parse_args()
    try:
        for variant in ('A','C'):
            prepare(variant)
        if args.allow_api:
            assert train.backend.OPENROUTER_API_KEY
            for variant in ('A','C'):
                call(variant)
            io.write(QUEUE/'status.json',dict(stage='complete',A=80,C=80,evaluation=False))
    except Exception as exc:
        io.write(QUEUE/'status.json',dict(stage='failed',error=repr(exc)))
        raise
