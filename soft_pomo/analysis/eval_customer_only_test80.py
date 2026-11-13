"""Customer-only v1 test caches and matched GPU evaluation; no prompt migration."""
import argparse
import json
import os
import subprocess
import sys
import time

import run_phase2_customer_only_ac as current
from travel_time_prompt_contract import ROOT


def main(stage, variant):
    run, source, train, io = current.setup(variant)
    current.validate(variant, True)
    out = train / 'eval_1000_test80'
    names = [r['name'] for r in io.read(run/'split.json')['test']]
    final = train/'training/checkpoint-1000.pt'
    assert io.read(train/'training/status.json')['status'] == 'complete'
    if stage == 'prepare':
        from eval_phase2_tt_test80 import validate_evaluation
        validate_evaluation(variant, True)
        _, _, old_train, _ = current.prior.setup(variant)
        old = old_train/'eval_1000_test80'
        protocol = io.read(old/'protocol.json')
        assert protocol['test_names'] == names
        protocol.update(version=current.VERSION, final_checkpoint_sha256=io.sha(final),
            prompt_sha256=io.read(train/'protocol.json')['prompt_sha256'],
            decoder='Customer beta*c only; zero depot; no negative transformation')
        current.immutable(io,out/'protocol.json',protocol)
        for name in names:
            folder = out/'requests'/name
            context = io.read(old/'requests'/name/'context.json')
            for group in context['fixed_instance']['groups']:
                group['required_output_customers'] = [n for n in group['required_output_customers'] if n]
            request = io.read(old/'requests'/name/'request.json')
            request['messages'] = [dict(role='user',content=current.prompt(variant)+'\n'+json.dumps(context,separators=(',', ':'),allow_nan=False))]
            schema = request['response_format']['json_schema']['schema']['properties']['confidence']
            schema['minItems'] = schema['maxItems'] = 100
            schema['items']['properties']['customer']['minimum'] = 1
            for filename,value in [('context.json',context),('request.json',request),('reference.json',io.read(old/'requests'/name/'reference.json'))]:
                current.immutable(io,folder/filename,value)
        current.immutable(io,out/'request_hashes.json',{n:io.sha(out/'requests'/n/'request.json') for n in names})
        io.write(out/'status.json',dict(stage='prepared',completed=80))
        print('PREPARED',variant,80,flush=True)
        return
    assert io.read(out/'protocol.json')['final_checkpoint_sha256'] == io.sha(final)
    hashes = io.read(out/'request_hashes.json')
    for name in names:
        assert io.sha(out/'requests'/name/'request.json') == hashes[name]
    if stage == 'call':
        sys.path.insert(0,str(ROOT))
        import VRPTWLLMModule as backend
        assert backend.OPENROUTER_API_KEY
        for index,name in enumerate(names):
            folder=out/'requests'/name
            response=folder/'response.txt'
            request=io.read(folder/'request.json')
            for attempt in range(3):
                if not response.exists():
                    print('CALL',variant,index+1,name,flush=True)
                    os.environ['OPENROUTER_RESPONSE_AUDIT_DIR']=str(folder/'api_metadata')
                    response.write_text(backend.query_llm(request['messages'][0]['content'],request['model'],response_format=request['response_format']),encoding='utf-8')
                try:
                    cache=current.decode(response.read_text(encoding='utf-8'),io.read(folder/'reference.json')['clusters'])
                    break
                except (AssertionError,ValueError,KeyError,TypeError):
                    response.rename(folder/f'response_rejected_{len(list(folder.glob("response_rejected_*.txt")))+1}.txt')
                    if attempt==2:
                        raise
            current.immutable(io,out/'cache'/(name+'_rl_cluster.json'),cache)
            io.write(out/'status.json',dict(stage='calling',completed=index+1,total=80))
            print('VALIDATED',variant,index+1,flush=True)
        current.immutable(io,out/'cache_manifest.json',dict(request_hashes_sha256=io.sha(out/'request_hashes.json'),files={p.name:io.sha(p) for p in (out/'cache').glob('*.json')}))
        io.write(out/'status.json',dict(stage='cache_complete',completed=80))
        return
    manifest=io.read(out/'cache_manifest.json')
    assert manifest['request_hashes_sha256']==io.sha(out/'request_hashes.json')
    for name in names:
        path=out/'cache'/(name+'_rl_cluster.json')
        cache=io.validate_cache(path)
        assert all(d==0 for d in cache['_depot'].values())
        assert io.sha(path)==manifest['files'][path.name]
    # Preserve the single-GPU training queue; evaluating does not change training.
    deadline=time.monotonic()+24*3600
    while True:
        status=io.read(current.QUEUE/'status.json')
        if status['stage']=='complete':
            break
        if status['stage']=='failed':
            raise RuntimeError(status)
        if time.monotonic()>deadline:
            raise TimeoutError('Training queue has not finished')
        io.write(out/'status.json',dict(stage='waiting_for_training_gpu',completed=80))
        time.sleep(30)
    import eval_phase2_ac_test80 as evaluator
    evaluator.setup=current.setup
    evaluator.VERSION=current.VERSION
    evaluator.objective_prompt=current.prompt
    evaluator.main('evaluate',variant,device='cuda',epoch=1000)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','call','evaluate','pipeline'])
    parser.add_argument('--variant',required=True,choices=['A','C'])
    parser.add_argument('--allow-api',action='store_true')
    args=parser.parse_args()
    if args.stage in ('call','pipeline') and not args.allow_api:
        parser.error('Paid calls require --allow-api')
    if args.stage=='pipeline':
        for stage in ('prepare','call','evaluate'):
            command=[sys.executable,'-u',__file__,stage,'--variant',args.variant]
            if stage=='call':
                command.append('--allow-api')
            subprocess.run(command,cwd=ROOT,check=True)
    else:
        main(args.stage,args.variant)
