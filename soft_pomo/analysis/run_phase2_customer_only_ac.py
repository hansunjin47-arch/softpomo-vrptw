"""New customer-only prompts/caches; branch A/C from matching Pure500 checkpoints."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import prepare_phase2_tt_train20 as prior
from travel_time_prompt_contract import ROOT, objective, prompt as old_prompt

VERSION = 'ontology_routes_only_customer_only_ac_v1'
QUEUE = ROOT / 'result/phase2_customer_only_ac_queue_20261012'


def prompt(variant):
    old = old_prompt(variant)
    interpretation = old.split('INTERPRETING LEARNED ROUTES\n')[1].split('\nOUTPUT')[0]
    return ('Assign static action-preference scores to reference-group CUSTOMERS ONLY.\n'
        + objective(variant) + '''
Capacity is hard; closing times are soft. All customers must be served.
SCORE MEANING AND EXACT DECODER CONTRACT
Customer scores in [0,1] are static action preferences, not probabilities.
The first customer's reference group stays active until depot return.
There is no guidance before the first customer is chosen.
For an active-group customer i, effective bias is b_i = 5*c_i.
For other-group customers and depot 0, effective bias is zero.
No customer score is transformed into a negative bias. No depot score is requested or inferred.
Final logits are RL base logits plus these biases; masks enforce visited/capacity constraints.
RL decides when to return to depot using its base logits. Positive customer bias nevertheless
favors continuing customer service relative to depot; consider this indirect effect conservatively.
Zero is neutral; equal positive scores favor group members over outsiders and depot.
Customers may be served outside their reference group, without carrying their score to that group.
You receive no live state; RL observes current position, time, load and remaining customers.
Rain bias is available after route start. Accident bias activates after disclosure on the current
vehicle clock, and remains available afterwards. Each new vehicle starts at time zero.

INTERPRETING LEARNED ROUTES
''' + interpretation + '''
OUTPUT
Return ONLY the supplied confidence JSON schema with {cluster,customer,score} entries.
Use one-based cluster IDs and every required CUSTOMER exactly once, 100 entries total.
Never output customer 0 or depot scores. No explanations, route edits, assignments or extra pairs.
''')


def setup(variant):
    run, source, _, io = prior.setup(variant)
    out = ROOT / f'result/rc101_phase2_train20_b128_s42_config{variant}_{VERSION}'
    return run, source, out, io


def immutable(io, path, value):
    if path.exists():
        assert io.read(path) == value, str(path)
    else:
        io.write(path, value)


def decode(text, clusters):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError('Duplicate JSON key')
            obj[key] = value
        return obj
    wire = json.loads(text, object_pairs_hook=unique)
    assert set(wire) == {'confidence'} and len(wire['confidence']) == 100
    expected = {(k+1, n) for k, group in enumerate(clusters) for n in group}
    cache = {str(k): {} for k in range(len(clusters))}
    seen = set()
    for entry in wire['confidence']:
        assert set(entry) == {'cluster', 'customer', 'score'}
        k, n, score = entry['cluster'], entry['customer'], entry['score']
        assert type(k) is int and type(n) is int
        assert (k,n) in expected and (k,n) not in seen
        assert type(score) in (int,float) and math.isfinite(score) and 0 <= score <= 1
        seen.add((k,n))
        cache[str(k-1)][str(n)] = score
    assert seen == expected
    # Compatibility field only: zero depot cannot trigger the strict c_i < d rule.
    cache['_depot'] = {str(k): 0.0 for k in range(len(clusters))}
    return cache


def validate(variant, caches=False):
    run, source, out, io = setup(variant)
    protocol = io.read(out/'protocol.json')
    assert protocol['version'] == VERSION and protocol['checkpoint_sha256'] == io.sha(run/'training/checkpoint-500.pt')
    for name, digest in io.read(out/'request_manifest.json').items():
        assert io.sha(out/name) == digest
    if caches:
        manifest = io.read(out/'cache_manifest.json')
        assert manifest['request_manifest_sha256'] == io.sha(out/'request_manifest.json')
        for name in protocol['train_names']:
            path = out/'cache'/(name+'_rl_cluster.json')
            cache = io.validate_cache(path)
            assert all(v == 0 for v in cache['_depot'].values())
            assert io.sha(path) == manifest['files'][path.name]


def main(stage, variant):
    run, source, out, io = setup(variant)
    if stage == 'prepare':
        prior.validate_artifacts(variant, require_cache=True)
        _, _, old, _ = prior.setup(variant)
        text = prompt(variant)
        prompt_path = ROOT/'analysis/prompts'/VERSION/f'phase2_config{variant}.txt'
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        if prompt_path.exists():
            assert prompt_path.read_text(encoding='utf-8') == text
        else:
            prompt_path.write_text(text, encoding='utf-8')
        protocol = io.read(old/'protocol.json')
        protocol.update(version=VERSION, prompt_sha256=io.sha(prompt_path),
            decoder='Customer beta*c only; zero depot; no negative transformation',
            reference_protocol_sha256=io.sha(old/'protocol.json'))
        immutable(io,out/'protocol.json',protocol)
        immutable(io,out/'source_config.json',io.read(old/'source_config.json'))
        for name in protocol['train_names']:
            folder = out/'requests'/name
            context = io.read(old/'requests'/name/'context.json')
            for group in context['fixed_instance']['groups']:
                group['required_output_customers'] = [n for n in group['required_output_customers'] if n != 0]
            request = io.read(old/'requests'/name/'request.json')
            request['messages'] = [dict(role='user',content=text+'\n'+json.dumps(context,separators=(',', ':'),allow_nan=False))]
            schema = request['response_format']['json_schema']['schema']['properties']['confidence']
            schema['minItems'] = schema['maxItems'] = 100
            schema['items']['properties']['customer']['minimum'] = 1
            for filename, value in [('context.json',context),('request.json',request),('reference.json',io.read(old/'requests'/name/'reference.json'))]:
                immutable(io,folder/filename,value)
        immutable(io,out/'request_manifest.json',{p.relative_to(out).as_posix():io.sha(p) for p in (out/'requests').glob('*/*.json')})
        io.write(out/'pipeline.json',dict(stage='prepared'))
        validate(variant)
        print('PREPARED',variant,flush=True)
    elif stage == 'call':
        validate(variant)
        sys.path.insert(0, str(ROOT))
        import VRPTWLLMModule as backend
        assert backend.OPENROUTER_API_KEY
        for index,name in enumerate(io.read(out/'protocol.json')['train_names']):
            folder=out/'requests'/name
            request=io.read(folder/'request.json')
            response=folder/'response.txt'
            clusters=io.read(folder/'reference.json')['clusters']
            for attempt in range(3):
                if not response.exists():
                    print('CALL',variant,index+1,name,attempt+1,flush=True)
                    os.environ['OPENROUTER_RESPONSE_AUDIT_DIR']=str(folder/'api_metadata')
                    response.write_text(backend.query_llm(request['messages'][0]['content'],request['model'],response_format=request['response_format']),encoding='utf-8')
                try:
                    cache=decode(response.read_text(encoding='utf-8'),clusters)
                    break
                except (AssertionError,ValueError,KeyError,TypeError):
                    response.rename(folder/f'response_rejected_{len(list(folder.glob("response_rejected_*.txt")))+1}.txt')
                    if attempt==2:
                        raise
            immutable(io,out/'cache'/(name+'_rl_cluster.json'),cache)
            io.write(out/'pipeline.json',dict(stage='calling',completed=index+1,total=20))
            print('VALIDATED',variant,index+1,flush=True)
        immutable(io,out/'cache_manifest.json',dict(request_manifest_sha256=io.sha(out/'request_manifest.json'),files={p.name:io.sha(p) for p in (out/'cache').glob('*.json')}))
        validate(variant,True)
        io.write(out/'pipeline.json',dict(stage='cache_complete',completed=20))
    elif stage == 'train':
        validate(variant,True)
        import prepare_phase2_ac_train20 as trainer
        trainer.setup=setup
        trainer.VERSION=VERSION
        trainer.train(variant)
        state=io.read(out/'training/status.json')
        assert state['status']=='complete' and state['epoch']==1000


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','call','train','queue'])
    parser.add_argument('--variant',choices=['A','C'])
    parser.add_argument('--allow-api',action='store_true')
    args=parser.parse_args()
    if args.stage in ('call','queue') and not args.allow_api:
        parser.error('Paid calls require --allow-api')
    if args.stage=='queue':
        QUEUE.mkdir(parents=True,exist_ok=True)
        try:
            for stage in ('prepare','call','train'):
                for variant in ('A','C'):
                    (QUEUE/'status.json').write_text(json.dumps(dict(stage=stage,variant=variant)),encoding='utf-8')
                    cmd=[sys.executable,'-u',__file__,stage,'--variant',variant]
                    if stage=='call':
                        cmd.append('--allow-api')
                    subprocess.run(cmd,cwd=ROOT,check=True)
            (QUEUE/'status.json').write_text(json.dumps(dict(stage='complete',evaluation=False)),encoding='utf-8')
        except Exception as exc:
            (QUEUE/'status.json').write_text(json.dumps(dict(stage='failed',error=repr(exc))),encoding='utf-8')
            raise
    else:
        if args.variant is None:
            parser.error('--variant required')
        main(args.stage,args.variant)
