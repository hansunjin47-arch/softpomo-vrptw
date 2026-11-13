"""RC train-only mixed Kim cache generation, with a two-case pilot gate."""
import argparse
import copy
import json
from pathlib import Path
from generate_kim_baseacc import ROOT, io, immutable
from base_event_prompt import EVENT_INSTRUCTION
from run_phase2_customer_only_ac import decode

BASE=ROOT/'result/kim_rl_example_base_pilot_rc_20261027'
OLD=ROOT/'result/kim_l5_gpt_rc_r_c_20261026/rc101/train'
OUT=ROOT/'result/kim_rl_example_rc_train_20261027'
SPLIT='train'


def configure(case, split):
    global BASE, OLD, OUT, SPLIT
    SPLIT=split
    short=case.replace('101','')
    BASE=ROOT/f'result/kim_rl_example_base_pilot_{short}_20261027'
    OLD=ROOT/f'result/kim_l5_gpt_rc_r_c_20261026/{case}/{split}'
    OUT=ROOT/f'result/kim_rl_example_{short}_{split}_20261027'


def prepare():
    protocol=io.read(OLD/'protocol.json')
    base=io.validate_cache(BASE/'cache/BASE_cluster.json')
    example=io.read(BASE/'requests/BASE/context.json')['rl_reference_example']
    assert example['scenario'] not in protocol['names']
    immutable(OUT/'protocol.json',dict(protocol,variant='Kim with BASE and fixed separate RL example',
        base_sha256=io.sha(BASE/'cache/BASE_cluster.json'),
        reference_sha256=io.sha(BASE/'reference.json'),accident_guidance='zero_then_acc',
        base_usage='Prompt reference only; accident pre-disclosure policy bias remains zero',
        test_generation=SPLIT=='test'))
    for name in protocol['names']:
        source=OLD/'requests'/name
        context=copy.deepcopy(io.read(source/'context.json'))
        assert set(context)=={'scenario','fixed_instance','public_event'}, set(context)
        context['prior_base_confidence']=base
        context['rl_reference_example']=example
        template=io.read(source/'request.json')
        text=template['messages'][0]['content'].rsplit('\n',1)[0]
        text=text.replace('No RL route, simulator-derived arrival/slack, travel-time multiplier, or counterfactual schedule is supplied.',
            'No RL solution or measured performance for the TARGET instance is supplied.')
        text+='\n'+EVENT_INSTRUCTION
        text+=('\nThe rl_reference_example is a separate, fixed RAIN case. Its route and measured '
            'performance came from one greedy Pure RL rollout, not an optimal solver or a confidence label. '
            'Do not copy its route, customer priorities or event into the target. '
            'The target public_event, not the example event, governs your target assessment. '
            'Use the example only as non-authoritative context about possible RL behavior and weighted costs. '
            'No current target RL route, target performance or live vehicle state is supplied.')
        template['messages']=[dict(role='user',content=text+'\n'+json.dumps(context,separators=(',',':'),allow_nan=False))]
        immutable(OUT/'requests'/name/'context.json',context)
        immutable(OUT/'requests'/name/'request.json',template)
    immutable(OUT/'request_hashes.json',{n:io.sha(OUT/'requests'/n/'request.json') for n in protocol['names']})
    return protocol


def generate(names,protocol):
    from compare_llm_single_customer_only import call_once
    hashes=io.read(OUT/'request_hashes.json')
    for name in names:
        folder=OUT/'requests'/name
        assert io.sha(folder/'request.json')==hashes[name]
        response=folder/'response.txt'
        if not response.exists():
            if (folder/'attempted.json').exists():
                raise RuntimeError('Inspect previous attempt before retry: '+name)
            immutable(folder/'attempted.json',dict(model=protocol['model']))
            io.write(OUT/'status.json',dict(stage='calling',scenario=name))
            print('CALL',name,flush=True)
            value=call_once(io.read(folder/'request.json'),folder)
            with response.open('x',encoding='utf-8') as f:f.write(value)
        cache=decode(response.read_text(encoding='utf-8'),protocol['clusters'])
        immutable(OUT/'cache'/(name+'_cluster.json'),cache)
        io.validate_cache(OUT/'cache'/(name+'_cluster.json'))
        base=io.read(BASE/'cache/BASE_cluster.json')
        changes=[abs(v-base[k][n]) for k,g in cache.items() if not k.startswith('_') for n,v in g.items()]
        io.write(folder/'validation.json',dict(customers=len(changes),changed=sum(x>1e-8 for x in changes),
            mean_absolute_change=sum(changes)/len(changes),max_change=max(changes)))
        print('VALIDATED',name,'changed',sum(x>1e-8 for x in changes),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','pilot','remaining','all']);ap.add_argument('--allow-api',action='store_true');ap.add_argument('--case',choices=['rc101','r101','c101'],default='rc101');ap.add_argument('--split',choices=['train','test'],default='train');args=ap.parse_args()
    configure(args.case,args.split)
    if args.stage!='prepare' and not args.allow_api:ap.error('--allow-api required')
    protocol=prepare()
    pilot=[next(n for n in protocol['names'] if '_RAIN_' in n),next(n for n in protocol['names'] if '_ACC_' in n)]
    if args.stage=='pilot':
        generate(pilot,protocol)
        io.write(OUT/'status.json',dict(stage='pilot_complete',names=pilot))
    elif args.stage in ('remaining','all'):
        if args.stage=='remaining':
            for n in pilot:io.validate_cache(OUT/'cache'/(n+'_cluster.json'))
        generate([n for n in protocol['names'] if args.stage=='all' or n not in pilot],protocol)
        immutable(OUT/'cache_manifest.json',dict(request_hashes_sha256=io.sha(OUT/'request_hashes.json'),
            files={n+'_cluster.json':io.sha(OUT/'cache'/(n+'_cluster.json')) for n in protocol['names']}))
        io.write(OUT/'status.json',dict(stage='complete',completed=len(protocol['names']),split=SPLIT))
