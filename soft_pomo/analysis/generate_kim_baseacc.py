"""New isolated Kim BASE -> ACC cache pipeline. Never edits legacy caches."""
import argparse
import copy
import json
from pathlib import Path
from generate_kim_l5_all_gpt import ROOT, CASES, validate, io, immutable, call
from base_event_prompt import BASE_INSTRUCTION, EVENT_INSTRUCTION

OLD = ROOT/'result/kim_l5_gpt_rc_r_c_20261026'
OUT = ROOT/'result/kim_l5_baseevent_gpt_rc_r_c_20261027'


def make_request(template, context, prior=None):
    request = copy.deepcopy(template)
    text = template['messages'][0]['content'].rsplit('\n', 1)[0]
    context = copy.deepcopy(context)
    if prior is None:
        # Construct BASE from an allowlist, never redact an ACC context in place.
        context = dict(scenario='BASE', fixed_instance=context['fixed_instance'], public_event={})
        text += '\n' + BASE_INSTRUCTION
    else:
        context['prior_base_confidence'] = prior
        text += '\n' + EVENT_INSTRUCTION
        text += '\nNo current RL routes, performance, visited customers or vehicle states are supplied.'
    request['messages'] = [dict(role='user', content=text+'\n'+json.dumps(context, separators=(',', ':')))]
    return request, context


def prepare_base(case):
    old = OLD/case/'train'
    protocol = io.read(old/'protocol.json')
    name = protocol['names'][0]
    context = io.read(old/'requests'/name/'context.json')
    validate(context, protocol['clusters'])
    request, context = make_request(io.read(old/'requests'/name/'request.json'), context)
    dest = OUT/case/'base'
    immutable(dest/'protocol.json', dict(protocol, names=['BASE'], split='base', accident_guidance='base_then_acc'))
    immutable(dest/'requests/BASE/context.json', context)
    immutable(dest/'requests/BASE/request.json', request)
    immutable(dest/'request_hashes.json', {'BASE':io.sha(dest/'requests/BASE/request.json')})


def prepare_events(case):
    base_path = OUT/case/'base/cache/BASE_cluster.json'
    base = io.validate_cache(base_path)
    for section in ('train', 'test'):
        old = OLD/case/section
        dest = OUT/case/section
        protocol = io.read(old/'protocol.json')
        assert protocol['clusters'] == io.read(OUT/case/'base/protocol.json')['clusters']
        immutable(dest/'protocol.json', dict(protocol, accident_guidance='base_then_acc',
            base_cache_sha256=io.sha(base_path), prior_base_in_acc_prompt=True,
            prior_base_in_rain_prompt=True))
        immutable(dest/'cache/BASE_cluster.json', base)
        for name in protocol['names']:
            source = old/'requests'/name
            context = io.read(source/'context.json')
            validate(context, protocol['clusters'])
            template = io.read(source/'request.json')
            assert '_ACC_' in name or '_RAIN_' in name
            request, context = make_request(template, context, base)
            immutable(dest/'requests'/name/'context.json', context)
            immutable(dest/'requests'/name/'request.json', request)
        immutable(dest/'request_hashes.json', {n:io.sha(dest/'requests'/n/'request.json') for n in protocol['names']})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare-base', 'call-base', 'prepare-events', 'call-events'])
    parser.add_argument('--allow-api', action='store_true')
    args=parser.parse_args()
    if args.stage.startswith('call') and not args.allow_api:
        parser.error('Paid calls require --allow-api')
    for case in CASES:
        if args.stage=='prepare-base': prepare_base(case)
        elif args.stage=='call-base': call(OUT/case/'base', kim=True)
        elif args.stage=='prepare-events': prepare_events(case)
        else:
            for section in ('train','test'): call(OUT/case/section, kim=True)


if __name__=='__main__': main()
