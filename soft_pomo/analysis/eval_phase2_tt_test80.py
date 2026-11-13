"""Evaluate corrected travel-time Phase2 only, with isolated caches and outputs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import eval_phase2_ac_test80 as evaluator
from prepare_phase2_tt_train20 import setup, validate_artifacts
from travel_time_prompt_contract import ROOT, VERSION, prompt, validate_config


def validate_evaluation(variant, require_cache=False):
    run, _, train, io = setup(variant)
    validate_artifacts(variant, require_cache=True)
    saved = io.read(train/'training/settings.json')
    validate_config(saved['config'], variant)
    out = train/'eval_1000_test80'
    protocol = io.read(out/'protocol.json')
    if protocol['version'] != VERSION or protocol['variant'] != variant:
        raise ValueError('Wrong evaluation objective/version')
    if protocol['reference_checkpoint_sha256'] != io.sha(run/'training/checkpoint-500.pt'):
        raise ValueError('Wrong evaluation reference checkpoint')
    names = [r['name'] for r in io.read(run/'split.json')['test']]
    if protocol['test_names'] != names:
        raise ValueError('Evaluation membership changed')
    hashes = io.read(out/'request_hashes.json')
    for name in names:
        folder = out/'requests'/name
        if io.sha(folder/'request.json') != hashes[name]:
            raise ValueError('Evaluation request changed')
        context = io.read(folder/'context.json')
        from phase2_routes_only import validate_payload
        validate_payload(context)
        expected = prompt(variant)+'\n'+json.dumps(context,separators=(',', ':'),allow_nan=False)
        if io.read(folder/'request.json')['messages'] != [dict(role='user',content=expected)]:
            raise ValueError('Wrong evaluation prompt')
    if require_cache:
        manifest = io.read(out/'cache_manifest.json')
        if manifest['request_hashes_sha256'] != io.sha(out/'request_hashes.json'):
            raise ValueError('Evaluation cache provenance mismatch')
        for name in names:
            path = out/'cache'/(name+'_rl_cluster.json')
            io.validate_cache(path)
            if io.sha(path) != manifest['files'][path.name]:
                raise ValueError('Evaluation cache changed')


def main(stage, variant, device='cuda', epoch=1000):
    if stage == 'pipeline':
        for child in ('prepare','call','evaluate'):
            command = [sys.executable,'-u',__file__,child,'--variant',variant,'--device',device]
            if child == 'call':
                command.append('--allow-api')
            subprocess.run(command,cwd=ROOT,check=True)
        return
    evaluator.setup = setup
    evaluator.VERSION = VERSION
    evaluator.objective_prompt = prompt
    validate_artifacts(variant, require_cache=True)
    if stage != 'prepare':
        validate_evaluation(variant, require_cache=stage == 'evaluate')
    if stage == 'prepare':
        from phase2_routes_only import prepare
        prepare(variant, test=True, cpu=True)
    else:
        evaluator.main(stage, variant, device=device, epoch=epoch)
    if stage == 'prepare':
        validate_evaluation(variant)
    if stage == 'call':
        _, _, train, io = setup(variant)
        out = train/'eval_1000_test80'
        io.write(out/'cache_manifest.json',dict(request_hashes_sha256=io.sha(out/'request_hashes.json'),
            files={p.name:io.sha(p) for p in (out/'cache').glob('*.json')}))
        validate_evaluation(variant, require_cache=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','call','evaluate','pipeline'])
    parser.add_argument('--variant',required=True,choices=['A','C'])
    parser.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    parser.add_argument('--epoch',type=int,default=1000,choices=[1000,2000,3000])
    parser.add_argument('--allow-api',action='store_true')
    args = parser.parse_args()
    if args.stage in ('call','pipeline') and not args.allow_api:
        parser.error('Paid calls require --allow-api')
    if args.epoch != 1000 and args.stage != 'evaluate':
        parser.error('Continuation evaluation must reuse existing caches')
    main(args.stage,args.variant,args.device,args.epoch)
