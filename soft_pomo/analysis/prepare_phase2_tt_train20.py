"""Travel-time Phase2 preparation, explicit API calls and cached training."""
import argparse
import json
from pathlib import Path
import sys

import prepare_phase2_ac_train20 as legacy
from travel_time_prompt_contract import ROOT, VERSION, prompt, validate_config, export


def setup(variant):
    run = ROOT / f'result/rc101_pure_train20_b128_s42_config{variant}_travel_time_1000_20261010_v2'
    source = run / 'source'
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source/'server'))
    import train_cached as io
    validate_config(io.read(run/'config.json'), variant)
    out = ROOT / f'result/rc101_phase2_train20_b128_s42_config{variant}_{VERSION}'
    return run, source, out, io


def configure():
    legacy.VERSION = VERSION
    legacy.setup = setup
    legacy.objective_prompt = prompt


def validate_artifacts(variant, require_cache=False):
    run, source, out, io = setup(variant)
    protocol = io.read(out/'protocol.json')
    checkpoint = run/'training/checkpoint-500.pt'
    if protocol['version'] != VERSION or protocol['variant'] != variant:
        raise ValueError('Wrong prompt version or objective')
    if Path(protocol['checkpoint']).resolve() != checkpoint.resolve() or protocol['checkpoint_sha256'] != io.sha(checkpoint):
        raise ValueError('Wrong reference checkpoint')
    validate_config(io.read(out/'source_config.json'), variant)
    names = io.read(run/'config.json')['trainer']['train_instances']
    if protocol['train_names'] != names:
        raise ValueError('Training membership changed')
    for name, digest in io.read(out/'request_manifest.json').items():
        if io.sha(out/name) != digest:
            raise ValueError(f'Request artifact changed: {name}')
    for name in names:
        request = io.read(out/'requests'/name/'request.json')
        context = io.read(out/'requests'/name/'context.json')
        from phase2_routes_only import validate_payload
        validate_payload(context)
        expected = prompt(variant)+'\n'+json.dumps(context,separators=(',', ':'),allow_nan=False)
        if request['messages'] != [dict(role='user',content=expected)]:
            raise ValueError('Prompt is not the corrected travel-time prompt')
        if require_cache:
            io.validate_cache(out/'cache'/(name+'_rl_cluster.json'))
    if require_cache:
        manifest = io.read(out/'cache_manifest.json')
        if manifest['request_manifest_sha256'] != io.sha(out/'request_manifest.json'):
            raise ValueError('Cache belongs to different requests')
        for name, digest in manifest['files'].items():
            if io.sha(out/'cache'/name) != digest:
                raise ValueError('Cache changed')


def main(stage, variant, cpu=False):
    configure()
    export()
    if stage == 'prepare':
        from phase2_routes_only import prepare
        prepare(variant, cpu=cpu)
        validate_artifacts(variant)
    elif stage == 'call':
        validate_artifacts(variant)
        legacy.call(variant)
        _, _, out, io = setup(variant)
        io.write(out/'cache_manifest.json', dict(request_manifest_sha256=io.sha(out/'request_manifest.json'),
            files={p.name:io.sha(p) for p in (out/'cache').glob('*.json')}))
        validate_artifacts(variant, require_cache=True)
    elif stage == 'train':
        validate_artifacts(variant, require_cache=True)
        legacy.train(variant)
    else:
        raise ValueError(stage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare','call','train'], nargs='?', default='prepare')
    parser.add_argument('--variant', required=True, choices=['A', 'C'])
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--allow-api', action='store_true')
    args = parser.parse_args()
    if args.stage == 'call' and not args.allow_api:
        parser.error('Paid calls require --allow-api')
    main(args.stage, args.variant, cpu=args.cpu)
