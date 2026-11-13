"""Validated route-only cache generation with bounded response retries."""
import argparse
import os
import sys

from prepare_phase2_tt_train20 import setup, validate_artifacts
from eval_phase2_tt_test80 import validate_evaluation


def main(variant, test):
    _, _, train, io = setup(variant)
    (validate_evaluation if test else validate_artifacts)(variant)
    from travel_time_prompt_contract import ROOT
    sys.path.insert(0, str(ROOT))
    import VRPTWLLMModule as backend
    from run_rc101_kim_conversation import decode
    if not backend.OPENROUTER_API_KEY:
        raise RuntimeError('OPENROUTER_API_KEY is unavailable')
    out = train / 'eval_1000_test80' if test else train
    names = io.read(out / 'protocol.json')['test_names' if test else 'train_names']
    for index, name in enumerate(names):
        folder = out / 'requests' / name
        request = io.read(folder / 'request.json')
        response = folder / 'response.txt'
        io.write(out / ('status.json' if test else 'pipeline.json'), dict(stage='calling', completed=index, total=len(names), name=name))
        for attempt in range(3):
            if not response.exists():
                print('CALL', variant, index + 1, name, 'attempt', attempt + 1, flush=True)
                os.environ['OPENROUTER_RESPONSE_AUDIT_DIR'] = str(folder / 'api_metadata')
                text = backend.query_llm(request['messages'][0]['content'], request['model'], response_format=request['response_format'])
                response.write_text(text, encoding='utf-8')
            try:
                _, cache = decode(response.read_text(encoding='utf-8'), io.read(folder / 'reference.json')['clusters'])
                break
            except (AssertionError, ValueError, KeyError, TypeError):
                response.rename(folder / f'response_rejected_{len(list(folder.glob("response_rejected_*.txt"))) + 1}.txt')
                if attempt == 2:
                    raise
        path = out / 'cache' / (name + '_rl_cluster.json')
        if path.exists() and io.read(path) != cache:
            raise ValueError('Existing cache differs from response')
        io.write(path, cache)
        io.validate_cache(path)
        print('VALIDATED', variant, index + 1, name, flush=True)
    key, file = ('request_hashes_sha256', 'request_hashes.json') if test else ('request_manifest_sha256', 'request_manifest.json')
    io.write(out / 'cache_manifest.json', {key: io.sha(out / file), 'files': {p.name: io.sha(p) for p in (out / 'cache').glob('*.json')}})
    (validate_evaluation if test else validate_artifacts)(variant, require_cache=True)
    io.write(out / ('status.json' if test else 'pipeline.json'), dict(stage='cache_complete', completed=len(names)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', required=True, choices=['A', 'C'])
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--allow-api', action='store_true')
    args = parser.parse_args()
    if not args.allow_api:
        parser.error('Paid calls require --allow-api')
    main(args.variant, args.test)
