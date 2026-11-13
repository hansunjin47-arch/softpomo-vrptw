"""Paired model pilot on five frozen R102 inputs; never trains a policy."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import generate_kim500_openrouter as gen


def main():
    if not os.environ.get('OPENROUTER_API_KEY'):
        raise RuntimeError('OPENROUTER_API_KEY is required')
    root = gen.ROOT / 'result_soft' / ('r102_model_pilot_' + time.strftime('%Y%m%d_%H%M%S'))
    root.mkdir(exist_ok=False)
    prepared = root / 'inputs'
    prepared.mkdir()
    # The existing generator's explicit output option is a resume interface.
    originals = {str(p): gen.sha(p) for p in (gen.ROOT/'result_soft/llm_cache_kim_initial_v1_20260917_214827/r1').glob('*.json')}
    checkpoint = gen.SOURCE/'r1/r102-r112/checkpoint-500.pt'
    originals[str(checkpoint)] = gen.sha(checkpoint)
    gen.save(prepared/'manifest.json', dict(original_hashes=originals, confidence_only=True, results={}))
    print('PILOT', root, flush=True)
    subprocess.run([sys.executable, '-u', str(Path(gen.__file__)), str(prepared), '--benchmark', 'r1', '--instances', '102', '--confidence-only', '--prepare-only'], check=True)
    prompts = sorted((prepared/'training').glob('*_prompt.txt'))
    assert len(prompts) == 5
    manifest = dict(models=['google/gemini-2.5-pro', 'anthropic/claude-sonnet-4.6'],
        purpose='Paired practical-default model comparison, not equal reasoning-compute comparison',
        temperature=0, max_tokens=16000, reasoning='provider/model default; not equalized',
        max_attempts_per_case=3, input_manifest_sha256=gen.sha(prepared/'manifest.json'), results={})
    gen.CONFIDENCE_ONLY = True
    for model in manifest['models']:
        gen.MODEL = model
        folder = root / model.replace('/', '__')
        folder.mkdir()
        for prompt_path in prompts:
            stem = prompt_path.name.removesuffix('_prompt.txt')
            source = prompt_path.parent
            inp = json.loads((source/f'{stem}_input.json').read_text())
            for path in [prompt_path, source/f'{stem}_input.json']:
                shutil.copy2(path, folder/path.name)
            print('CALL', model, stem, flush=True)
            try:
                confidence, meta = gen.query(prompt_path.read_text(encoding='utf-8'), inp['clusters'], folder, stem)
                reference = json.loads((source/(inp['source_reference']+'_reference.json')).read_text())
                cache = {str(k): {str(n): confidence[str(k+1)][str(n)] for n in ns} for k, ns in enumerate(inp['clusters'])}
                cache['_depot'] = {str(k):confidence[str(k+1)]['0'] for k in range(len(inp['clusters']))}
                cache['_stats'] = reference['replay']['_stats']
                target = stem+'.json' if inp['wave'] is not None else stem+'_rl_cluster.json'
                gen.save(folder/target, cache)
                manifest['results'][model+':'+stem] = dict(success=True, prompt_sha256=gen.sha(prompt_path), **meta)
                print('SUCCESS', model, stem, flush=True)
            except Exception as exc:
                manifest['results'][model+':'+stem] = dict(success=False, error=type(exc).__name__)
                gen.save(root/'manifest.json', manifest)
                raise RuntimeError('Pilot stopped; no fallback. See saved validation artifacts.') from None
            gen.save(root/'manifest.json', manifest)
    assert all(gen.sha(Path(p)) == h for p,h in originals.items())
    print('DONE', root, flush=True)


if __name__ == '__main__':
    main()
