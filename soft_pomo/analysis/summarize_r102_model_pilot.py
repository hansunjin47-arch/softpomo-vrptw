"""Audit paired pilot artifacts and summarize score distributions."""
import json
from pathlib import Path
import statistics
import sys

import generate_kim500_openrouter as gen


def main():
    root = Path(sys.argv[1])
    manifest = json.loads((root/'manifest.json').read_text())
    lines = ['# R102 Model Pilot', '',
        'Same frozen prompts and reference routes. No policy training or routing-quality comparison. Reasoning uses model/provider defaults, not equal compute.', '',
        '| Model | Case | Customer mean | Range | Depot mean | Customer > own depot |',
        '|---|---|---:|---|---:|---:|']
    costs = {}
    for model in manifest['models']:
        folder = root/model.replace('/', '__')
        cost = 0.0
        for prompt in sorted((root/'inputs/training').glob('*_prompt.txt')):
            stem = prompt.name.removesuffix('_prompt.txt')
            result = manifest['results'][model+':'+stem]
            assert result['success']
            assert gen.sha(prompt) == gen.sha(folder/prompt.name) == result['prompt_sha256']
            inp = json.loads((folder/f'{stem}_input.json').read_text())
            target = stem+'.json' if inp['wave'] is not None else stem+'_rl_cluster.json'
            cache = json.loads((folder/target).read_text())
            scores = [v for k in cache if not k.startswith('_') for v in cache[k].values()]
            depot = list(cache['_depot'].values())
            above = sum(v > cache['_depot'][k] for k in cache if not k.startswith('_') for v in cache[k].values())
            assert len(scores) == 100
            lines.append(f'| {model} | {stem} | {statistics.mean(scores):.3f} | {min(scores):.3f}-{max(scores):.3f} | {statistics.mean(depot):.3f} | {above}/100 |')
            value = (result.get('usage') or {}).get('cost')
            if value is not None:
                cost += value
        costs[model] = cost
    lines += ['', 'Successful-attempt reported costs (USD; excludes failed attempts): '+json.dumps(costs), '',
        'A wide range, high score, or low correlation with table order does not itself establish good guidance. Scores are action biases, not calibrated probabilities. Compare continuation training before claiming routing improvements.']
    (root/'comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
