"""Isolated BASE-only Kim cache generation with strict validation and raw logs."""
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train_soft_cluster import SoftClusterTrainer
from SoftClusterOntology import SoftClusterOntology
from SoftClusterLLMModule import build_all_clusters_prompt
from vrptw_env import load_solomon
from run_c_fewshot_pilot import strict_response


def save(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    tmp.replace(path)


def main():
    out = (Path(sys.argv[1]).resolve() if len(sys.argv)>1 else
           ROOT/'result_soft'/('llm_cache_kim_initial_v1_' + time.strftime('%Y%m%d_%H%M%S')))
    out.mkdir(parents=True, exist_ok=len(sys.argv)>1)
    names = ([f'C{i}' for i in range(101,110)] + [f'RC{i}' for i in range(101,109)]
             + [f'R{i}' for i in range(101,113)])
    config = dict(model='gpt-oss:20b', stream=False, think=True,
                  options=dict(temperature=0, num_ctx=131072, num_predict=32768))
    manifest = dict(names=names, config=config, prompt_version='kim-initial-v1',
                    scenarios=['BASE'], fewshot=False, misfit=False, reward_config='E',
                    retries=0, results={}, code_hashes={})
    for filename in ['SoftClusterLLMModule.py', 'train_soft_cluster.py',
                     'analysis/generate_kim_initial_cache.py']:
        manifest['code_hashes'][filename] = hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()
    if (out/'manifest.json').exists():
        manifest=json.loads((out/'manifest.json').read_text())
        assert manifest['config']==config
    save(out/'manifest.json', manifest)
    print('OUTPUT', out, flush=True)
    owner = SoftClusterTrainer.__new__(SoftClusterTrainer)
    for name in names:
        if name in manifest['results']:
            continue
        start = time.time()
        family = 'rc1' if name.startswith('RC') else 'c1' if name.startswith('C') else 'r1'
        folder = out/family
        folder.mkdir(exist_ok=True)
        try:
            source = ROOT.parent/'data/Solomon'/f'{name.lower()}.txt'
            inst = load_solomon(str(source))
            assert not inst.get('preset_events'), 'BASE only'
            clusters = owner._cluster_customers(inst, max(1, math.ceil(float(inst['node_demand'].sum()))))
            assert sorted(n for ns in clusters for n in ns) == list(range(1, inst['n_customers']+1))
            ont = SoftClusterOntology(inst)
            contexts = [ont.get_cluster_context(ns, k) for k, ns in enumerate(clusters)]
            prompt = build_all_clusters_prompt(inst, clusters, contexts,
                use_cot=True, use_ontology=True, reward_config='E', use_misfit=False,
                cluster_source='kim')
            (folder/f'{name}_prompt.txt').write_text(prompt, encoding='utf-8')
            save(folder/f'{name}_input.json', dict(clusters=clusters,
                data_hash=hashlib.sha256(source.read_bytes()).hexdigest(),
                prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()))
            print('CALL', name, 'clusters', len(clusters), flush=True)
            request = urllib.request.Request('http://localhost:11434/api/generate',
                data=json.dumps(dict(config, prompt=prompt)).encode(),
                headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=900) as response:
                raw = json.load(response)
            save(folder/f'{name}_raw.json', raw)
            assert raw.get('done') and raw.get('done_reason') == 'stop', 'incomplete generation'
            conf = strict_response(raw['response'], clusters)
            cache = {str(k): {str(n): conf[str(k+1)][str(n)] for n in ns}
                     for k, ns in enumerate(clusters)}
            cache['_depot'] = {str(k): conf[str(k+1)]['0'] for k in range(len(clusters))}
            save(folder/f'{name}_cluster.json', cache)
            result = dict(success=True, clusters=len(clusters), eval_count=raw.get('eval_count'))
        except Exception as exc:
            result = dict(success=False, error=f'{type(exc).__name__}: {exc}')
        result['seconds'] = time.time()-start
        manifest['results'][name] = result
        save(out/'manifest.json', manifest)
        print(name, json.dumps(result), flush=True)
        if not result['success']:
            print('Invalid response preserved; no success cache written.', flush=True)


if __name__ == '__main__':
    main()
