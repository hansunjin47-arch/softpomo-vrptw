"""Generate one RCH01 BASE cache, then compare frozen policies with actual guidance."""
import copy
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import SelectiveAdvanceLLMModule as llm
import VRPTWLLMModule as backend
import train_soft_cluster as train
import train_vrptw
from extract_rl_routes import routes_to_cluster_json
from vrptw_env import load_solomon, _solution_stats


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    prior_path = ROOT / 'analysis/rc_fixed_gen_eval_20260920_205417/results.json'
    prior = json.loads(prior_path.read_text())
    ref = next(r for r in prior['results'] if r['model'] == 'pure500' and r['instance'] == 'RCH01')
    inst = load_solomon(str(ROOT.parent / 'data/Solomon/RCH01.txt'))
    assert sha(ROOT.parent / 'data/Solomon/RCH01.txt') == ref['input_sha256']
    assert all(sha(Path(m['path'])) == m['sha256'] for m in prior['models'].values())
    assert not inst['preset_events']
    clusters = [[int(n) for n in route if n] for route in ref['routes']]
    reference = routes_to_cluster_json(ref['routes'], inst)
    stats = {int(k): v for k, v in reference['_stats'].items()}
    objective = ('Minimize finite weighted cost, NOT lexicographic priority. '
                 'Serve all customers subject to capacity/vehicle constraints. '
                 'RC1 reward Config E parameters: ' + json.dumps(train_vrptw._REWARD_CONFIGS_RC1['E']) +
                 '. Late-stop count is reported but has zero direct penalty; total lateness is penalized. '
                 'Travel time includes event multipliers.')
    if not backend.OPENROUTER_API_KEY:
        raise RuntimeError('OpenRouter key unavailable; no request sent')
    out = ROOT / 'analysis' / ('rch01_proposed_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(exist_ok=False)
    manifest = dict(reference_source=str(prior_path), reference_sha256=sha(prior_path),
                    reference_routes=ref['routes'], models=prior['models'], results=[],
                    model='anthropic/claude-sonnet-4.6', beta=5, seed=42, pomo=100,
                    augmentation=8, amp=False, scope='RCH01 BASE only')
    (out / 'reference.json').write_text(json.dumps(reference, indent=2), encoding='utf-8')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('CALL RCH01 BASE', out, flush=True)
    cache = llm.generate_advance(inst, clusters, stats=stats, objective=objective,
                                 model=manifest['model'], output_dir=out / 'attempt1')
    llm.load_advance_cache(out / 'attempt1/cache.json', instance='RCH01', customer_ids=range(1, 101))
    assert cache['clusters'] == clusters
    runtime = out / 'runtime_cache'
    runtime.mkdir()
    shutil.copyfile(out / 'attempt1/cache.json', runtime / 'RCH01_advance.json')
    print('VALIDATED selected=', cache['advance'], flush=True)
    def reject(*args, **kwargs):
        raise RuntimeError('No additional LLM calls allowed during evaluation')
    llm.query_llm = reject
    backend.query_llm = reject
    train.get_all_clusters_confidence = reject
    tp = copy.deepcopy(train.trainer_params)
    tp.update(train_instances=['RCH01'], test_instances=['RCH01'],
              result_dir=str(out / 'runtime'), llm_cache_dir=str(runtime),
              cluster_mode='rl', guidance_mode='selective_advance', bias_mode='cluster',
              start_mode='random', pomo_size=100, model_load={'enable': False},
              test_only=True, experience_refresh_epoch=-1, benchmark='rc1', use_amp=False)
    lp = dict(train.llm_params, enabled=True, bias_strength=5., force_refresh=False)
    trainer = train.SoftClusterTrainer(dict(train.env_params, **train_vrptw._REWARD_CONFIGS_RC1['E']),
                copy.deepcopy(train.model_params), copy.deepcopy(train.optimizer_params), tp, lp)
    counts = {'calls': 0, 'nonzero_calls': 0}
    original = trainer._acc_conf_now
    def observed(*args, **kwargs):
        value = original(*args, **kwargs)
        counts['calls'] += 1
        counts['nonzero_calls'] += int(torch.count_nonzero(value).item() > 0)
        return value
    trainer._acc_conf_now = observed
    for label, source in prior['models'].items():
        checkpoint = torch.load(source['path'], map_location=trainer.device, weights_only=False)
        trainer.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        trainer.model.eval()
        trainer.llm_params['bias_strength'] = 5. if label == 'phase2' else 0.
        counts.update(calls=0, nonzero_calls=0)
        train._set_seed(42)
        with torch.no_grad():
            routes, reward = trainer._best_solution_aug(inst)
        routes = [[0] + [int(n) for n in r if n] + [0] for r in routes if any(r)]
        visited = [n for r in routes for n in r if n]
        assert len(visited) == len(set(visited))
        s = _solution_stats(inst, routes)
        row = dict(model=label, beta=trainer.llm_params['bias_strength'], vehicles=len(routes),
                   served=len(visited), distance=float(sum(v['rdist'] for v in s['per_vehicle'])),
                   late_stops=int(s['late_count']), total_late=float(s['total_late']),
                   customer_late=sum(x['late'] > 0 for v in s['per_vehicle'] for x in v['stop_details']),
                   depot_late=sum(v['ret_late'] > 0 for v in s['per_vehicle']),
                   reward=float(reward), routes=routes, bias_observation=dict(counts))
        row['customer_late'] = int(row['customer_late'])
        row['depot_late'] = int(row['depot_late'])
        if label != 'phase2':
            assert counts['nonzero_calls'] == 0
        elif cache['advance']:
            assert counts['nonzero_calls'] > 0
        manifest['results'].append(row)
        (out / 'results.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(json.dumps({k: v for k, v in row.items() if k != 'routes'}), flush=True)
        assert sha(Path(source['path'])) == source['sha256']
    print('DONE', out, flush=True)


if __name__ == '__main__':
    main()
