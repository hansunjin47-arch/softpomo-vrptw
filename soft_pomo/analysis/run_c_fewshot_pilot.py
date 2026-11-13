"""Offline verified demonstrations and a bounded Ollama prompt pilot. No training."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from vrptw_env import load_solomon, _solution_stats, VRPTWEnv, make_batch
from train_vrptw import env_params, _REWARD_CONFIGS
from SoftClusterLLMModule import build_all_clusters_prompt
from SoftClusterOntology import SoftClusterOntology

CACHE = ROOT / 'result_soft/llm_cache_twclose_c1'
CLARIFICATION = '''
Interpretation clarification: reference slack is TW_close minus arrival on the
REFERENCE route, not a context-free urgency label. A late-position customer with
small positive slack does not automatically deserve the highest confidence.
Consider waiting, service and harm to downstream customers before changing order.
Preserve a feasible reference order when there is no grounded improvement.
Confidence is a soft relative preference, not a calibrated probability.
'''


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def load(name):
    path = CACHE / f'{name}_rl_cluster.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    clusters = [list(map(int, data[str(k)])) for k in range(sum(k.isdigit() for k in data))]
    # Cache writer preserves decoded route order; independently check timing below.
    inst = load_solomon(str(ROOT.parent / f'data/Solomon/{name.lower()}.txt'))
    for k, ns in enumerate(clusters):
        replay = _solution_stats(inst, [[0] + ns + [0]])
        detail = data['_stats'][str(k)]['node_detail']
        for stop in replay['per_vehicle'][0]['stop_details']:
            assert abs(float(detail[str(stop['node'])]['slack']) -
                       (float(stop['tw_close']) - float(stop['arrival']))) < .1
    return inst, data, clusters


def metrics(inst, clusters):
    s = _solution_stats(inst, [[0] + ns + [0] for ns in clusters])
    return {'cost': 6 * len(clusters) / inst['n_customers'] +
            9 * float(s['total_late']) / float(inst['T']) / 5 +
            3 * float(s['total_travel_time']) / float(inst['T']) / 6,
            'late': float(s['total_late']), 'travel': float(s['total_travel_time']),
            'late_count': int(s['late_count'])}


def verify_env(inst, clusters, expected):
    env = VRPTWEnv(**dict(env_params, **_REWARD_CONFIGS['E'], pomo_size=1))
    env.load_problems(make_batch(inst, 1, torch.device('cpu')))
    env.reset()
    env.step(torch.tensor([[0]]))
    for ns in clusters:
        for n in ns + [0]:
            _, reward, done = env.step(torch.tensor([[n]]))
    assert done and reward is not None
    error = abs(float(reward.item()) + expected['cost'])
    assert error < .002, (error, reward, expected)
    return error


def prompt(inst, clusters, stats, examples=''):
    ont = SoftClusterOntology(inst)
    contexts = [ont.get_cluster_context(ns, k) for k, ns in enumerate(clusters)]
    return build_all_clusters_prompt(inst, clusters, contexts, use_cot=True,
        use_ontology=True, reward_config='E', bks_fewshot=examples,
        stats=stats, use_misfit=False)


def demonstrations():
    bank, blocks = [], []
    for name, k in [('C108', 5), ('C105', 0)]:
        inst, data, clusters = load(name)
        ref = metrics(inst, clusters)
        err = verify_env(inst, clusters, ref)
        ns = clusters[k]
        candidates = set()
        for i in range(len(ns)):
            for j in range(len(ns)):
                seq = ns.copy()
                seq.insert(j, seq.pop(i))
                candidates.add(tuple(seq))
                seq = ns.copy()
                seq[i], seq[j] = seq[j], seq[i]
                candidates.add(tuple(seq))
        evaluated = []
        for seq in sorted(candidates):
            cs = [r.copy() for r in clusters]
            cs[k] = list(seq)
            evaluated.append((metrics(inst, cs), list(seq)))
        best_m, best = min(evaluated, key=lambda x: x[0]['cost'])
        chosen = best if ref['cost'] - best_m['cost'] > .002 else ns
        changed = [r.copy() for r in clusters]
        changed[k] = chosen
        verify_env(inst, changed, metrics(inst, changed))
        urgent = min(ns, key=lambda n: data['_stats'][str(k)]['node_detail'][str(n)]['slack'])
        first = [urgent] + [n for n in ns if n != urgent]
        cs = [r.copy() for r in clusters]
        cs[k] = first
        urgent_m = metrics(inst, cs)
        scores = {str(n): round(.8 - .4 * i / max(1, len(chosen)-1), 4)
                  for i, n in enumerate(chosen)}
        scores['0'] = 0.0
        reason = (f'Reference lateness={ref["late"]:.3f}. Moving lowest-slack '
                  f'customer {urgent} first gives total lateness={urgent_m["late"]:.3f}. '
                  f'Local swap/relocate search cost: {ref["cost"]:.6f} -> {best_m["cost"]:.6f}. '
                  'Scores encode the selected relative order heuristically, not true probabilities; '
                  'no global optimality or depot decision is established.')
        p = prompt(inst, [ns], {0: data['_stats'][str(k)]})
        blocks.append(f'### Verified demonstration {name}, original cluster {k}\nINPUT:\n{p}\nOUTPUT:\n' +
                      json.dumps({'confidence': {'1': scores}, 'reason': reason}))
        bank.append(dict(instance=name, cluster=k, reference=ns, selected=chosen,
                         reference_metrics=ref, best_metrics=best_m,
                         lowest_slack_first=urgent_m, lowest_slack_node=urgent,
                         candidates=[dict(route=r, **m) for m, r in evaluated],
                         env_cost_error=err, score_encoding='0.8-0.4*rank/(m-1)'))
    return bank, '\n'.join(blocks)


def strict_response(raw, clusters):
    def pairs(items):
        d = {}
        for k, v in items:
            if k in d:
                raise ValueError(f'Duplicate JSON key: {k}')
            d[k] = v
        return d
    text = re.sub(r'<think>.*?</think>', '', raw, flags=re.S).strip()
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text).strip()
    obj = json.loads(text, object_pairs_hook=pairs)
    conf = obj['confidence']
    assert set(conf) == {str(k+1) for k in range(len(clusters))}
    for k, ns in enumerate(clusters):
        values = conf[str(k+1)]
        assert set(values) == set(map(str, [0]+ns)), 'coverage'
        assert all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1
                   for v in values.values()), 'score range/type'
    return conf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--instance', default='C102', choices=['C102', 'C103', 'C101'])
    parser.add_argument('--model', default='gpt-oss:20b')
    parser.add_argument('--num-predict', type=int, default=8192)
    args = parser.parse_args()
    out = ROOT / 'result_soft/fewshot_c_pilot' / time.strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    print('OUTPUT', out, flush=True)
    bank, examples = demonstrations()
    save(out/'teacher_audit.json', bank)
    (out/'fewshot_bank.txt').write_text(examples, encoding='utf-8')
    inst, data, clusters = load(args.instance)
    stats = {int(k): v for k, v in data['_stats'].items()}
    prompts = {'P0': prompt(inst, clusters, stats),
               'P1': prompt(inst, clusters, stats) + CLARIFICATION,
               'P2': prompt(inst, clusters, stats, examples) + CLARIFICATION}
    config = dict(model=args.model, stream=False, think=False,
                  options=dict(temperature=0.0, num_ctx=32768, num_predict=args.num_predict))
    manifest = dict(demonstrations=['C108', 'C105'], validation=[args.instance],
                    config=config, settings_note='Controlled pilot, not historical-cache replication',
                    retraining=False, retries=0, results={}, source_hashes={})
    for name in ['C108', 'C105', args.instance]:
        p = CACHE/f'{name}_rl_cluster.json'
        manifest['source_hashes'][str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    for arm, p in prompts.items():
        (out/f'{arm}_prompt.txt').write_text(p, encoding='utf-8')
        manifest.setdefault('prompt_hashes', {})[arm] = hashlib.sha256(p.encode()).hexdigest()
    save(out/'manifest.json', manifest)
    for arm, p in prompts.items():
        if not args.generate:
            continue
        print('CALL', arm, 'chars', len(p), flush=True)
        start = time.time()
        try:
            request = urllib.request.Request('http://localhost:11434/api/generate',
                data=json.dumps(dict(config, prompt=p)).encode(),
                headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request, timeout=900) as response:
                raw = json.load(response)
            save(out/f'{arm}_raw.json', raw)
            assert raw.get('done') is True and raw.get('done_reason') == 'stop', 'incomplete generation'
            conf = strict_response(raw['response'], clusters)
            cache = dict(data)
            for k, ns in enumerate(clusters):
                cache[str(k)] = {str(n): conf[str(k+1)][str(n)] for n in ns}
            cache['_depot'] = {str(k): conf[str(k+1)]['0'] for k in range(len(clusters))}
            save(out/f'{arm}_{args.instance}_rl_cluster.json', cache)
            orders = [sorted(ns, key=lambda n: -conf[str(k+1)][str(n)])
                      for k, ns in enumerate(clusters)]
            values = [conf[str(k+1)][str(n)] for k, ns in enumerate(clusters) for n in ns]
            agree = sum(conf[str(k+1)][str(a)] > conf[str(k+1)][str(b)]
                        for k, ns in enumerate(clusters) for i,a in enumerate(ns) for b in ns[i+1:])
            ties = sum(conf[str(k+1)][str(a)] == conf[str(k+1)][str(b)]
                        for k, ns in enumerate(clusters) for i,a in enumerate(ns) for b in ns[i+1:])
            pairs = sum(len(ns)*(len(ns)-1)//2 for ns in clusters)
            result = dict(success=True, score_mean=sum(values)/len(values),
                score_min=min(values), score_max=max(values), reference_pair_agree=agree,
                pair_ties=ties, pairs=pairs, rank_replay=metrics(inst, orders),
                reference=metrics(inst, clusters),
                warning='Score-sorted route replay is NOT policy inference; ties preserve reference order.')
        except Exception as exc:
            result = dict(success=False, error=f'{type(exc).__name__}: {exc}')
        result['elapsed_seconds'] = time.time()-start
        manifest['results'][arm] = result
        save(out/'manifest.json', manifest)
        print(arm, json.dumps(result), flush=True)
    for path, digest in manifest['source_hashes'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest


if __name__ == '__main__':
    main()
