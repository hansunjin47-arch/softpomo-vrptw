"""Compare reference-slack pair relations with replay-verified order preferences.

No API calls, model inference, cache writes, or training. Pair preferences are
conditional on the tested complete routes, not ground-truth confidence labels.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from run_c_fewshot_pilot import ROOT, CACHE, load, metrics, verify_env
from vrptw_env import _solution_stats

EPS = 0.002
DEVELOPMENT = {'C102', 'C103'}


def reversed_routes(route, i, j):
    swap = route.copy()
    swap[i], swap[j] = swap[j], swap[i]
    earlier = route.copy()
    earlier.insert(i, earlier.pop(j))
    later = route.copy()
    node = later.pop(i)
    later.insert(j, node)
    return sorted(set(map(tuple, [swap, earlier, later])))


def main():
    out = ROOT / 'analysis/c_pair_audit' / time.strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    rows, summaries, sources = [], {}, {}
    instances = {}
    for number in range(102, 110):
        name = f'C{number}'
        inst, data, clusters = load(name)
        instances[name] = (inst, clusters)
        for path in [CACHE / f'{name}_rl_cluster.json',
                     ROOT.parent / f'data/Solomon/{name.lower()}.txt']:
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        reference = metrics(inst, clusters)
        error = verify_env(inst, clusters, reference)
        counts = Counter()
        for k, route in enumerate(clusters):
            stats = data['_stats'][str(k)]['node_detail']
            local_ref = metrics(inst, [route])
            memo = {}
            for i, first in enumerate(route):
                for j in range(i+1, len(route)):
                    second = route[j]
                    s1, s2 = [float(stats[str(n)]['slack']) for n in (first, second)]
                    counts['all_pairs'] += 1
                    if min(s1, s2) <= 0 or abs(s1-s2) < 1e-5:
                        counts['excluded_nonpositive_or_equal_slack'] += 1
                        continue
                    a, b = (first, second) if s1 < s2 else (second, first)
                    values = data[str(k)]
                    confidence_relation = ('smaller_slack_higher' if values[str(a)] > values[str(b)]
                        else 'larger_slack_higher' if values[str(a)] < values[str(b)] else 'tie')
                    alternatives = []
                    for candidate in reversed_routes(route, i, j):
                        if candidate not in memo:
                            local = metrics(inst, [list(candidate)])
                            memo[candidate] = {key: reference[key] + local[key] - local_ref[key]
                                               for key in reference}
                        alternatives.append(dict(route=list(candidate), metrics=memo[candidate]))
                    best = min(alternatives, key=lambda x: x['metrics']['cost'])
                    delta = reference['cost'] - best['metrics']['cost']
                    winner_first = second if delta > EPS else first if delta < -EPS else None
                    preference = ('inconclusive' if winner_first is None else
                                  'smaller_slack_first' if winner_first == a else 'larger_slack_first')
                    counts['confidence:'+confidence_relation] += 1
                    counts['preference:'+preference] += 1
                    counts['change:'+('improve' if delta > EPS else 'harm' if delta < -EPS else 'tie')] += 1
                    row = dict(instance=name, cluster=k, a_smaller_slack=a, b_larger_slack=b,
                        slack_a=min(s1,s2), slack_b=max(s1,s2), confidence_a=values[str(a)],
                        confidence_b=values[str(b)], confidence_relation=confidence_relation,
                        reference_route=route, reference_metrics=reference,
                        alternative_routes=alternatives, best_reversed_route=best['route'],
                        best_reversed_metrics=best['metrics'], delta_cost=delta,
                        supported_order=preference, development_only=name in DEVELOPMENT,
                        adjacent=j==i+1)
                    rows.append(row)
        summaries[name] = dict(counts=counts, reference=reference, env_cost_error=error)
        print(name, json.dumps(summaries[name]), flush=True)
    # Prefer adjacent pairs to reduce changes involving third-party customers.
    # Select without consulting development-instance results or policy performance.
    selected = []
    for kind, change in [('smaller_slack_first', 'preserve'),
                         ('larger_slack_first', 'preserve'),
                         ('smaller_slack_first', 'improve'),
                         ('larger_slack_first', 'improve')]:
        pool = [r for r in rows if not r['development_only'] and r['supported_order'] == kind
                and (r['delta_cost'] > EPS if change == 'improve' else r['delta_cost'] < -EPS)]
        aligned_relation = kind.replace('_first', '_higher')
        pool.sort(key=lambda r: (not r['adjacent'], r['confidence_relation'] == aligned_relation,
                                 -abs(r['delta_cost']),
                                 r['instance'], r['cluster'], r['a_smaller_slack']))
        used = set()
        for row in pool:
            key = (row['instance'], row['cluster'])
            if key in used:
                continue
            used.add(key)
            row['selection_category'] = change
            inst, clusters = instances[row['instance']]
            changed = [r.copy() for r in clusters]
            changed[row['cluster']] = row['best_reversed_route']
            row['alternative_env_error'] = verify_env(inst, changed, row['best_reversed_metrics'])
            replay = _solution_stats(inst, [[0]+row['reference_route']+[0]])
            altered = _solution_stats(inst, [[0]+row['best_reversed_route']+[0]])
            row['reference_stop_details'] = replay['per_vehicle'][0]['stop_details']
            row['alternative_stop_details'] = altered['per_vehicle'][0]['stop_details']
            selected.append(row)
            if len(used) == 2:
                break
    result = dict(threshold=EPS, summaries=summaries, selected=selected, pairs=rows,
        source_hashes=sources, development_instances=sorted(DEVELOPMENT),
        caveats=['Reference slack is fixed, not recalculated for pair classification.',
                 'Restricted swap/relocate comparison, not global optimality.',
                 'Cost includes entire instance, depot lateness and reward E normalization.',
                 'A route-order preference is not a calibrated confidence label.',
                 'C101 excluded. No synthetic perturbed reference examples.'])
    def default(value):
        if hasattr(value, 'item'):
            return value.item()
        raise TypeError(type(value).__name__)
    (out/'audit.json').write_text(json.dumps(result, indent=2, default=default), encoding='utf-8')
    lines = ['# C BASE Pair-Preference Audit', '',
        'No LLM calls or training. C101 excluded; C102/C103 reserved for prompt development.', '',
        '| Instance | Eligible pairs | Smaller slack first | Larger slack first | Inconclusive | Improving reversals |',
        '|---|---:|---:|---:|---:|---:|']
    for name, summary in summaries.items():
        c = summary['counts']
        lines.append(f'| {name} | {c["all_pairs"]-c["excluded_nonpositive_or_equal_slack"]} | '
                     f'{c["preference:smaller_slack_first"]} | {c["preference:larger_slack_first"]} | '
                     f'{c["preference:inconclusive"]} | {c["change:improve"]} |')
    lines += ['', '## Selected Contrast Candidates', '',
        'A has smaller reference slack than B. All metrics below are whole-instance totals.']
    for r in selected:
        lines += ['', f'### {r["instance"]} cluster {r["cluster"]} (zero-based)',
            f'- A={r["a_smaller_slack"]}: slack={r["slack_a"]:.3f}, confidence={r["confidence_a"]}',
            f'- B={r["b_larger_slack"]}: slack={r["slack_b"]:.3f}, confidence={r["confidence_b"]}',
            f'- Reference: {r["reference_route"]}',
            f'- Best reversed candidate: {r["best_reversed_route"]}',
            f'- Supported direction: {r["supported_order"]}; adjacent={r["adjacent"]}',
            f'- Cost: {r["reference_metrics"]["cost"]:.6f} -> {r["best_reversed_metrics"]["cost"]:.6f}',
            f'- Lateness: {r["reference_metrics"]["late"]:.3f} -> {r["best_reversed_metrics"]["late"]:.3f}',
            f'- Travel: {r["reference_metrics"]["travel"]:.3f} -> {r["best_reversed_metrics"]["travel"]:.3f}']
    lines += ['', '## Limits', ''] + ['- '+s for s in result['caveats']]
    (out/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    for path, digest in sources.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    print('OUTPUT', out, flush=True)


if __name__ == '__main__':
    main()
