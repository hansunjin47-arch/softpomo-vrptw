"""Compare saved route partitions and orders; no training or API calls."""
import hashlib
import itertools
import json
from pathlib import Path
import statistics as st
import tarfile
from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment
from analyze_ontology_v5_results import FROZEN, load_solomon, replay

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis/saved_route_structure.json'


def routes(record):
    rr = [[int(x) for x in r if x != 0] for r in record['routes']]
    rr = [r for r in rr if r]
    assert sorted(itertools.chain.from_iterable(rr)) == list(range(1, 101))
    return rr


def structure(rr):
    pairs = {}
    edges = set()
    for r in rr:
        for i, j in itertools.combinations(r, 2):
            pairs[tuple(sorted((i, j)))] = i < j
        edges.update(zip([0] + r, r + [0]))
    return pairs, edges


def compare(a, b):
    ra, rb = routes(a), routes(b)
    pa, ea = structure(ra)
    pb, eb = structure(rb)
    common = pa.keys() & pb.keys()
    weights = np.array([[len(set(x) & set(y)) for y in rb] for x in ra])
    rows, cols = linear_sum_assignment(-weights)
    changes = [dict(before=ra[i], after=rb[j]) for i, j in zip(rows, cols)
               if ra[i] != rb[j]]
    return dict(
        pair_retention=len(common) / len(pa),
        pair_jaccard=len(common) / len(pa.keys() | pb.keys()),
        split_pairs=len(pa.keys() - pb.keys()), joined_pairs=len(pb.keys() - pa.keys()),
        common_pairs=len(common), reversed_pairs=sum(pa[k] != pb[k] for k in common),
        reversed_fraction=sum(pa[k] != pb[k] for k in common) / len(common) if common else 0,
        reassigned_min=100-int(weights[rows, cols].sum()),
        identical_customer_sets=sum(frozenset(r) in {frozenset(s) for s in rb} for r in ra),
        identical_ordered_routes=sum(tuple(r) in {tuple(s) for s in rb} for r in ra),
        edge_jaccard=len(ea & eb) / len(ea | eb),
        route_changes=changes,
        unmatched_before=[r for i, r in enumerate(ra) if i not in rows],
        unmatched_after=[r for i, r in enumerate(rb) if i not in cols],
        delta={k: b[k]-a[k] for k in ('vehicles', 'travel_time', 'late_count', 'total_late', 'cost')})


def main():
    archive = ROOT / 'server_export/train20_seed43_results.tar.gz'
    member = 'train20/runs/pure128_4000_seed43/eval_1000_test80/results.json'
    with tarfile.open(archive) as tf:
        raw = tf.extractfile(member).read()
    data = {'pure': json.loads(raw)}
    hashes = {'pure': hashlib.sha256(raw).hexdigest()}
    for key, folder in [('scale5', 'rc101_ontology_v5_b128_s44_scale5'),
                        ('scale10', 'rc101_ontology_v5_b128_s44_scale10_diagnostic')]:
        raw = (ROOT / 'result' / folder / 'eval_1000_test80/results.json').read_bytes()
        hashes[key] = hashlib.sha256(raw).hexdigest()
        data[key] = json.loads(raw)
    data = {k: {r['name']: r for r in records} for k, records in data.items()}
    assert all(set(d) == set(data['pure']) and len(d) == 80 for d in data.values())
    replay_rows = {k: [] for k in data}
    for name in data['pure']:
        inst = load_solomon(str(FROZEN / 'eval_addon/data' / (name + '.txt')))
        for k, d in data.items():
            row, _ = replay(inst, d[name])
            replay_rows[k].append(row)
    metrics = {k: {m: st.mean(r[m] for r in d.values()) for m in
                  ('vehicles', 'travel_time', 'late_count', 'total_late', 'cost')}
               for k, d in data.items()}
    for k in metrics:
        metrics[k].update({m: st.mean(r[m] for r in replay_rows[k]) for m in
                           ('base_travel', 'event_travel', 'customer_late', 'depot_late')})
    comparisons = {}
    for ka, kb in [('pure', 'scale10'), ('pure', 'scale5'), ('scale10', 'scale5')]:
        cases = {n: compare(data[ka][n], data[kb][n]) for n in data[ka]}
        summary = {k: st.mean(c[k] for c in cases.values()) for k in next(iter(cases.values()))
                   if k not in ('delta', 'route_changes', 'unmatched_before', 'unmatched_after')}
        summary['pooled_reversed_fraction'] = (sum(c['reversed_pairs'] for c in cases.values()) /
                                              sum(c['common_pairs'] for c in cases.values()))
        summary['lateness_improved_tied_worse'] = [sum((c['delta']['total_late'] < -1e-7,
            abs(c['delta']['total_late']) <= 1e-7, c['delta']['total_late'] > 1e-7)[i]
            for c in cases.values()) for i in range(3)]
        summary['travel_improved'] = sum(c['delta']['travel_time'] < 0 for c in cases.values())
        summary['largest_lateness_increases'] = [dict(name=n, **cases[n]) for n in
            sorted(cases, key=lambda n: cases[n]['delta']['total_late'], reverse=True)[:3]]
        changes = Counter((tuple(x['before']), tuple(x['after'])) for c in cases.values()
                          for x in c['route_changes'])
        summary['recurring_changes'] = [dict(before=a, after=b, count=v)
                                       for (a, b), v in changes.most_common(5)]
        comparisons[ka+'->'+kb] = dict(summary=summary, cases=cases)
    result = dict(scope='Descriptive model comparison, not isolated confidence effect. '
                  'Reassignment uses maximum-overlap one-to-one route matching; '
                  'pair reversal only compares customers co-routed in both solutions.',
                  hashes=hashes, metrics=metrics, comparisons=comparisons)
    OUT.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(dict(metrics=metrics, comparisons={k: {x: y for x, y in v['summary'].items()
        if x != 'largest_lateness_increases'} for k, v in comparisons.items()}), indent=2))


if __name__ == '__main__':
    main()
