"""Audit saved GPT guidance without new API calls or simulator feedback."""
import collections
import hashlib
import json
from pathlib import Path
import shutil
import statistics as st

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis/gpt_guidance_cache_audit_20261015'
RUNS = {
    'kim_late1': 'rc101_kim_train20_bks_customer_only_gpt54_v1',
    'kim_late10': 'rc101_kim_train20_bks_late10_gpt54_v1',
    'phase2_late5': 'rc101_phase2_train20_bks_late5_gpt54_v1',
    'phase2_late10': 'rc101_phase2_train20_bks_late10_gpt54_v1',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    OUT.mkdir(exist_ok=False)
    report, records, manifest = {}, {}, {}
    for label, run in RUNS.items():
        for split, suffix in [('train', ''), ('test', 'test80')]:
            source = ROOT / 'result' / run / suffix
            paths = sorted((source / 'cache').glob('*.json'))
            if not paths:
                continue
            key = label + '_' + split
            protocol = read(source / 'protocol.json')
            dest = OUT / 'caches' / key
            dest.mkdir(parents=True)
            rows, vectors, groups_by_name = [], {}, {}
            for path in paths:
                name = path.name.replace('_rl_cluster.json', '').replace('_cluster.json', '')
                cache = read(path)
                request_folder = source / 'requests' / name
                request = read(request_folder / 'request.json')
                assert request['model'] == 'openai/gpt-5.4'
                groups = protocol.get('clusters') or read(request_folder / 'reference.json')['clusters']
                flat = {int(n): float(v) for g, nodes in cache.items() if g != '_depot' for n, v in nodes.items()}
                assert len(flat) == sum(len(nodes) for g, nodes in cache.items() if g != '_depot') == 100
                assert sorted(flat) == list(range(1, 101))
                assert all(np.isfinite(v) and 0 <= v <= 1 for v in flat.values())
                assert all(v == 0 for v in cache['_depot'].values())
                for k, group in enumerate(groups):
                    assert set(cache[str(k)]) == set(map(str, group))
                vec = [flat[n] for n in range(1, 101)]
                vectors[name], groups_by_name[name] = vec, groups
                pair_total = forward = tied = 0
                monotonic, ranges = [], []
                for group in groups:
                    vals = [flat[n] for n in group]
                    ranges.append(max(vals)-min(vals))
                    if len(vals) > 1:
                        monotonic.append(all(a >= b for a, b in zip(vals, vals[1:])))
                    for i, a in enumerate(vals):
                        for b in vals[i+1:]:
                            pair_total += 1
                            forward += a > b
                            tied += a == b
                usage = read(request_folder / 'api_metadata/response.json').get('usage', {})
                row = dict(name=name, mean=st.mean(vec), std=st.pstdev(vec),
                    minimum=min(vec), maximum=max(vec), unique_scores=len(set(vec)),
                    zero_count=vec.count(0), group_range=st.mean(ranges),
                    monotonic_group_fraction=st.mean(monotonic),
                    reference_order_pair_fraction=forward / pair_total,
                    reference_order_tie_fraction=tied / pair_total,
                    cost=usage.get('cost', 0), output_tokens=usage.get('completion_tokens'),
                    reasoning_tokens=usage.get('completion_tokens_details', {}).get('reasoning_tokens'),
                    cached_tokens=usage.get('prompt_tokens_details', {}).get('cached_tokens', 0))
                rows.append(row)
                shutil.copy2(path, dest / path.name)
                manifest[str((dest / path.name).relative_to(OUT))] = dict(source=str(path),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    request=str(request_folder / 'request.json'))
            counter = collections.Counter(tuple(v) for v in vectors.values())
            matrix = np.array(list(vectors.values()))
            report[key] = dict(count=len(rows),
                averages={k: st.mean(r[k] for r in rows) for k in ('mean', 'std', 'unique_scores',
                    'zero_count', 'group_range', 'monotonic_group_fraction', 'reference_order_pair_fraction',
                    'reference_order_tie_fraction')},
                unique_customer_score_vectors=len(counter), largest_identical_vector_group=max(counter.values()),
                mean_customer_std_across_scenarios=float(matrix.std(axis=0).mean()),
                total_cost=sum(r['cost'] for r in rows),
                cached_calls=sum(r['cached_tokens'] > 0 for r in rows),
                output_token_counts=dict(collections.Counter(r['output_tokens'] for r in rows)),
                reasoning_token_counts=dict(collections.Counter(r['reasoning_tokens'] for r in rows)), rows=rows)
            records[key] = (vectors, groups_by_name)
    pairs = {}
    for left, right in [('kim_late1_train', 'kim_late10_train'),
                        ('phase2_late5_train', 'phase2_late10_train'),
                        ('phase2_late5_test', 'phase2_late10_test')]:
        a, ga = records[left]
        b, gb = records[right]
        names = sorted(set(a) & set(b))
        pairs[left + '_vs_' + right] = dict(count=len(names),
            exact_score_vectors=sum(a[n] == b[n] for n in names),
            exact_group_lists=sum(ga[n] == gb[n] for n in names),
            mean_customer_score_mae=st.mean(float(np.abs(np.array(a[n])-b[n]).mean()) for n in names),
            mean_spearman=st.mean(float(spearmanr(a[n], b[n]).statistic) for n in names))
    (OUT / 'summary.json').write_text(json.dumps(dict(experiments=report, paired=pairs), indent=2), encoding='utf-8')
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(dict(experiments={k: {a: b for a, b in v.items() if a != 'rows'} for k,v in report.items()}, paired=pairs), indent=2))


if __name__ == '__main__':
    main()
