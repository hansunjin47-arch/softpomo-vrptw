"""Audit completed followup routes without API calls or training changes."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics as st

from analyze_ontology_v5_results import (
    ROOT, FROZEN, read, replay, guidance_details, summarize, scalar, load_solomon,
)

OUT = ROOT / 'result/rc101_ontology_v5_followups_b128'


def main():
    inventory = read(OUT / 'results_summary.json')['runs']
    details, summaries, hashes = {}, {}, {}
    names = None
    for run in inventory:
        key = f"s{run['seed']}_e{run['epoch']}"
        path = Path(run['source']) / 'results.json'
        raw = read(path)
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        current = [r['name'] for r in raw]
        assert len(current) == len(set(current)) == 80
        if names is None:
            names = current
        assert names == current
        details[key] = {}
        rows, late_nodes, patterns = [], Counter(), Counter()
        for record in raw:
            name = record['name']
            inst = load_solomon(str(FROZEN / 'eval_addon/data' / (name + '.txt')))
            metrics, stats = replay(inst, record)
            obs = read(FROZEN / 'eval_addon/data' / (name + '.observations.json'))
            guidance = guidance_details(name, stats, obs)
            for visit in guidance['visits']:
                if visit['late'] > 1e-7:
                    late_nodes[visit['node']] += 1
                    route = guidance['routes'][visit['route']]['stops']
                    patterns[(visit['node'], tuple(route), round(visit['late'], 6))] += 1
            details[key][name] = dict(metrics=metrics, guidance=guidance,
                                     vehicles=stats['per_vehicle'])
            rows.append(metrics)
        summaries[key] = dict(metrics=summarize(rows), late_nodes=late_nodes.most_common(),
            recurring_late_routes=[dict(node=k[0], route=k[1], late=k[2], count=v)
                                   for k, v in patterns.most_common(12)])
    comparisons = {}
    for seed in sorted(set(r['seed'] for r in inventory)):
        early, late = f's{seed}_e1000', f's{seed}_e2000'
        if early not in details or late not in details:
            continue
        pairs = [(details[early][n]['metrics'], details[late][n]['metrics']) for n in names]
        comparisons[str(seed)] = dict(
            new_late=[b['name'] for a, b in pairs if a['total_late'] <= 1e-7 < b['total_late']],
            biggest_increases=[dict(name=a['name'], before=a, after=b)
                for a, b in sorted(pairs, key=lambda p: p[1]['total_late']-p[0]['total_late'], reverse=True)[:8]],
            deltas={k: st.mean(b[k]-a[k] for a, b in pairs)
                    for k in ('vehicles','travel_time','late_count','total_late','customer_late','depot_late','base_late','cost')})
    report = dict(scope='Descriptive saved-route replay, not action-logit or causal attribution.',
                  summaries=summaries, comparisons=comparisons, scenarios=details, source_sha256=hashes)
    output = OUT / 'route_audit.json'
    output.write_text(json.dumps(report, indent=2, default=scalar, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(summaries=summaries, comparisons=comparisons), indent=2, default=scalar))


if __name__ == '__main__':
    main()
