"""Compare saved routes across Pure1000, v2 and matched without training."""
import json
from link_cache_routes import ROOT, CACHE, solution, metrics


def main():
    rows = []
    for family, pool in [('r1','r102-r112'), ('rc1','rc102-rc108'), ('c1','c102-c109')]:
        prefix = family.replace('1','101').upper()
        base_routes = {}
        for suffix in ['', '_ACC2', '_RAIN2']:
            name = prefix + suffix
            d = json.loads((CACHE / f'{prefix if suffix == "_ACC2" else name}_rl_cluster.json').read_text())
            clusters = {k:list(map(int,v)) for k,v in d.items() if k.isdigit()}
            scores = {int(n):v for k,c in d.items() if k.isdigit() for n,v in c.items()}
            reference = {int(n):v for c in d['_stats'].values() for n,v in c['node_detail'].items()}
            row = {'instance':name}
            for label, folder in [('pure', f'result/config_E_{family}_final_weightedsum_b32_aug8'),
                                  ('v2','result_soft'), ('matched','result_soft_matched/with_bias')]:
                routes, timing = solution(ROOT / folder / pool / 'plots' / f'{name}_solution.txt')
                if not suffix:
                    base_routes[label] = routes
                m = metrics(routes, clusters, scores)
                m['same_routes_as_base'] = routes == base_routes[label]
                m['late_customers'] = sum(v['late'] > .1 for v in timing.values())
                m['late_with_positive_reference_slack'] = sum(v['late'] > .1 and reference[n]['slack'] >= 0 for n,v in timing.items())
                m['customer_late_sum'] = round(sum(v['late'] for v in timing.values()),2)
                if label == 'pure':
                    pure_timing = timing
                m['newly_late_vs_pure'] = sum(v['late'] > .1 and pure_timing[n]['late'] <= .1 for n,v in timing.items())
                row[label] = m
            rows.append(row)
            print(json.dumps(row),flush=True)
    (ROOT/'analysis/matched_route_comparison.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
