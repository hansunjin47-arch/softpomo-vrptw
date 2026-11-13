"""Link archived solution timings to cached reference clusters and scores."""
import collections
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE = ROOT / 'result_soft/llm_cache_final_nomisfit'


def solution(path):
    text = path.read_text(encoding='utf-8')
    routes = [[int(n) for n in m.split('->')][1:-1] for m in
              re.findall(r'^\s*V\d+:\s*([\d\s>\-]+)\s*\[', text, re.M)]
    timing = {}
    for line in text.splitlines():
        m = re.search(r'\d+\s*->\s*(\d+).*?arr=\s*([\d.]+).*?TW=\[\s*([\d.]+),\s*([\d.]+)\]', line)
        if m:
            n, arr, opening, close = map(float, m.groups())
            timing[int(n)] = dict(arr=arr, late=round(max(0, arr-close), 2))
    assert len(timing) == 100 and sum(map(len, routes)) == 100, path
    return routes, timing


def metrics(routes, clusters, score):
    membership = {n:k for k,ns in clusters.items() for n in ns}
    route_id = {n:i for i,r in enumerate(routes) for n in r}
    edges = [(a,b) for r in routes for a,b in zip(r,r[1:])]
    pairs = [(a,b) for r in routes for i,a in enumerate(r) for b in r[i+1:]
             if membership[a] == membership[b] and score[a] != score[b]]
    return dict(vehicles=len(routes),
                mixed_routes=sum(len({membership[n] for n in r})>1 for r in routes),
                cluster_purity=round(sum(max(collections.Counter(membership[n] for n in r).values()) for r in routes)/100,3),
                within_cluster_edges=sum(membership[a]==membership[b] for a,b in edges), edges=len(edges),
                split_clusters=sum(len({route_id[n] for n in ns})>1 for ns in clusters.values()),
                score_order_agree=sum(score[a]>score[b] for a,b in pairs), score_order_pairs=len(pairs))


def main():
    output = []
    for family,pool in [('r1','r102-r112'),('rc1','rc102-rc108'),('c1','c102-c109')]:
        prefix=family.replace('1','101').upper()
        for suffix in ['', '_RAIN2', '_ACC2']:
            name=prefix+suffix
            cache=json.loads((CACHE/f'{prefix if suffix=="_ACC2" else name}_rl_cluster.json').read_text())
            clusters={k:list(map(int,c)) for k,c in cache.items() if k.isdigit()}
            score={int(n):v for k,c in cache.items() if k.isdigit() for n,v in c.items()}
            stats={int(n):v for c in cache['_stats'].values() for n,v in c['node_detail'].items()}
            pr,pt=solution(ROOT/f'result/config_E_{family}_final_weightedsum_b32_aug8/{pool}/plots/{name}_solution.txt')
            sr,st=solution(ROOT/f'result_soft/{pool}/plots/{name}_solution.txt')
            nodes=[dict(node=n, score=score[n], reference_slack=stats[n]['slack'],
                        pure_late=pt[n]['late'], proposed_late=st[n]['late'],
                        delta=round(st[n]['late']-pt[n]['late'],2)) for n in score]
            row=dict(instance=name, reference_clusters=len(clusters),
                     reference_late=sum(c['total_late'] for c in cache['_stats'].values()),
                     pure=metrics(pr,clusters,score), proposed=metrics(sr,clusters,score),
                     newly_late=sum(n['pure_late']<=0.1 and n['proposed_late']>0.1 for n in nodes),
                     fixed_late=sum(n['pure_late']>0.1 and n['proposed_late']<=0.1 for n in nodes),
                     nodes=sorted(nodes,key=lambda n:n['delta'],reverse=True),
                     proposed_routes=sr, pure_routes=pr)
            output.append(row)
            print(json.dumps({k:v for k,v in row.items() if k not in ['nodes','proposed_routes','pure_routes']}))
            print('TOP',json.dumps(row['nodes'][:5]))
    (ROOT/'analysis/cache_route_comparison.json').write_text(json.dumps(output,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
