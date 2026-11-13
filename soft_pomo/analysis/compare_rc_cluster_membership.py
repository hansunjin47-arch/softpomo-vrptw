"""Compare partitions without interpreting arbitrary cluster labels as identity."""
import json
from pathlib import Path
from collections import Counter
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vrptw_env import load_solomon
OLD=ROOT/'result_soft/llm_cache_kim_initial_v1_20260917_214827/rc1'
NEW=ROOT/'result_soft/llm_cache_kim500_sonnet46_rc1_20260918_123744'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def groups(d):
    return [list(map(int,d[str(k)])) for k in range(len(d['_depot']))]


def pairs(cs):
    return {tuple(sorted((a,b))) for ns in cs for i,a in enumerate(ns) for b in ns[i+1:]}


def compare(old,new,inst):
    oldmap={n:k for k,ns in enumerate(old) for n in ns}
    newmap={n:k for k,ns in enumerate(new) for n in ns}
    assert sorted(oldmap)==sorted(newmap)==list(range(1,101))
    op,np=pairs(old),pairs(new)
    parts=[dict(new_cluster=k+1,route=ns,old_sources={str(j+1):[n for n in ns if oldmap[n]==j] for j in sorted({oldmap[n] for n in ns})}) for k,ns in enumerate(new)]
    loads=lambda cs:[round(sum(float(inst['node_demand'][n-1])*float(inst['vehicle_capacity']) for n in ns),3) for ns in cs]
    ol,nl=loads(old),loads(new)
    return dict(old_k=len(old),new_k=len(new),old_sizes=sorted(map(len,old)),new_sizes=sorted(map(len,new)),
        exact_groups=sum(set(ns) in [set(o) for o in old] for ns in new),
        old_split=sum(len({newmap[n] for n in ns})>1 for ns in old),
        new_mixed=sum(len(p['old_sources'])>1 for p in parts),
        old_pairs=len(op),new_pairs=len(np),retained_pairs=len(op&np),
        pair_retention=len(op&np)/len(op),new_pair_overlap=len(op&np)/len(np),
        dominant_source_fraction=sum(max(Counter(oldmap[n] for n in ns).values()) for ns in new)/100,
        old_loads=ol,new_loads=nl,capacity=float(inst['vehicle_capacity']),
        old_overcapacity=sum(x>float(inst['vehicle_capacity'])+1e-3 for x in ol),
        new_overcapacity=sum(x>float(inst['vehicle_capacity'])+1e-3 for x in nl),details=parts)


def main():
    records=[]
    for i in range(101,109):
        name=f'RC{i}'
        folder=NEW/('evaluation' if i==101 else 'training')
        for suffix in ['', '_RAIN2']:
            old=groups(read(OLD/f'{name}{suffix}_cluster.json'))
            new=read(folder/f'{name}{suffix}_reference.json')['routes']
            inst=load_solomon(str(ROOT.parent/f'data/Solomon/{name.lower()}{suffix.lower()}.txt'))
            records.append(dict(name=name,scenario=suffix or 'BASE',**compare(old,new,inst)))
    (ROOT/'analysis/rc_cluster_membership_comparison.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    lines=['# RC Kim vs Epoch-500 Route Membership','',
        'Cluster indices are arbitrary. Exact matches ignore visit order. Pair retention is the fraction of originally co-clustered customer pairs that remain together. It is not a customer-movement percentage.',
        '','| Instance | Scenario | Old K | New K | Exact groups | Old groups split | New mixed groups | Pair retention |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append(f"| {r['name']} | {r['scenario']} | {r['old_k']} | {r['new_k']} | {r['exact_groups']} | {r['old_split']} | {r['new_mixed']} | {r['pair_retention']:.1%} |")
    lines+=['','## Provenance','',
        'Old: original Kim v1 cache actually used for training, not capacity-rerun clusters.',
        'New: aug-8 reference routes extracted from the Kim-guided epoch-500 checkpoint for Sonnet generation. Not every training episode or a fresh Kim clustering.',
        'ACC caches keep BASE membership and order in both versions; only wave confidence changes. No separate post-accident optimized membership is represented.',
        'RC101 is held-out evaluation; RC102-RC108 are training instances. No routing-quality conclusion follows from changed membership alone.']
    for r in records:
        lines += ['',f"## {r['name']} {r['scenario']}",'',f"Sizes: {r['old_sizes']} -> {r['new_sizes']}"]
        for p in r['details']:
            lines.append(f"- New {p['new_cluster']}: route {p['route']}; old sources {p['old_sources']}")
    (ROOT/'analysis/rc_cluster_membership_comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines[:23]))
    print('Overcapacity:',[(r['name'],r['scenario'],r['old_overcapacity'],r['new_overcapacity']) for r in records])


if __name__=='__main__':main()
