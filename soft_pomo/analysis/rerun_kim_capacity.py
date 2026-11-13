"""Recluster BASE instances only; preserve confidence caches and make no LLM calls."""
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from train_soft_cluster import SoftClusterTrainer
from vrptw_env import load_solomon


def main():
    old=ROOT/'result_soft/llm_cache_kim_initial_v1_20260917_214827'
    out=ROOT/'analysis/kim_capacity_rerun'/time.strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=False)
    trainer=SoftClusterTrainer.__new__(SoftClusterTrainer)
    rows=[]
    for family,lo,hi,prefix in [('c1',101,110,'C'),('rc1',101,109,'RC'),('r1',101,113,'R')]:
        for num in range(lo,hi):
            name=f'{prefix}{num}'
            source=ROOT.parent/f'data/Solomon/{name.lower()}.txt'
            inst=load_solomon(str(source))
            oldpath=old/family/f'{name}_cluster.json'
            digest=hashlib.sha256(oldpath.read_bytes()).hexdigest()
            data=json.loads(oldpath.read_text())
            before=[list(map(int,data[str(k)])) for k in range(len(data['_depot']))]
            with patch('train_soft_cluster.get_all_clusters_confidence',side_effect=AssertionError('No LLM calls')):
                clusters=trainer._cluster_customers(inst,max(1,math.ceil(float(inst['node_demand'].sum()))))
            loads=[sum(float(inst['node_demand'][n-1]) for n in ns) for ns in clusters]
            assert max(loads)<=1+1e-6
            assert sorted(n for ns in clusters for n in ns)==list(range(1,inst['n_customers']+1))
            assert len(clusters)<=min(25,inst['vehicle_limit'])
            assert hashlib.sha256(oldpath.read_bytes()).hexdigest()==digest
            row=dict(name=name,old_K=len(before),new_K=len(clusters),
                membership_changed=sorted(map(sorted,before))!=sorted(map(sorted,clusters)),
                exact_order_changed=before!=clusters,max_normalized_load=max(loads),
                capacity=float(inst['vehicle_capacity']),
                loads=[x*float(inst['vehicle_capacity']) for x in loads],
                old_cache_hash=digest,input_hash=hashlib.sha256(source.read_bytes()).hexdigest(),
                clusters=clusters)
            rows.append(row)
            (out/f'{name}_clusters.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
    result=dict(rows=rows,code_hash=hashlib.sha256((ROOT/'train_soft_cluster.py').read_bytes()).hexdigest(),
                llm_calls=0,capacity_tolerance=1e-6,
                note='Clustering artifacts only; not confidence caches. Existing soft-TW limit policy unchanged.')
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('OUTPUT',out)
    print('CHANGED',json.dumps([r for r in rows if r['exact_order_changed']]))
    print('VALIDATED',len(rows))


if __name__=='__main__': main()
