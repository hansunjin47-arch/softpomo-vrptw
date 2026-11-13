"""Descriptive fixed-fact/confidence correlations; no target routing feedback."""
import hashlib
import json
import math
from pathlib import Path
import statistics as st

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'result/confidence_features_20261111'
FOLDERS={
    'R101_weight100':ROOT/'result/r101_pair100_l5_mask_kim_cache_20261106',
    'RC101_weight100':ROOT/'result/rc101_pair100_l5_mask_kim_cache_20261111',
    'RC101_weight200':ROOT/'result/rc101_pair200_l5_mask_kim_cache_20261111',
}


def read(p):return json.loads(p.read_text())


def ranks(values):
    order=sorted(range(len(values)),key=lambda i:values[i])
    result=[0.]*len(values)
    i=0
    while i<len(order):
        j=i+1
        while j<len(order) and values[order[j]]==values[order[i]]:j+=1
        for pos in range(i,j):result[order[pos]]=(i+1+j)/2
        i=j
    return result


def pearson(x,y):
    assert len(x)==len(y)
    if len(x)<2:return None
    xx=[v-st.mean(x) for v in x];yy=[v-st.mean(y) for v in y]
    denominator=math.sqrt(sum(v*v for v in xx)*sum(v*v for v in yy))
    return sum(a*b for a,b in zip(xx,yy))/denominator if denominator else None


def correlations(x,y):return dict(pearson=pearson(x,y),spearman=pearson(ranks(x),ranks(y)))


def main():
    assert ranks([3,1,1,2])==[4.,1.5,1.5,3.]
    report=dict(scope='Fixed BASE facts only; no target RL metrics, API, GPU, Test evaluation, or training.',datasets={})
    all_hashes={}
    for name,folder in FOLDERS.items():
        source=None
        for path in sorted((folder/'calls').rglob('request.json')):
            if not(path.parent/'response.json').exists():continue
            request=read(path);ctx=json.loads(request['messages'][1]['content'])
            if not ctx['public_events'] and 'prior_base_confidence' not in ctx:
                source=path;facts=ctx['fixed_instance'];break
        assert source is not None
        used=[source,folder/'base_confidence.json',folder/'groups.json']
        all_hashes.update({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in used})
        scores={int(r['customer']):float(r['score']) for r in read(folder/'base_confidence.json')['confidence']}
        groups=read(folder/'groups.json')
        customers=[dict(zip(facts['customer_columns'],row)) for row in facts['customers']]
        assert len(customers)==101 and len({c['id'] for c in customers})==101
        depot=next(c for c in customers if c['id']==0)
        customers=sorted((c for c in customers if c['id']),key=lambda c:c['id'])
        assert [c['id'] for c in customers]==list(range(1,101))==sorted(scores)
        assert sorted(c for group in groups for c in group)==list(range(1,101))
        for c in customers:
            c['depot_distance']=math.hypot(c['x']-depot['x'],c['y']-depot['y'])
            c['tw_width']=round(c['close_min']-c['open_min'],4);c['confidence']=scores[c['id']]
        byid={c['id']:c for c in customers}
        features=('close_min','open_min','depot_distance','demand','service_min','tw_width')
        overall={f:correlations([c['confidence'] for c in customers],[c[f] for c in customers]) for f in features}
        within=[];group_rows=[]
        for i,ids in enumerate(groups):
            cs=[byid[n] for n in ids]
            within.append(dict(group=i,n=len(ids),correlations={f:correlations([c['confidence'] for c in cs],[c[f] for c in cs]) for f in features}))
            group_rows.append(dict(group=i,n=len(ids),mean_confidence=st.mean(c['confidence'] for c in cs),
                mean_close=st.mean(c['close_min'] for c in cs),earliest_close=min(c['close_min'] for c in cs)))
        within_summary={}
        for f in features:
            valid=[g['correlations'][f]['spearman'] for g in within if g['correlations'][f]['spearman'] is not None]
            within_summary[f]=dict(valid_groups=len(valid),mean_spearman=st.mean(valid) if valid else None,
                median_spearman=st.median(valid) if valid else None,
                positive_groups=sum(v>0 for v in valid),negative_groups=sum(v<0 for v in valid))
        report['datasets'][name]=dict(customer_count=100,group_count=len(groups),source_request=str(source),
            customer_level=overall,within_group_summary=within_summary,within_group=within,
            group_level={f:correlations([g['mean_confidence'] for g in group_rows],[g[f] for g in group_rows]) for f in ('mean_close','earliest_close','n')},
            group_rows=group_rows,customers=customers)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in all_hashes.items())
    report['input_hashes_unchanged']=True
    report['limitations']=['Correlations describe these fixed instances and cached outputs, not causal effects or evidence that a hand-crafted feature can replace LLM guidance.',
        'Within-group mean correlations summarize groups equally and are not pooled causal estimates. Constant features yield null correlation.',
        'Depot distance is Euclidean coordinate distance, not realized event-adjusted travel time.',
        'TW width is rounded to 0.0001 minutes to avoid interpreting float-export noise as real variability; all widths/service times here are constant.',
        'RC100/RC200 share geometry and differ in objective/reference/cache generation; they are not independent generalization samples.',
        'Spatial event-change analyses are reused from rc_kim_cache_comparison_20261111; no new event inference performed.']
    OUT.mkdir(exist_ok=True)
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    (OUT/'input_hashes.json').write_text(json.dumps(all_hashes,indent=2))
    print(json.dumps({k:{f:v[f] for f in ('customer_level','within_group_summary','group_level')} for k,v in report['datasets'].items()},indent=2))


if __name__=='__main__':main()
