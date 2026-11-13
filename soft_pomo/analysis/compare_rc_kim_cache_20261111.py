"""Read-only CPU cache comparison. No confidence change is a causal estimate."""
import collections
import difflib
import hashlib
import json
import math
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'result/rc_kim_cache_comparison_20261111'
OLD = ROOT/'result/pair200_l5_mask_kim_cache_20261105'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def flat(payload):
    return {int(x['customer']): float(x['score']) for x in payload['confidence']}


def describe(scores, groups):
    values = list(scores.values())
    constant = [g for g in groups if max(scores[n] for n in g)-min(scores[n] for n in g)<1e-9]
    return dict(mean=st.mean(values), sd=st.pstdev(values), minimum=min(values), maximum=max(values),
        zero_fraction=st.mean(x==0 for x in values), one_fraction=st.mean(x==1 for x in values),
        near_zero_fraction=st.mean(x<=.05 for x in values), near_one_fraction=st.mean(x>=.95 for x in values),
        constant_groups=len(constant), constant_group_customer_fraction=sum(map(len,constant))/len(values),
        within_group_sd_mean=st.mean(st.pstdev(scores[n] for n in g) for g in groups))


def contrast(a,b,groups,affected=None):
    delta={n:b[n]-a[n] for n in a}
    offsets={n:st.mean(delta[c] for c in g) for g in groups for n in g}
    residual={n:delta[n]-offsets[n] for n in a}
    order=collections.Counter()
    top_changed=0
    for g in groups:
        top_changed += ({n for n in g if a[n]==max(a[c] for c in g)} != {n for n in g if b[n]==max(b[c] for c in g)})
        for i,x in enumerate(g):
            for y in g[i+1:]:
                u,v=a[x]-a[y],b[x]-b[y]
                order['inverted' if u*v<0 else 'both_tied' if u==0 and v==0 else 'tie_changed' if u==0 or v==0 else 'same_order']+=1
    rms=lambda xs:math.sqrt(st.mean(x*x for x in xs))
    result=dict(mean_delta=st.mean(delta.values()), mean_abs_delta=st.mean(abs(x) for x in delta.values()),
        positive_changes=sum(x>1e-9 for x in delta.values()),negative_changes=sum(x < -1e-9 for x in delta.values()),zero_changes=sum(abs(x)<=1e-9 for x in delta.values()),
        rms_delta=rms(delta.values()),group_offset_rms=rms(offsets.values()),within_group_residual_rms=rms(residual.values()),
        changed_customers=sum(abs(x)>1e-9 for x in delta.values()), group_top_set_changed=top_changed,
        pair_order_counts=dict(order), customer_deltas=delta,group_offsets=offsets,residual_deltas=residual)
    if affected is not None:
        exposed=set(affected)&a.keys()
        unexposed=a.keys()-exposed
        result['spatial_exposure']={kind:dict(count=len(nodes), mean_delta=st.mean(delta[n] for n in nodes) if nodes else None,
            mean_abs_delta=st.mean(abs(delta[n]) for n in nodes) if nodes else None) for kind,nodes in [('affected',exposed),('outside',unexposed)]}
    return result


def requests(folder):
    result=[]
    for path in (folder/'calls').rglob('request.json'):
        if not(path.parent/'response.json').exists():continue
        q=read(path)
        result.append((q,json.loads(q['messages'][1]['content'])))
    return result


def main():
    OUT.mkdir(exist_ok=True)
    inputs=[OLD]+[ROOT/f'result/rc101_pair{w}_l5_mask_kim_cache_20261111' for w in (100,200)]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for d in inputs for p in d.rglob('*') if p.is_file()}
    report=dict(scope='RC Train cache content only; no API/Test/model inference/training',weights={},comparability={})
    snapshots={}
    for folder in inputs:
        groups=read(folder/'groups.json'); base=flat(read(folder/'base_confidence.json'))
        reqs=requests(folder)
        contexts={}
        for q,c in reqs:
            events=c.get('public_events',[])
            if events:contexts[json.dumps(events,sort_keys=True)]=c
        # Match by public reports from the immutable split data, not request order.
        from experiments.concurrent_fleet_bias_guided.support import load
        split=read(folder/'split.json')
        from experiments.concurrent_fleet_guided.prompts import PUBLIC_FIELDS
        rows={r['name']:r for r in split['train']}
        cases={}
        for p in sorted((folder/'train_cache').glob('*.json')):
            cache=read(p); inst,reports=load(rows[p.stem])
            affected=set(n for e in reports for n in e.get('affected_nodes',[]))
            event=flat(cache['post_disclosure'] or cache['initial'])
            cases[p.stem]=dict(kind='ACC' if '_ACC_' in p.stem else 'RAIN',scores=event,affected=sorted(affected),
                description=describe(event,groups),event_minus_base=contrast(base,event,groups,affected))
        settings={k:sorted({json.dumps(q.get(k,'UNSPECIFIED'),sort_keys=True) for q,_ in reqs}) for k in ('model','temperature','top_p','provider','max_tokens')}
        snapshots[str(folder.name)]=dict(base=base,base_description=describe(base,groups),cases=cases,request_settings=settings)
    for w in (100,200):
        new=inputs[w//100]; key=new.name; snap=snapshots[key]
        weight_report=dict(base=snap['base_description'],request_settings=snap['request_settings'],event_summary={})
        for kind in ('RAIN','ACC'):
            rs=[r for r in snap['cases'].values() if r['kind']==kind]
            weight_report['event_summary'][kind]={k:st.mean(r['event_minus_base'][k] for r in rs) for k in ('mean_delta','mean_abs_delta','group_offset_rms','within_group_residual_rms','changed_customers','group_top_set_changed')}
            weight_report['event_summary'][kind]['mean_score']=st.mean(r['description']['mean'] for r in rs)
            weight_report['event_summary'][kind]['change_sign_counts_total']={k:sum(r['event_minus_base'][k] for r in rs) for k in ('positive_changes','negative_changes','zero_changes')}
            weight_report['event_summary'][kind]['near_one_fraction']=st.mean(r['description']['near_one_fraction'] for r in rs)
            weight_report['event_summary'][kind]['constant_group_customer_fraction']=st.mean(r['description']['constant_group_customer_fraction'] for r in rs)
            weight_report['event_summary'][kind]['exposure']={exposure:{metric:st.mean(r['event_minus_base']['spatial_exposure'][exposure][metric] for r in rs) for metric in ('count','mean_delta','mean_abs_delta')} for exposure in ('affected','outside')}
        report['weights'][w]=weight_report
    new=inputs[2];a,b=snapshots[OLD.name],snapshots[new.name];groups=read(new/'groups.json')
    oldref,newref=read(OLD/'reference.json'),read(new/'reference.json')
    oldsys=requests(OLD)[0][0]['messages'][0]['content'];newsys=requests(new)[0][0]['messages'][0]['content']
    report['comparability']=dict(weight100='No matched old maskON weight100 lambda5 Kim cache found.',
        weight200={k:read(OLD/k)==read(new/k) for k in ('groups.json','split.json','config.json','request_template.json')},
        reference_fields_equal={k:oldref.get(k)==newref.get(k) for k in set(oldref)|set(newref)},
        request_settings_equal=a['request_settings']==b['request_settings'],
        prompt_diff=list(difflib.unified_diff(oldsys.splitlines(),newsys.splitlines(),lineterm='')),
        caveat='Fresh non-deterministic calls; reference name changed; event prompts also receive changed BASE. Not a controlled causal prompt-effect estimate.')
    paired={'BASE':contrast(a['base'],b['base'],groups)}
    conditional={}
    for name,record in b['cases'].items():paired[name]=contrast(a['cases'][name]['scores'],record['scores'],groups,record['affected'])
    for name,record in b['cases'].items():
        olddelta={n:a['cases'][name]['scores'][n]-a['base'][n] for n in a['base']}
        newdelta={n:record['scores'][n]-b['base'][n] for n in b['base']}
        conditional[name]=contrast(olddelta,newdelta,groups,record['affected'])
    report['weight200_paired_summary']={}
    for kind in ('BASE','RAIN','ACC'):
        rs=[v for k,v in paired.items() if k=='BASE'] if kind=='BASE' else [v for k,v in paired.items() if '_'+kind+'_' in k]
        report['weight200_paired_summary'][kind]={k:st.mean(r[k] for r in rs) for k in ('mean_delta','mean_abs_delta','group_offset_rms','within_group_residual_rms','changed_customers','group_top_set_changed')}
    report['weight200_change_in_event_minus_base']={kind:{k:st.mean(r[k] for name,r in conditional.items() if '_'+kind+'_' in name) for k in ('mean_delta','mean_abs_delta','group_offset_rms','within_group_residual_rms','changed_customers')} for kind in ('RAIN','ACC')}
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report['original_inputs_unchanged']=True
    report['limitations']=['Spatial exposure is descriptive, not an oracle for desired confidence sign.',
        'Guidance remains anchor-group only, no NEW or dispatch bias; content changes do not imply route/performance changes.',
        'Weight100 vs200 changes objective and reference policy, so cross-weight score differences are not prompt effects.']
    for name,obj in [('summary.json',report),('snapshots.json',snapshots),('paired200.json',paired),('conditional200.json',conditional),('input_hashes.json',hashes)]:
        (OUT/name).write_text(json.dumps(obj,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
